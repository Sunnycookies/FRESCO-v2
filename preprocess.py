import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler,ControlNetModel,StableDiffusionPipeline

from tqdm import tqdm, trange
import yaml
import torch
import torch.nn as nn
import argparse
from torchvision.io import write_video
from pathlib import Path
from PIL import Image
from src.utils import *
import torchvision.transforms as T
import random


class Preprocess(nn.Module):
    def __init__(self, device, config) -> None:
        super().__init__()
        self.device = device
        
        self.use_depth = False

        self.sd_version = config['sd_path']
        self.use_controlnet = config['use_controlnet']
        # self.save_path = os.path.join(config['inv_save_path'])


        import sys
        sys.path.append("./src/ebsynth/deps/gmflow/")
        sys.path.append("./src/EGNet/")
        sys.path.append("./src/ControlNet/")

        from gmflow.gmflow import GMFlow
        from model import build_model
        from annotator.hed import HEDdetector
        from annotator.canny import CannyDetector
        from annotator.midas import MidasDetector


        # controlnet
        if config['controlnet_type'] not in ['hed', 'depth', 'canny']:
            print('unsupported control type, set to hed')
            config['controlnet_type'] = 'hed'
        self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-"+config['controlnet_type'], 
                                                     torch_dtype=torch.float16)
        self.controlnet.to("cuda") 
        if config['controlnet_type'] == 'depth':
            self.detector = MidasDetector()
        elif config['controlnet_type'] == 'canny':
            self.detector = CannyDetector()
        else:
            self.detector = HEDdetector()
        print('create controlnet model-' + config['controlnet_type'] + ' successfully!')

        if config['sd_path'] == 'stabilityai/stable-diffusion-2-1-base':
            self.pipe = StableDiffusionPipeline.from_pretrained(config['sd_path'], torch_dtype=torch.float16, local_files_only=False)
        else:
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16, local_files_only=False)
            self.pipe = StableDiffusionPipeline.from_pretrained(config['sd_path'], vae=self.vae, torch_dtype=torch.float16, local_files_only=False)
        # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        self.unet = self.pipe.unet
        self.scheduler = DDIMScheduler.from_pretrained(config['sd_path'] ,subfolder="scheduler", local_files_only=True)
        #noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        self.pipe.to("cuda")
        self.scheduler.set_timesteps(config['inv_inference_steps'], device=self.pipe._execution_device)
        

        for param in self.controlnet.parameters():
            param.requires_grad = False
        for param in self.pipe.unet.parameters():
            param.requires_grad = False



    def add_control(self, x, detector, config):
        if config['controlnet_type'] == 'depth':
            detected_map, _ = detector(x)
        elif config['controlnet_type'] == 'canny':
            detected_map = detector(x, 50, 100)
        else:
            detected_map = detector(x)
        return detected_map
    
    def controlnet_pred(self, latent_model_input, t, text_embed_input, controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            controlnet_cond=controlnet_cond,
            conditioning_scale=1,
            return_dict=False,
        )
        
        # apply the denoising network
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embed_input,
            cross_attention_kwargs={},
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]
        return noise_pred

    @torch.no_grad()
    def ddim_inversion(self, cond, latent_frames, save_path, batch_size, save_latents=True, control_edge = None, timesteps_to_save=None):
        timesteps = reversed(self.scheduler.timesteps)
        timesteps_to_save = timesteps_to_save if timesteps_to_save is not None else timesteps
        for i, t in enumerate(tqdm(timesteps)):
            for b in range(0, latent_frames.shape[0], batch_size):
                x_batch = latent_frames[b:b + batch_size]
                model_input = x_batch
                cond_batch = cond.repeat(x_batch.shape[0], 1, 1)
              
                                                                    
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.pipe.unet(model_input, t, encoder_hidden_states=cond_batch).sample if not self.use_controlnet \
                    else self.controlnet_pred(x_batch, t, cond_batch, torch.cat([control_edge[b: b + batch_size]]))
                pred_x0 = (x_batch - sigma_prev * eps) / mu_prev
                latent_frames[b:b + batch_size] = mu * pred_x0 + sigma * eps

            if save_latents and t in timesteps_to_save:
                # print(save_latents,t.item())
                torch.save(latent_frames, os.path.join(save_path, 'latents', f'noisy_latents_{t}.pt'))
        print(latent_frames.shape)
        torch.save(latent_frames, os.path.join(save_path, 'latents', f'noisy_latents_{t}.pt'))
        return latent_frames
    
    def run_inversion(self,config):

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        random.seed(1)
        np.random.seed(1)

        if not os.path.exists(os.path.join(config['inv_save_path'],'latents')):
            os.makedirs(os.path.join(config['inv_save_path'],'latents'))

        device = self.pipe._execution_device
        guidance_scale = 7.5
        do_classifier_free_guidance = guidance_scale > 1
        assert(do_classifier_free_guidance)
        timesteps = self.pipe.scheduler.timesteps
        cond_scale = [config['cond_scale']] * config['num_inference_steps']

        video_cap = cv2.VideoCapture(config['file_path'])
        frame_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        inv_prompt = config['inv_prompt']
        extra_prompts = [''] * frame_num

        print(inv_prompt)

        prompt_embeds = self.pipe._encode_prompt(
            inv_prompt,
            device,
            1,
            do_classifier_free_guidance,
            negative_prompt='',
        )[0]


        imgs = []
        for i in range(frame_num):  
            success, frame = video_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = resize_image(frame, 512)
            imgs += [img]
        imgs_torch = torch.cat([numpy2tensor(img) for img in imgs], dim=0).to(torch.float16)
        # latents = self.encode_imgs(imgs, deterministic=True).to(torch.float16).to(self.device)
        # B, C, H, W = imgs.shape
        with torch.no_grad():
            latents = torch.cat([self.pipe.vae.config.scaling_factor * self.pipe.vae.encode(imgs_torch[i:i+config['inv_batch_size']]).latent_dist.sample() for i in range(0, frame_num, config['inv_batch_size'])])
            latents.to(torch.float16)
           
        

        edges = torch.cat([numpy2tensor(self.add_control(img, self.detector, config)[:, :, None]) for img in imgs], dim=0)
        edges = edges.repeat(1,3,1,1).cuda() * 0.5 + 0.5
        
        edges = edges.to(self.pipe.unet.dtype)

        self.ddim_inversion(cond=prompt_embeds,
                            latent_frames=latents,
                            save_path=config['inv_save_path'],
                            batch_size=config['inv_batch_size'],
                            save_latents=True,
                            control_edge=edges,
                            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, 
                        default='config/config_bread.yaml',
                        help='The configuration file.')
    opt = parser.parse_args()

    print('=' * 100)
    print('loading configuration...')
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
        
    for name, value in sorted(config.items()):
        print('%s: %s' % (str(name), str(value)))  
    preprocess = Preprocess('cuda',config)
    preprocess.run_inversion(config)

    