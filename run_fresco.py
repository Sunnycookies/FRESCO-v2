import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# In China, set this to use huggingface
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import cv2
import io
import gc
import yaml
import argparse
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler, ControlNetModel, DDIMScheduler

from src.utils import *
from src.keyframe_selection import get_keyframe_ind, get_keyframe_ind_extend
from src.diffusion_hacked import apply_FRESCO_attn, apply_FRESCO_opt, register_conv_control_efficient
from src.pipe_FRESCO import inference
from src.tokenflow_utils import *

def get_models(config):
    print('\n' + '=' * 100)
    print('creating models...')
    import sys
    sys.path.append("./src/ebsynth/deps/gmflow/")
    sys.path.append("./src/EGNet/")
    sys.path.append("./src/ControlNet/")
    
    from gmflow.gmflow import GMFlow
    from model import build_model
    from annotator.hed import HEDdetector
    from annotator.canny import CannyDetector
    from annotator.midas import MidasDetector

    # optical flow
    flow_model = GMFlow(feature_channels=128,
                   num_scales=1,
                   upsample_factor=8,
                   num_head=1,
                   attention_type='swin',
                   ffn_dim_expansion=4,
                   num_transformer_layers=6,
                   ).to('cuda')
    
    local_files_only = False
    
    checkpoint = torch.load(config['gmflow_path'], map_location=lambda storage, loc: storage)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    flow_model.load_state_dict(weights, strict=False)
    flow_model.eval() 
    print('create optical flow estimation model successfully!')
    
    # saliency detection
    sod_model = build_model('resnet')
    sod_model.load_state_dict(torch.load(config['sod_path']))
    sod_model.to("cuda").eval()
    print('create saliency detection model successfully!')
    
    # controlnet
    if config['controlnet_type'] not in ['hed', 'depth', 'canny']:
        print('unsupported control type, set to hed')
        config['controlnet_type'] = 'hed'
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-"+config['controlnet_type'], 
                                                 torch_dtype=torch.float16, local_files_only=local_files_only)
    controlnet.to("cuda") 
    if config['controlnet_type'] == 'depth':
        detector = MidasDetector()
    elif config['controlnet_type'] == 'canny':
        detector = CannyDetector()
    else:
        detector = HEDdetector()
    print('create controlnet model-' + config['controlnet_type'] + ' successfully!')
    
    # diffusion model
    if config['sd_path'] == 'stabilityai/stable-diffusion-2-1-base':
        pipe = StableDiffusionPipeline.from_pretrained(config['sd_path'], torch_dtype=torch.float16, local_files_only=local_files_only)
    else:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16, local_files_only=local_files_only)
        pipe = StableDiffusionPipeline.from_pretrained(config['sd_path'], vae=vae, torch_dtype=torch.float16, local_files_only=local_files_only)
    if config['edit_mode'] == 'SDEdit':
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DDIMScheduler.from_pretrained(config['sd_path'], subfolder="scheduler", local_files_only=local_files_only)
    pipe.to("cuda")
    pipe.scheduler.set_timesteps(config['num_inference_steps'], device=pipe._execution_device)
    
    if config['use_freeu']:
        from src.free_lunch_utils import apply_freeu
        apply_freeu(pipe, b1=1.2, b2=1.5, s1=1.0, s2=1.0)

    frescoProc = apply_FRESCO_attn(pipe, config['use_inversion'], True, config['edit_mode'])
    frescoProc.controller.disable_controller()
    apply_FRESCO_opt(pipe, use_inversion=config['use_inversion'])
    print('create diffusion model ' + config['sd_path'] + ' successfully!')
    
    for param in flow_model.parameters():
        param.requires_grad = False    
    for param in sod_model.parameters():
        param.requires_grad = False
    for param in controlnet.parameters():
        param.requires_grad = False
    for param in pipe.unet.parameters():
        param.requires_grad = False
    
    return pipe, frescoProc, controlnet, detector, flow_model, sod_model

def apply_control(x, detector, config):
    if config['controlnet_type'] == 'depth':
        detected_map, _ = detector(x)
    elif config['controlnet_type'] == 'canny':
        detected_map = detector(x, 50, 100)
    else:
        detected_map = detector(x)
    return detected_map

@torch.autocast(dtype=torch.float16, device_type='cuda')
def run_keyframe_translation(config, run_ebsynth, run_tokenflow):
    pipe, frescoProc, controlnet, detector, flow_model, sod_model = get_models(config)
    device = pipe._execution_device
    guidance_scale = 7.5
    do_classifier_free_guidance = guidance_scale > 1
    assert(do_classifier_free_guidance)
    timesteps = pipe.scheduler.timesteps
    cond_scale = [config['cond_scale']] * config['num_inference_steps']
    dilate = Dilate(device=device)
    
    base_prompt = config['prompt']
    if 'n_prompt' in config and 'a_prompt' in config:
        a_prompt = config['a_prompt']
        n_prompt = config['n_prompt']
    elif 'Realistic' in config['sd_path'] or 'realistic' in config['sd_path']:
        a_prompt = ', RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3, '
        n_prompt = '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation'
    else:
        a_prompt = ', best quality, extremely detailed, '
        n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing finger, extra digit, fewer digits, cropped, worst quality, low quality'    

    print('\n' + '=' * 100)
    print('key frame selection for \"%s\"...'%(config['file_path']))
    
    video_cap = cv2.VideoCapture(config['file_path'])
    frame_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if config['primary_select']:
        primary_indexes = get_keyframe_ind(config['file_path'], mininterv=config['mininterv'], maxinterv=config['maxinterv'])
        print(f'choose primary indexes {primary_indexes}')
    else:
        primary_indexes = list(range(frame_num))
    primary_indexes_posmap = {primary_indexes[i]:i for i in range(len(primary_indexes))}
    keys = get_keyframe_ind_extend(config['file_path'], config['keyframe_select_mode'], config['keyframe_select_radix'], 
                                   primary_indexes, config['mininterv'], config['maxinterv'])
    sublists_all = []
    for ind in range(len(keys)):
        sublists = [keys[ind][i:i+config['batch_size']-2] for i in range(2, len(keys[ind]), config['batch_size']-2)]
        sublists[0].insert(0, keys[ind][0])
        sublists[0].insert(1, keys[ind][1])
        while len(sublists_all) < len(sublists):
            sublists_all.append([])
        for batch_ind, keys_batch in enumerate(sublists):
            sublists_all[batch_ind].append(keys_batch)
    print(f"split keyframes into groups {sublists_all}")

    keylists = []
    for keys_group in sublists_all:
        keys_all = []
        for key in keys_group:
            keys_all += key
        # keylists.append(list(np.unique(keys_all)))
        keylists.append(keys_all)
    print(f"split keyframes into batches {keylists}")

    # you can set extra_prompts for individual keyframe
    # for example, extra_prompts[38] = ', closed eyes' to specify the person frame38 closes the eyes
    extra_prompts = [''] * frame_num

    os.makedirs(config['save_path'], exist_ok=True)
    if os.path.exists(os.path.join(config['save_path'], 'keys')):
        os.system(f"rm -rf {os.path.join(config['save_path'], 'keys')}")
    os.makedirs(os.path.join(config['save_path'], 'keys'))
    if run_ebsynth:
        os.makedirs(os.path.join(config['save_path'], 'video'), exist_ok=True)

    if config['edit_mode']=='pnp':
        pnp_attn_t = int(config["num_inference_steps"] * config["pnp_attn_t"])
        qk_injection_timesteps = pipe.scheduler.timesteps[:pnp_attn_t] if pnp_attn_t >= 0 else []
        frescoProc.controller.set_qk_injection_timesteps(qk_injection_timesteps)
        
        pnp_f_t = int(config["num_inference_steps"] * config["pnp_f_t"])
        conv_injection_timesteps = pipe.scheduler.timesteps[:pnp_f_t] if pnp_f_t >= 0 else []
        register_conv_control_efficient(pipe, conv_injection_timesteps)

    gc.collect()
    torch.cuda.empty_cache()
    
    print('\n' + '=' * 100)
    print('video to video translation...')

    batch_ind = 0
    imgs = []
    img_idx = []
    record_latent = []
    video_cap = cv2.VideoCapture(config['file_path'])
    for i in range(frame_num):
        success, frame = video_cap.read()
        if success == False:
            print(f"{'/' * 100}\nAn error occurred when reading frame from {config['file_path']} frame {i}\n{'/' * 100}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = resize_image(frame, 512)
        if run_ebsynth:
            Image.fromarray(img).save(os.path.join(config['save_path'], 'video/%04d.png'%(i)))
        
        if i not in primary_indexes:
            continue

        if i not in keylists[batch_ind] and not run_tokenflow:
            continue

        imgs += [img]
        img_idx += [i]
        
        if batch_ind < len(keylists) - 1:
            if i != keylists[batch_ind][-1]:
                continue
        elif run_tokenflow:
            if i != primary_indexes[-1]:
                continue
        elif i != keylists[batch_ind][-1]:
            continue

        print(f"processing batch [{batch_ind + 1}/{len(keylists)}] with images {img_idx}")

        propagation_mode = batch_ind > 0

        prompts = [base_prompt + a_prompt + extra_prompts[ind] for ind in img_idx]
        if propagation_mode:
            if run_tokenflow:
                assert len(img_idx) == primary_indexes_posmap[img_idx[-1]] - primary_indexes_posmap[keylists[batch_ind - 1][-1]] + 2
            else:
                assert len(img_idx) == len(keylists[batch_ind]) + 2
            prompts = ref_prompts + prompts
        inv_prompts = [config['inv_prompt']] * len(img_idx)

        edges = torch.cat([numpy2tensor(apply_control(img, detector, config)[:, :, None]) for img in imgs], dim=0)
        edges = edges.repeat(1,3,1,1).cuda() * 0.5 + 0.5
        if do_classifier_free_guidance:
            edges = torch.cat([edges.to(pipe.unet.dtype)] * 2)

        pos_map = {img_idx[i]:i for i in range(len(img_idx))}
        prefix = [0, 1] if propagation_mode else []
        keylists_pos = [prefix + [pos_map[key] for key in keygroup] + ([pos_map[img_idx[-1]]] if img_idx[-1] not in keygroup else [])
                        for keygroup in sublists_all[batch_ind]]

        print(f"keyframe indexes of images {sublists_all[batch_ind]}")
        print(f"keyframe indexes of position {keylists_pos}")

        gc.collect()
        torch.cuda.empty_cache()

        latents = inference(pipe, controlnet, frescoProc, imgs, edges, timesteps, img_idx, keylists_pos, n_prompt, 
                            prompts, inv_prompts, config['inv_latent_path'], config['temp_paras_save_path'], 
                            config['end_opt_step'], propagation_mode, False, config['use_fresco'], do_classifier_free_guidance, 
                            run_tokenflow, config['edit_mode'], False, config['use_controlnet'], config['use_saliency'], 
                            config['use_inversion'], cond_scale, config['num_inference_steps'], config['num_warmup_steps'], 
                            config['seed'], guidance_scale, record_latent, config['num_intraattn_steps'], 
                            flow_model=flow_model, sod_model=sod_model, dilate=dilate)

        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            start = 2 if propagation_mode else 0
            size = len(latents)
            image = []
            for i in range(start, size, config['batch_size']):
                end = min(size, i + config['batch_size'])
                image_batch = pipe.vae.decode(latents[i:end] / pipe.vae.config.scaling_factor, return_dict=False)[0]
                image.append(image_batch.cpu())
            image = torch.cat(image)
            image = torch.clamp(image, -1, 1)
            save_imgs = tensor2numpy(image)
            for ind, num in enumerate(img_idx[start:]):
                Image.fromarray(save_imgs[ind]).save(os.path.join(config['save_path'], 'keys/%04d.png'%(num)))

        gc.collect()
        torch.cuda.empty_cache()

        batch_ind += 1
        imgs = [imgs[0], imgs[-1]]
        img_idx = [img_idx[0], img_idx[-1]]
        ref_prompts = [prompts[0], prompts[-1]]
        if batch_ind == len(keylists):
            break
    
    if run_tokenflow:
        keys = primary_indexes
    else:
        keys = keys[config['num_inference_steps'] % len(keys)]

    gc.collect()
    torch.cuda.empty_cache()

    return keys

def run_full_video_translation(config, keys, run_ebsynth):
    gc.collect()
    torch.cuda.empty_cache()

    print('\n' + '=' * 100)
    video_cap = cv2.VideoCapture(config['file_path'])
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    video_name = config['file_path'].split('/')[-1]
    video_name = video_name.split('-')[0]
    video_name = video_name.split('_')[0]
    video_name += f"_{config['edit_mode']}_{config['synth_mode']}_{config['keyframe_select_mode']}"
    if not config['use_fresco']:
        video_name += '_no-fresco'
    n_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(keys) == n_frames:
        cmd = f"python frames2video.py -r {config['save_path']} -o {config['save_path']} -n {video_name} -f {fps}"
        print('\n```')
        print(cmd)
        print('\n```')

        os.system(cmd)

        print('\n' + '=' * 100)
        print('Done')
        return
    else:
        print('translating full video with:')

    o_video = os.path.join(config['save_path'], f'{video_name}.mp4')
    max_process = config['max_process']
    key_ind = io.StringIO()
    for k in keys:
        print('%d'%(k), end=' ', file=key_ind)
    cmd = (
        f"python video_blend.py {config['save_path']} --key keys "
        f"--key_ind {key_ind.getvalue()} --output {o_video} --fps {fps} "
        f"--n_proc {max_process} -ps")
    
    if run_ebsynth:
        print('\n```')
        print(cmd)
        print('```')
        os.system(cmd)
    
    print('\n' + '=' * 100)
    print('Done')
    
def check_config(config):
    run_ebsynth = config['synth_mode'] in ['ebsynth', 'Mixed'] 
    run_tokenflow = config['synth_mode'] in ['Tokenflow', 'Mixed'] 
    
    # use inversion only when using pnp or tokenflow
    assert(not config['use_inversion'] or config['edit_mode'] == 'pnp' or run_tokenflow)
    
    # only using ebsynth method requires key frame indexes to be fixed
    assert(run_tokenflow or config['keyframe_select_mode'] == 'fixed')
    
    # synth_mode 'None' means every frame is key frame
    assert(config['synth_mode'] != 'None' or config['maxinterv'] == config['mininterv'] == 1)
    
    # maxinterv and miniterv will be used when key frame indexes are fixed
    assert(config['keyframe_select_mode'] == 'loop' or config['mininterv'] <= config['maxinterv'])
    
    # use two levels of key frame selection only when using Mixed method
    assert(not config['primary_select'] or config['synth_mode'] == 'Mixed')
    
    # use controlnet only when using SDEdit method
    assert(not config['use_controlnet'] or config['edit_mode'] == 'SDEdit')
    
    # use FRESCO spatial-guided attention only when using SDEdit method
    assert(config['num_intraattn_steps'] == 0 or config['edit_mode'] == 'SDEdit')

    return run_ebsynth, run_tokenflow    

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
        
    # check if configurations are valid
    run_ebsynth, run_tokenflow = check_config(config)
    
    # run video2video translation
    keys = run_keyframe_translation(config, run_ebsynth, run_tokenflow)
    run_full_video_translation(config, keys, run_ebsynth)
