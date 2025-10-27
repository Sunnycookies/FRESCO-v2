from src.utils import *
from src.flow_utils import warp_tensor
import torch
import torchvision
import gc
import glob
from diffusers.utils import torch_utils

from src.diffusion_hacked import *
from src.tokenflow_utils import *
"""
==========================================================================
* step(): one DDPM step with background smoothing 
* step_ddim(): one DDIM step without background smoothing
* denoise_step(): universal step_fn interface
* prepare_parameters(): prepare parameters for key frame editing and tokenflow (optional)
* inference(): translate one batch with FRESCO
==========================================================================
"""

def step(pipe, model_output, timestep, sample, generator, repeat_noise=False, 
         visualize_pipeline=False, flows=None, occs=None, saliency=None):
    """
    DDPM step with background smoothing
    * background smoothing: warp the background region of the previous frame to the current frame
    """
    scheduler = pipe.scheduler
    # 1. get previous step value (=t-1)
    prev_timestep = scheduler.previous_timestep(timestep)

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.one

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t    
    
    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    """
    [HACK] add background smoothing
    decode the feature
    warp the feature of f_{i-1}
    fuse the warped f_{i-1} with f_{i} in the non-salient region (i.e., background)
    encode the fused feature
    """
    if saliency is not None and flows is not None and occs is not None:
        image = pipe.vae.decode(pred_original_sample / pipe.vae.config.scaling_factor).sample 
        image = warp_tensor(image, flows, occs, saliency, unet_chunk_size=1)
        pred_original_sample = pipe.vae.config.scaling_factor * pipe.vae.encode(image).latent_dist.sample()    
    
    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t    
    
    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample    
    
    
    variance = beta_prod_t_prev / beta_prod_t * current_beta_t
    variance = torch.clamp(variance, min=1e-20)
    variance = (variance ** 0.5) * torch.randn(model_output.shape, generator=generator, 
                                               device=model_output.device, dtype=model_output.dtype)
    """
    [HACK] background smoothing
    applying the same noise could be good for static background
    """    
    if repeat_noise:
        variance = variance[0:1].repeat(model_output.shape[0],1,1,1)
        
    if visualize_pipeline: # for debug
        image = pipe.vae.decode(pred_original_sample / pipe.vae.config.scaling_factor).sample 
        viz = torchvision.utils.make_grid(torch.clamp(image, -1, 1), image.shape[0], 1)
        visualize(viz.cpu(), 90)

    pred_prev_sample = pred_prev_sample + variance
    
    return (pred_prev_sample, pred_original_sample)

def step_ddim(pipe, model_output, timestep, sample, eta = 0.0, use_clipped_model_output = False, generator = None, repeat_noise = False,
              variance_noise = None, return_dict = True, visualize_pipeline = False, flows = None, occs = None, saliency = None) :
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            eta (`float`): weight of noise for added noise in diffusion step.
            use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output` from the clipped
                predicted original sample. Necessary because predicted original sample is clipped to [-1, 1] when
                `self.config.clip_sample` is `True`. If no clipping has happened, "corrected" `model_output` would
                coincide with the one provided as input and `use_clipped_model_output` will have not effect.
            generator: random number generator.
            variance_noise (`torch.FloatTensor`): instead of generating noise for the variance using `generator`, we
                can directly provide the noise for the variance itself. This is useful for methods such as
                CycleDiffusion. (https://arxiv.org/abs/2210.05559)
            return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        scheduler = pipe.scheduler

        if scheduler.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf


        # if self.config.prediction_type == "epsilon":
        #     pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        #     pred_epsilon = model_output
        # elif self.config.prediction_type == "sample":
        #     pred_original_sample = model_output
        #     pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        # elif self.config.prediction_type == "v_prediction":
        #     pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        #     pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        # else:
        #     raise ValueError(
        #         f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
        #         " `v_prediction`"
        #     )

        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output

        # BackGround smoothing in DDIM will cause background distortion. temporarily disabled.
        """
        [HACK] add background smoothing
        decode the feature
        warp the feature of f_{i-1}
        fuse the warped f_{i-1} with f_{i} in the non-salient region (i.e., background)
        encode the fused feature
        """
        # if saliency is not None and flows is not None and occs is not None:
        #     print('s')
        #     image = pipe.vae.decode(pred_original_sample / pipe.vae.config.scaling_factor).sample 
        #     image = warp_tensor(image, flows, occs, saliency, unet_chunk_size=1)
        #     pred_original_sample = pipe.vae.config.scaling_factor * pipe.vae.encode(image).latent_dist.sample()  

        # 4. Clip or threshold "predicted x_0"

        # if self.config.thresholding:
        #     pred_original_sample = self._threshold_sample(pred_original_sample)
        # elif self.config.clip_sample:
        #     pred_original_sample = pred_original_sample.clamp(
        #         -self.config.clip_sample_range, self.config.clip_sample_range
        #     )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = scheduler._get_variance(timestep, prev_timestep)
        
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = torch_utils.randn_tensor(model_output.shape, generator=generator, 
                                                          device=model_output.device, dtype=model_output.dtype
                )

            # if repeat_noise:
            #     variance = variance[0:1].repeat(model_output.shape[0],1,1,1)
        
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance


        if visualize_pipeline: # for debug
            image = pipe.vae.decode(pred_original_sample / pipe.vae.config.scaling_factor).sample 
            viz = torchvision.utils.make_grid(torch.clamp(image, -1, 1), image.shape[0], 1)
            visualize(viz.cpu(), 90)

        if not return_dict:
            return (prev_sample,)

        return (prev_sample, pred_original_sample)

def denoise_step(pipe, model_output, timestep, sample, generator, repeat_nosie = False,
                 visualize_pipeline=False, flows=None, occs=None, saliency=None, ddim = False):
    step_fn = step_ddim if ddim else step
    return step_fn(pipe=pipe, model_output=model_output, timestep=timestep, sample=sample, generator=generator, 
                   repeat_noise=repeat_nosie, visualize_pipeline=visualize_pipeline, flows = flows, occs = occs, saliency=saliency)

@torch.autocast(dtype=torch.float16, device_type='cuda')
@torch.no_grad()    
def prepare_parameters(pipe, frescoProc, device, imgs, imgs_torch, keylists, prompts, n_prompt, edit_mode, latent_len, propagation_mode, 
                       use_tokenflow, use_saliency, paras_save_path, seed, do_classifier_free_guidance, sod_model, flow_model, dilate):
    # prepare parameters
    print("preparing parameters...")

    if os.path.exists(paras_save_path):
        os.system(f"rm -rf {paras_save_path}")
    image_edit_paras_path = os.path.join(paras_save_path, 'image_edit')
    os.makedirs(image_edit_paras_path)
    
    if use_tokenflow:
        tokenflow_paras_path = os.path.join(paras_save_path, 'tokenflow')
        os.makedirs(tokenflow_paras_path)
        tokenflow_decoder_attn_store_path = os.path.join(paras_save_path, 'tokenflow_decoder_attn')
        os.makedirs(tokenflow_decoder_attn_store_path)
    else:
        tokenflow_paras_path = None

    frescoProc.controller.enable_intraattn(True)
    if use_tokenflow:
        deactivate_tokenflow(pipe.unet)
        frescoProc.controller.set_tokenflow_store_path(tokenflow_decoder_attn_store_path)

    for ind, keygroup in enumerate(keylists):
        imgs_group = [imgs[i] for i in keygroup]
        imgs_group_torch = imgs_torch[keygroup]
        prompt_embeds_group = pipe._encode_prompt(
            [prompts[i] for i in keygroup],
            device,
            1,
            do_classifier_free_guidance,
            [n_prompt] * len(keygroup)
        )

        saliency_group = get_saliency(imgs_group, sod_model, dilate) if use_saliency else None

        flows_group, occs_group, attn_mask_group, interattn_paras_group \
            = get_flow_and_interframe_paras(flow_model, imgs_group)
        
        if edit_mode == 'pnp':
            correlation_matrix_group = []
        else:
            correlation_matrix_group = get_intraframe_paras(pipe, imgs_group_torch, frescoProc, prompt_embeds_group,
                                                            do_classifier_free_guidance, seed, False, ind)
        
        paras = (saliency_group, flows_group, occs_group, attn_mask_group, 
                 interattn_paras_group, correlation_matrix_group)
        torch.save(paras, os.path.join(image_edit_paras_path, f"paras_{ind}.pt"))

        if use_tokenflow:
            frescoProc.controller.enable_tokenflow()
            for j, key in enumerate(keygroup):
                end = latent_len if key == keygroup[-1] else keygroup[j + 1]
                if propagation_mode and key == 0:
                    end = 1
                
                imgs_tokenflow = imgs[key:end]
                imgs_tokenflow_torch = imgs_torch[key:end]
                prompt_embeds_tokenflow = pipe._encode_prompt(
                    prompts[key:end],
                    device,
                    1,
                    do_classifier_free_guidance,
                    [n_prompt] * (end - key)
                )

                saliency_tokenflow = get_saliency(imgs_tokenflow, sod_model, dilate) if use_saliency else None

                if end - key > 1:
                    flows_tokenflow, occs_tokenflow, attn_mask_tokenflow, interattn_paras_tokenflow \
                        = get_flow_and_interframe_paras(flow_model, imgs_tokenflow)
                    
                    if edit_mode == 'pnp':
                        correlation_matrix_tokenflow = []
                    else:
                        correlation_matrix_tokenflow = get_intraframe_paras(pipe, imgs_tokenflow_torch, frescoProc, prompt_embeds_tokenflow,
                                                                            do_classifier_free_guidance, seed, False, ind)
                else:
                    flows_tokenflow, occs_tokenflow, attn_mask_tokenflow, interattn_paras_tokenflow, correlation_matrix_tokenflow, \
                            = None, None, None, None, []

                paras_tokenflow = (saliency_tokenflow, flows_tokenflow, occs_tokenflow, attn_mask_tokenflow, 
                                   interattn_paras_tokenflow, correlation_matrix_tokenflow)
                torch.save(paras_tokenflow, os.path.join(tokenflow_paras_path, f"paras_{ind}_{j}.pt"))

            frescoProc.controller.disable_tokenflow()

    if use_tokenflow:
        set_tokenflow(pipe.unet, do_classifier_free_guidance)
    frescoProc.controller.disable_intraattn(False)

    print("parameters prepared!")
    
    return image_edit_paras_path, tokenflow_paras_path

@torch.autocast(dtype=torch.float16, device_type='cuda')
@torch.no_grad()
def inference(pipe, controlnet, frescoProc, imgs, edges, timesteps, img_idxs, keylists, n_prompt, prompts, inv_prompts, 
              inv_latent_path, paras_save_path, end_opt_step, propagation_mode, repeat_noise = False, use_fresco = True, 
              do_classifier_free_guidance = True, use_tokenflow = False, edit_mode = 'SDEdit', visualize_pipeline = False, 
              use_controlnet = True, use_saliency = False, use_inversion = False, cond_scale = [0.7] * 20, num_inference_steps = 20, 
              num_warmup_steps = 6, seed = 0, guidance_scale = 7.5, record_latents = [], num_intraattn_steps = 1, 
              step_interattn_end = 350, bg_smoothing_steps = [16, 17], flow_model = None, sod_model = None, dilate = None):

    gc.collect()
    torch.cuda.empty_cache()

    device = pipe._execution_device
    noise_scheduler = pipe.scheduler 
    generator = torch.Generator(device=device).manual_seed(seed)

    imgs_torch = torch.cat([numpy2tensor(img) for img in imgs], dim=0)

    # calculate for prompt_embed.dtype
    prompt_embeds = pipe._encode_prompt(
        prompts,
        device,
        1,
        do_classifier_free_guidance,
        [n_prompt] * len(prompts)
    )

    # prepate initial latents (noise)
    if edit_mode == 'pnp':
        noisest = max([int(x.split('_')[-1].split('.')[0]) for x in glob.glob(os.path.join(inv_latent_path, f'noisy_latents_*.pt'))])
        latents_path = os.path.join(inv_latent_path, f'noisy_latents_{noisest}.pt')
        latents = torch.load(latents_path)[img_idxs]
    else:
        B, _, H, W = imgs_torch.shape
        latents = pipe.prepare_latents(
            B,
            pipe.unet.config.in_channels,
            H,
            W,
            prompt_embeds.dtype,
            device,
            generator,
            latents = None,
        )
    
    if repeat_noise:
        latents = latents[0:1].repeat(B,1,1,1).detach()

    batch_size = 8
    latent_x0 = []
    for i in range(0, len(imgs_torch), batch_size):
        end = min(len(imgs_torch), i + batch_size)
        latent_x0_batch = pipe.vae.config.scaling_factor * pipe.vae.encode(imgs_torch[i:end].to(pipe.unet.dtype)).latent_dist.sample()
        latent_x0.append(latent_x0_batch)
    latent_x0 = torch.cat(latent_x0)

    if edit_mode == 'pnp':
        alpha_prod_T = noise_scheduler.alphas_cumprod[noisest]
        mu_T, sigma_T = alpha_prod_T ** 0.5, (1 - alpha_prod_T) ** 0.5
        latents = (latents - mu_T * latent_x0) / sigma_T
        latents_init = noise_scheduler.add_noise(latent_x0, latents, timesteps[num_warmup_steps]).detach()
    elif num_warmup_steps <= 0:
        latents_init = latents.detach()
        num_warmup_steps = 0
    else:
        latents_init = noise_scheduler.add_noise(latent_x0, latents, timesteps[num_warmup_steps]).detach()
    
    del latent_x0

    keys_loop_len = len(keylists)

    if use_fresco:
        (
            image_edit_paras_path,
            tokenflow_paras_path
        ) = prepare_parameters(pipe, frescoProc, device, imgs, imgs_torch, keylists, prompts, n_prompt, edit_mode, len(latents_init), propagation_mode, 
                               use_tokenflow, use_saliency, paras_save_path, seed, do_classifier_free_guidance, sod_model, flow_model, dilate)

    gc.collect()
    torch.cuda.empty_cache()

    with pipe.progress_bar(total=num_inference_steps - num_warmup_steps) as progress_bar:
        latents = latents_init
        for i, t in enumerate(timesteps[num_warmup_steps:]):
            i_ = i % keys_loop_len
            keygroup = keylists[i_]

            # prepare a group of frame based on keygroup
            # print(f"processing keygroup [{i_ + 1}/{len(keylists)}] with keyframes {keygroup}")

            if use_inversion:
                ref_inv_latent = load_source_latents_t(t, inv_latent_path)[img_idxs]

            # Turn on FRESCO support
            if use_fresco:
                paras = torch.load(os.path.join(image_edit_paras_path, f"paras_{i_}.pt"))

                saliency, flows, occs, attn_mask, interattn_paras, correlation_matrix = paras

                '''
                Flexible settings for attention:
                * Turn off FRESCO-guided attention: frescoProc.controller.disable_controller() 
                Then you can turn on one specific attention submodule
                * Turn on Cross-frame attention: frescoProc.controller.enable_cfattn() 
                * Turn on Spatial-guided attention: frescoProc.controller.enable_intraattn() 
                * Turn on Temporal-guided attention: frescoProc.controller.enable_interattn()

                Flexible settings for optimization:
                * Turn off Spatial-guided optimization: set optimize_temporal = False in apply_FRESCO_opt()
                * Turn off Temporal-guided optimization: set correlation_matrix = [] in apply_FRESCO_opt()
                * Turn off FRESCO-guided optimization: disable_FRESCO_opt(pipe)

                Flexible settings for background smoothing:
                * Turn off background smoothing: set saliency = None in apply_FRESCO_opt()
                '''   
                frescoProc.controller.enable_controller(interattn_paras, attn_mask, False, i_)
                apply_FRESCO_opt(pipe, steps = timesteps[:end_opt_step], flows = flows, occs = occs, 
                                 correlation_matrix = correlation_matrix, saliency = saliency, 
                                 optimize_temporal = True, use_inversion = use_inversion)
                
                if i >= num_intraattn_steps:
                    frescoProc.controller.disable_intraattn(False)
                if t < step_interattn_end:
                    frescoProc.controller.disable_interattn()

            if propagation_mode:
                latents[0:2] = record_latents[i].detach().clone()
                record_latents[i] = latents[[0,len(latents)-1]].detach().clone()
            else:
                record_latents += [latents[[0,len(latents)-1]].detach().clone()]

            latent_model_input = latents[keygroup]
            keyframe_edges = edges[keygroup]
            if do_classifier_free_guidance:
                keyframe_edges = torch.cat([keyframe_edges.to(pipe.unet.dtype)] * 2)
                latent_model_input = torch.cat([latent_model_input] * 2)
            
            keyframe_prompt_embeds = pipe._encode_prompt(
                [prompts[k] for k in keygroup],
                device,
                1,
                do_classifier_free_guidance,
                [n_prompt] * len(keygroup)
            )
            if use_inversion:
                inv_keyframe_prompt_embeds = pipe._encode_prompt(
                    [inv_prompts[k] for k in keygroup], device, 1, False, ''
                )
                keyframe_prompt_embeds = torch.cat([inv_keyframe_prompt_embeds, keyframe_prompt_embeds])

                latent_model_input = torch.cat([ref_inv_latent[keygroup], latent_model_input])

            if use_controlnet:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = keyframe_prompt_embeds
                controlnet_cond = keyframe_edges

                down_block_res_samples, mid_block_res_sample = controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=controlnet_cond,
                    conditioning_scale=cond_scale[i+num_warmup_steps],
                    guess_mode=False,
                    return_dict=False,
                )
            else:
                down_block_res_samples, mid_block_res_sample = None, None
            
            if use_tokenflow:
                register_pivotal(pipe.unet, True)

            register_time(pipe, t)

            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=keyframe_prompt_embeds,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                return_dict=False,
            )[0]

            if use_fresco:
                frescoProc.controller.disable_controller(False)
                disable_FRESCO_opt(pipe, use_inversion)
            
            if not use_tokenflow:
                if use_inversion:
                    noise_pred = torch.cat(list(noise_pred.chunk(3)[1:]))

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                flows_, occs_, saliency_ = None, None, None
                if use_fresco and i + num_warmup_steps in bg_smoothing_steps:
                    flows_, occs_, saliency_ = flows, occs, saliency
                latents = denoise_step(pipe, noise_pred, t, latents, generator, repeat_noise, visualize_pipeline=visualize_pipeline,
                                       flows=flows_, occs=occs_, saliency=saliency_, ddim=edit_mode=='pnp')[0]
                    
                if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipe.scheduler.order == 0):
                    progress_bar.update()

                continue

            # using tokenflow method below
            register_pivotal(pipe.unet, False)       

            if use_fresco:
                frescoProc.controller.enable_tokenflow()

            denoised_latents = []

            for j, key in enumerate(keygroup):
                end = len(latents) if key == keygroup[-1] else keygroup[j + 1]
                if propagation_mode and key == 0:
                    end = 1

                # print(f"conducting tokenflow on images [{key}:{end}]")

                if use_fresco:
                    paras = torch.load(os.path.join(tokenflow_paras_path, f"paras_{i_}_{j}.pt"))

                    saliency, flows, occs, attn_mask, interattn_paras, correlation_matrix = paras
                    '''
                    Flexible settings for attention:
                    * Turn off FRESCO-guided attention: frescoProc.controller.disable_controller() 
                    Then you can turn on one specific attention submodule
                    * Turn on Cross-frame attention: frescoProc.controller.enable_cfattn() 
                    * Turn on Spatial-guided attention: frescoProc.controller.enable_intraattn() 
                    * Turn on Temporal-guided attention: frescoProc.controller.enable_interattn()

                    Flexible settings for optimization:
                    * Turn off Spatial-guided optimization: set optimize_temporal = False in apply_FRESCO_opt()
                    * Turn off Temporal-guided optimization: set correlation_matrix = [] in apply_FRESCO_opt()
                    * Turn off FRESCO-guided optimization: disable_FRESCO_opt(pipe)

                    Flexible settings for background smoothing:
                    * Turn off background smoothing: set saliency = None in apply_FRESCO_opt()
                    '''   
                    frescoProc.controller.enable_controller(interattn_paras, attn_mask, False, i_)
                    apply_FRESCO_opt(pipe, steps = timesteps[:end_opt_step], flows = flows, occs = occs, 
                                     correlation_matrix = correlation_matrix, saliency = saliency, 
                                     optimize_temporal = True, use_inversion = use_inversion)

                    if i >= num_intraattn_steps:
                        frescoProc.controller.disable_intraattn(False)
                    if t < step_interattn_end:
                        frescoProc.controller.disable_interattn()

                register_batch_ind = j + (j > 0) - (key == keygroup[-1])

                register_batch_idx(pipe.unet, register_batch_ind)
                
                full_latent = latents[key:end]
                latent_model_input_tokenflow = full_latent
                edges_tokenflow = edges[key:end]
                if do_classifier_free_guidance:
                    edges_tokenflow = torch.cat([edges_tokenflow.to(pipe.unet.dtype)] * 2)
                    latent_model_input_tokenflow = torch.cat([latent_model_input_tokenflow] * 2)
                
                prompt_embeds_tokenflow = pipe._encode_prompt(
                    prompts[key:end],
                    device,
                    1,
                    do_classifier_free_guidance,
                    [n_prompt] * (end - key)
                )
                inv_prompt_embeds_tokenflow = pipe._encode_prompt(
                    inv_prompts[key:end], device, 1, False, ''
                )
                prompt_embeds_tokenflow = torch.cat([inv_prompt_embeds_tokenflow, prompt_embeds_tokenflow])

                latent_model_input_tokenflow = torch.cat([ref_inv_latent[key:end], latent_model_input_tokenflow])
                
                if use_controlnet:
                    control_model_input = latent_model_input_tokenflow
                    controlnet_prompt_embeds = prompt_embeds_tokenflow
                    controlnet_cond = edges_tokenflow

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=controlnet_cond,
                        conditioning_scale=cond_scale[i+num_warmup_steps],
                        guess_mode=False,
                        return_dict=False,
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None

                noise_pred = pipe.unet(
                    latent_model_input_tokenflow,
                    t,
                    encoder_hidden_states=prompt_embeds_tokenflow,
                    cross_attention_kwargs=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                if use_fresco:
                    frescoProc.controller.disable_controller(False)
                    disable_FRESCO_opt(pipe, use_inversion)

                noise_pred = torch.cat(list(noise_pred.chunk(3)[1:]))

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                flows_, occs_, saliency_ = None, None, None
                if use_fresco and i + num_warmup_steps in bg_smoothing_steps:
                    flows_, occs_, saliency_ = flows, occs, saliency
                
                latents_batch = denoise_step(pipe, noise_pred, t, full_latent, generator, repeat_noise, visualize_pipeline=visualize_pipeline,
                                             flows=flows_, occs=occs_, saliency=saliency_, ddim=edit_mode=='pnp')[0]
                
                denoised_latents.append(latents_batch)

            latents = torch.cat(denoised_latents)

            if use_fresco:
                frescoProc.controller.disable_tokenflow()

            if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()

    if use_fresco:    
        frescoProc.controller.clear_store()
        frescoProc.controller.disable_controller()
        os.system(f"rm -rf {paras_save_path}")

    gc.collect()
    torch.cuda.empty_cache()

    return latents