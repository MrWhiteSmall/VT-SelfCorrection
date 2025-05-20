# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by Yuhao Xu for OOTDiffusion (https://github.com/levihsu/OOTDiffusion)
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

# from transformers import AutoProcessor, CLIPVisionModelWithProjection

from .unet_vton_2d_condition import UNetVton2DConditionModel
from .unet_garm_2d_condition import UNetGarm2DConditionModel

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
# from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    deprecate,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

# from .leanable_small_model import SmallModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.preprocess
def preprocess(image):
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    deprecate("preprocess", "1.0.0", deprecation_message, standard_warn=False)
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


'''
self.pipe = OotdPipeline.from_pretrained(
    MODEL_PATH,
    unet_garm=unet_garm,
    unet_vton=unet_vton,
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    safety_checker=None,
    requires_safety_checker=False,
).to(self.gpu_id)
'''
class OotdPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    r"""
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """
    model_cpu_offload_seq = "text_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "vton_latents"]

    @register_to_config
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet_garm: UNetGarm2DConditionModel,
        unet_vton: UNetVton2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        # print(unet_vton.config)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image_garm: PipelineImageInput = None,
        image_vton: PipelineImageInput = None,
        mask: PipelineImageInput = None,
        image_ori: PipelineImageInput = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        learnable_model = None, # 传入learnable model
        **kwargs,
    ):
        # print(f'''pipeline ootd call
        # {type(prompt)} None
        # {np.array(image_garm).shape} PIL 1024 768 3
        # {np.array(image_vton).shape} PIL 1024 768 3
        # {np.array(mask).shape} PIL 1024 768
        # {np.array(image_ori).shape} PIL 1024 768 3
        # num_inference_steps:{num_inference_steps} 20
        # guidance_scale??{guidance_scale} 7.5
        # image_guidance_scale??{image_guidance_scale} 2.0
        # {type(negative_prompt)} None
        # num_images_per_prompt??{num_images_per_prompt} 4
        # eta::{eta} 0.0
        # {type(generator)}
        # {type(latents)} None
        # {prompt_embeds.shape}  1 2 768
        # {type(negative_prompt_embeds)} None
        # {output_type} str
        # {type(callback_on_step_end)} None
        # {len(callback_on_step_end_tensor_inputs)} list
        # {kwargs} dict 空的
        # ''')

        # 0. Check inputs
        # self.check_inputs(
        #     prompt,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     callback_on_step_end_tensor_inputs,
        # )
        self._guidance_scale = guidance_scale # 7.5
        # 主要用来判断是否要用无分类引导 > 1 = True 否则 False
        self._image_guidance_scale = image_guidance_scale # 2.0

        # 1. Define call parameters
        batch_size = prompt_embeds.shape[0] # 1 2 768

        device = self._execution_device
        # check if scheduler is in sigmas space
        # scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")
        # ！！！！scheduler_is_in_sigma_space = False ！！
        # print(f'scheduler_is_in_sigma_space {scheduler_is_in_sigma_space}')

        # 2. Encode input prompt
        # 1 2 768
        with torch.no_grad():
            prompt_embeds = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )

        # 3. Preprocess image
            image_garm = self.image_processor.preprocess(image_garm)
            image_vton = self.image_processor.preprocess(image_vton)
            image_ori = self.image_processor.preprocess(image_ori)
        mask = np.array(mask)
        mask[mask < 127] = 0
        mask[mask >= 127] = 255
        # print(f'''对garm vton ori 都做了预处理
        # {image_garm.shape} torch 1 3 1024 768
        # {image_vton.shape} torch 1 3 1024 768
        # {image_ori.shape} torch 1 3 1024 768
        # {mask.shape}  1024 768
        # ''')
        
        #############  可以在这里把mask存一下？？ ##############
        mask = torch.tensor(mask)
        mask = mask / 255
        # bz 1 h w
        mask = mask.reshape(-1, 1, mask.size(-2), mask.size(-1))

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        with torch.no_grad():
            # 5. Prepare Image latents
            garm_latents = self.prepare_garm_latents(
                image_garm,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                self.do_classifier_free_guidance,
                generator,
            )
            # 8 4 128 96
            # print(f'prepare_garm_latents????{garm_latents.shape}')

            vton_latents, mask_latents, image_ori_latents = self.prepare_vton_latents(
                image_vton,
                mask,
                image_ori,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                self.do_classifier_free_guidance,
                generator,
            )
        # torch.Size([8, 4, 128, 96]),torch.Size([4, 1, 128, 96]),torch.Size([4, 4, 128, 96])
        # print(f'prepare_vton_latents???{vton_latents.shape},{mask_latents.shape},{image_ori_latents.shape}')

        height, width = vton_latents.shape[-2:] # 128 96
        # vae scale factor = 8
        # print(f'vae scale factor???{self.vae_scale_factor}')
        height = height * self.vae_scale_factor # 1024
        width = width * self.vae_scale_factor # 768

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        # print(f'num_channels_latents???{num_channels_latents}')
        
        with torch.no_grad():
            # 传入的latents就是None
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        # 4 4 128 96
        # print(f'prepare_latents???{latents.shape}')
        noise = latents.clone()

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # 空的
        # print(f'extra_step_kwargs?????{extra_step_kwargs}')

        # 9. Denoising loop
        '''
        热身步骤：热身步骤通常是指在实际推理或训练之前的初始步骤，模型通过这些步骤进行初始调整，
        使得后续的推理或训练过程更加稳定和高效。
        '''
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)



        ################# 从garm model来的空间注意力 spatial_attn_outputs ##################
        with torch.no_grad():
            _, spatial_attn_outputs = self.unet_garm( # 16个8 12288 40
                garm_latents,
                0,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )
            # garm只传入一次空间注意力数据，后面都是通过vton中的交叉注意力处理


        ##################### learnable W 对于 spatial_attn_outputs #####################
        # print('开始处理 learnable',type(learnable_model)) # NoneType
        spatial_attn_outputs_ori = [g.clone() for g in spatial_attn_outputs]
        spatial_attn_outputs = [(learnable_model(g.unsqueeze(0))).squeeze(0) for g in spatial_attn_outputs]  # [(8 1228 40),.....]
        # for s in spatial_attn_outputs:
        #     print('处理过的 spatial_attn_outputs shape',s.shape,end='\t')
        
        # vton_latents = learnable_model(vton_latents)                             # 8 4 h w
        
        
        # 对于处理前后的prompt 需要做loss，所以最终需要返回出去
        # 尝试一下，直接返回loss可不可以？
        
        # 计算第一个特征对的L2距离
        prompt_loss = None
        for i in range(len(spatial_attn_outputs_ori)):
            if prompt_loss is None:
                prompt_loss = F.mse_loss(spatial_attn_outputs_ori[i], spatial_attn_outputs[i])
            else:
                prompt_loss += F.mse_loss(spatial_attn_outputs_ori[i], spatial_attn_outputs[i])
                
        # pre_prompt    = torch.concat(spatial_attn_outputs_ori,dim=0)
        # now_prompt        = torch.concat(spatial_attn_outputs,dim=0)
        # print('pre_prompt',pre_prompt.requires_grad,'now_prompt',now_prompt.requires_grad)
        print('prompt_loss',prompt_loss)
        ####################################################################################



        # 开始去噪 steps....
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # index & time
            for i, t in enumerate(timesteps):
                # 模型输入  [latents] * 2
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # 前一半是噪声，后一半是img
                latent_vton_model_input = torch.cat([scaled_latent_model_input, vton_latents], dim=1)
                # latent_vton_model_input = scaled_latent_model_input + vton_latents
                # print(f'''
                # {t.shape}  torch.Size([])
                # {latent_model_input.shape} torch.Size([8, 4, 128, 96])
                # {scaled_latent_model_input.shape} torch.Size([8, 4, 128, 96])
                # {latent_vton_model_input.shape} torch.Size([8, 8, 128, 96])
                # ''')

                # !!!!!!!!!!!!!!!  每次迭代都获取的是同一份空间注意力 ！！！！！！！！！！！！！！！！！
                spatial_attn_inputs = spatial_attn_outputs.copy()

                # 预测出来的噪声，也称score(xt,t) #################
                # predict the noise residual
                # 输入 8 8 128 96
                noise_pred = self.unet_vton(
                    latent_vton_model_input,
                    spatial_attn_inputs,
                    t,
                    # 输入文本特征
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )
                # print(f'vton unet处理前??{len(noise_pred)}') # 1
                noise_pred = noise_pred[0]
                # 8 4 128 96
                # print(f'vton unet处理后??{noise_pred.shape}')

                # 执行 【输入】-noise_pred ########################

                # 执行无分类引导 #################################
                # perform guidance
                if self.do_classifier_free_guidance: # True
                    # 这里的意思就是前一半是按条件预测的，后一半是无条件预测的
                    # 前一半是灵活生成，后一半是规则限制
                    # 4 4 128 96 |||| 4 4 128 96
                    noise_pred_text_image, noise_pred_text = noise_pred.chunk(2)
                    # 变成 4 4 128 96
                    noise_pred = (
                        noise_pred_text
                        + self.image_guidance_scale * (noise_pred_text_image - noise_pred_text)
                    )


                # 所以到此为止，noise_pred才算结束计算
                ############## 输出 noise_pred #############
                ############## 保存 noise_pred #############
                '''
                noise_pred::>
                <class 'torch.Tensor'>
                torch.Size([4, 4, 128, 96])
                '''
                # print('noise_pred::>')
                # print(type(noise_pred))
                # print(noise_pred.shape)
                # prefix = '/public/home/yangzhe/ltt/lsj/OOTD_attn'
                # torch.save(noise_pred,f'{prefix}/noise_pred-{i}.pth')


                ############# 开始计算xt-1 ##############
                # 计算xt-1
                # compute the previous noisy sample x_t -> x_t-1
                # extra_step_kwargs = {} 是空的
                latents = self.scheduler.step(noise_pred, t, latents, 
                                              **extra_step_kwargs, return_dict=False)
                # print(f'self.scheduler.step::??{len(latents)}') # 1
                latents = latents[0]
                # torch.Size([4, 4, 128, 96])
                # print(f'self.scheduler.step::??之后呢?{latents.shape}')

                init_latents_proper = image_ori_latents * self.vae.config.scaling_factor
                # scale factor 0.18215
                # print(f'self.vae.config.scaling_factor??{self.vae.config.scaling_factor}')
                # torch.Size([4, 4, 128, 96])
                # print(f'init_latents_proper??{init_latents_proper.shape}')

                # 重新渲染试穿图片
                # repainting
                # 给原图像加噪
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([noise_timestep])
                    )

                latents = (1 - mask_latents) * init_latents_proper + mask_latents * latents

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                # 这里 最终的latent 就是xt-1
                '''
                latent=xt-1::
                <class 'torch.Tensor'>
                torch.Size([4, 4, 128, 96])

                !!!! timesteps是个tensor !!!!
                '''
                '''
                timesteps : torch.Size([20])
                '''
                # print('#'*10,'latent=xt-1::','#'*10)
                # print(type(timesteps))
                # print(timesteps.shape)
                # prefix = '/public/home/yangzhe/ltt/lsj/OOTD_attn'
                # torch.save(latents,f'{prefix}/X-{i}.pth')

        # 结束去噪 step=0s

        # pil
        # print(f'*********output_type:{output_type}*****************')
        # vae decode之前？？torch.Size([4, 4, 128, 96])
        # print(f'vae decode之前？？{latents.shape}')
        if not output_type == "latent":
            # torch.Size([4, 3, 1024, 768])
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)
            # print(f'image vae decode len{len(image)}') # 1
            image = image[0]
            # image vae decode torch.Size([4, 3, 1024, 768])
            # print(f'image vae decode {image.shape}')
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        '''
        ########## 去规格化？？ ##########
        None
        [True, True, True, True]
        '''
        # print('#'*10,'去规格化？？','#'*10)
        # print(has_nsfw_concept)
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        # print(do_denormalize)

        ############## 查看此处image是不是满足高斯分布 #############
        ###### 处理前 ###########
        '''
        ########## 处理image前 ##########
        <class 'torch.Tensor'>
        torch.Size([4, 3, 1024, 768])
        '''
        # print('#'*10,'处理image前','#'*10)
        # print(type(image))
        # print(image.shape)
        # prefix = '/public/home/yangzhe/ltt/lsj/OOTD_attn'
        # torch.save(image,f'{prefix}/image-pre.pth')

        '''
        File "/data/lsj/OOTDiffusion/ootd/inference_ootd_hd.py", line 141, in __call__
            images = self.pipe(prompt_embeds=prompt_embeds,
        File "/data/lsj/OOTDiffusion/ootd/pipelines_ootd/pipeline_ootd.py", line 485, in __call__
            image = self.image_processor.postprocess(
        File "/home/syy/miniconda3/envs/ootd201/lib/python3.10/site-packages/diffusers/image_processor.py", line 401, in postprocess
            image = self.pt_to_numpy(image)
        File "/home/syy/miniconda3/envs/ootd201/lib/python3.10/site-packages/diffusers/image_processor.py", line 134, in pt_to_numpy
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
        '''
        # 对图片后处理，处理成为rgb格式
        # image = image.cpu().detach().numpy().astype(np.float32)
        # image = image.cpu().numpy().astype(np.float32)
        print('在此之前，image需要grad？？',image.requires_grad) # True
        image = self.image_processor.postprocess(
            # torch.tensor(image),
            image,
            output_type=output_type, 
            do_denormalize=do_denormalize)
        ###### 处理后 ############
        '''
        $$$$$$$$$$ 处理image后 $$$$$$$$$$
        <class 'list'>
        4
        <class 'PIL.Image.Image'>
        '''

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept,prompt_loss)

        return StableDiffusionPipelineOutput(
            images=image, 
            nsfw_content_detected=has_nsfw_concept),prompt_loss
    ################ 处理文本特征，输出文本特征 ########################
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None, # 1 2 768
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        # prompt 就是 None
        batch_size = prompt_embeds.shape[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # 1         2 
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        # print(f'处理前{prompt_embeds.shape}') # 1 2 768
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # (1 8 768) num_images_per_prompt=4
        # print(f'处理ing repeat {prompt_embeds.shape} \t num_images_per_prompt??{num_images_per_prompt}')
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        # torch.Size([4, 2, 768])           1               4                      2
        # print(f'处理后{prompt_embeds.shape} {bs_embed} {num_images_per_prompt} {seq_len}')

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None: # True
            uncond_tokens: List[str]
            if negative_prompt is None: # True
                uncond_tokens = [""] * batch_size
            # elif type(prompt) is not type(negative_prompt):
            #     raise TypeError(
            #         f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
            #         f" {type(prompt)}."
            #     )
            # elif isinstance(negative_prompt, str):
            #     uncond_tokens = [negative_prompt]
            # elif batch_size != len(negative_prompt):
            #     raise ValueError(
            #         f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
            #         f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
            #         " the batch size of `prompt`."
            #     )
            # else:
            #     uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

        # True 总是执行无分类引导
        # print('&'*10,do_classifier_free_guidance,' 无分类引导','&'*10)
        # 利用prompt_embeds做一次无分类引导
        # 4 2 768
        # print(f'无分类引导前？{prompt_embeds.shape}')
        if do_classifier_free_guidance: # True
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds])
        # 8 2 768
        # print(f'无分类引导后？{prompt_embeds.shape}')
        # 文本特征 8*2*768
        # print('文本特征：：：',prompt_embeds.shape)
        # prefix = '/public/home/yangzhe/ltt/lsj/OOTD_attn'
        # torch.save(prompt_embeds,f'{prefix}/prompt_embeds.pth')

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        # deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # bz,ch, 1024/8=128 768/8=96
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # 1.0  
        # z noise shape:torch.Size([4, 4, 128, 96])
        # print(f'self.scheduler.init_noise_sigma:{self.scheduler.init_noise_sigma}')
        # print(f'z noise shape:{latents.shape}')
        return latents

    def prepare_garm_latents(
        self, image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        image_latents = self.vae.encode(image).latent_dist.mode()

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, uncond_image_latents], dim=0)

        return image_latents
    
    def prepare_vton_latents(
        self, image, mask, image_ori, batch_size, 
        num_images_per_prompt, dtype, device, 
        do_classifier_free_guidance, generator=None
    ):
        image = image.to(device=device, dtype=dtype)
        image_ori = image_ori.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        image_latents = self.vae.encode(image).latent_dist.mode()
        image_ori_latents = self.vae.encode(image_ori).latent_dist.mode()
        # print(f'image shape[1] {image.shape}') #([1, 3, 1024, 768])
        # False,<torch._C.Generator object at 0x7fe6cebe86f0>
        # print(f'{isinstance(generator, list)},{generator}')

        # mask是单独采样成 4 1 128 96
        mask = torch.nn.functional.interpolate(
            mask, size=(image_latents.size(-2), image_latents.size(-1))
        )
        mask = mask.to(device=device, dtype=dtype)

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            mask = torch.cat([mask] * additional_image_per_prompt, dim=0)
            image_ori_latents = torch.cat([image_ori_latents] * additional_image_per_prompt, dim=0)
        else: # 执行这个
            image_latents = torch.cat([image_latents], dim=0)
            mask = torch.cat([mask], dim=0)
            image_ori_latents = torch.cat([image_ori_latents], dim=0)

        if do_classifier_free_guidance:
            # uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents] * 2, dim=0)

        # print(f'''
        # {do_classifier_free_guidance} True
        # {batch_size} 4
        # {image_latents.shape} torch.Size([8, 4, 128, 96])
        # {image_latents.shape[0]} 8
        # ''')

        return image_latents, mask, image_ori_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_freeu
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        self.unet_vton.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet_vton.disable_freeu()

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self.image_guidance_scale >= 1.0
