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
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .attention_vton import BasicTransformerBlock

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import ImagePositionalEmbeddings
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate
# from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import CaptionProjection, PatchEmbed
# 不传入lora layer则和普通的conv，linear一样
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle


@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class Transformer2DModel(ModelMixin, ConfigMixin):
    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,# 8
        attention_head_dim: int = 88, # (40,80,160)*2 160 (160,80,40)*3
        in_channels: Optional[int] = None, # 320 640 1280
        out_channels: Optional[int] = None,
        num_layers: int = 1, # 1
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None, # 768
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
    ):
        super().__init__()
        # print(f'''
        # 多少注意力头？？{num_attention_heads},
        # 注意力头维度？？{attention_head_dim},
        # 交叉注意力维度？？{cross_attention_dim}
        # in_channels？？{in_channels}
        # out_channels？？{out_channels}
        # num_layers？？{num_layers}
        # sample_size？？{sample_size}
        # num_vector_embeds？？{num_vector_embeds}
        # patch_size？？{patch_size}
        # num_embeds_ada_norm？？{num_embeds_ada_norm}
        # use_linear_projection？？{use_linear_projection}
        # only_cross_attention？？{only_cross_attention}
        # double_self_attention??{double_self_attention}
        # upcast_attention??{upcast_attention}
        # caption_channels??{caption_channels}
        # ''')
        # layer_norm
        # print(f'norm_type??{norm_type}')
        # USE_PEFT_BACKEND???::False
        # print(f'USE_PEFT_BACKEND {USE_PEFT_BACKEND}')    

        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # False 所以 LoRACompatibleConv  LoRACompatibleLinear
        # 不传入lora layer则和普通的conv linear 没区别，传入的话，就是普通的conv、linear的输出+lora layer的输出
        conv_cls =  LoRACompatibleConv
        linear_cls =  LoRACompatibleLinear

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None) # True
        self.is_input_vectorized = num_vector_embeds is not None # False
        self.is_input_patches = in_channels is not None and patch_size is not None # False
        # print(f'''
        # {self.is_input_continuous}
        # {self.is_input_vectorized}
        # {self.is_input_patches}
        # ''')


        # 2. Define input layers
        if self.is_input_continuous: # True
            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            self.proj_in = conv_cls(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim, # head=8 * head_dim(40,80,160)
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                )
                for d in range(num_layers) # 1 层
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        self.proj_out = conv_cls(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

        # 5. PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False

        self.caption_projection = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        spatial_attn_inputs = [],
        spatial_attn_idx = 0,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # print(f'''
        # hidden_states??{hidden_states.shape}
        # encoder_hidden_states??{type(encoder_hidden_states)}
        # timestep??{type(timestep)} None
        # added_cond_kwargs??{type(added_cond_kwargs)} None
        # class_labels??{type(class_labels)} None
        # cross_attention_kwargs??{type(cross_attention_kwargs)} None
        # attention_mask??{type(attention_mask)} None
        # encoder_attention_mask??{type(encoder_attention_mask)} None
        # return_dict??{type(return_dict)}
        # ''')
        # print(f'''
        # hidden_states??{hidden_states.shape}
        # encoder_hidden_states??{encoder_hidden_states.shape} 8 2 768 怎么来的？？ 1 2 768 -》 1 8 768 -》 4 2 768 -》 无分类引导[pro,pro] 8 2 768
        # return_dict??{return_dict} False
        # ''')

        # Retrieve lora scale.
        # 1.0      cross_attention_kwargs is None
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_continuous: # True
            batch, _, height, width = hidden_states.shape
            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            
            # torch.Size([8, 320, 128, 96])
            # print(f'Input hidden state{hidden_states.shape}')
            hidden_states = (
                self.proj_in(hidden_states, scale=lora_scale)
                if not USE_PEFT_BACKEND
                else self.proj_in(hidden_states) # 选这个
            )
            # torch.Size([8, 320, 128, 96]) 没变
            # print(f'pro_i >>Input hidden state{hidden_states.shape}')
            inner_dim = hidden_states.shape[1]
            # print(f'innert dim {inner_dim}') # 320
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            # 改造了一下 bz h*w inner_dim  ！！！torch.Size([8, 12288, 320])！！！
            # print(f'改造了一下 bz h*w inner_dim  {hidden_states.shape}')

        # 2. Blocks

        for block in self.transformer_blocks:
            hidden_states, spatial_attn_inputs, spatial_attn_idx = block(
                hidden_states,
                spatial_attn_inputs,
                spatial_attn_idx,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )
            # print('这里的block就是transformer block')
            # 处理中.....hidden torch.Size([8, 12288, 320])
            # print(f'处理中.....hidden {hidden_states.shape}')

        # 3. Output
        if self.is_input_continuous: # True
            if not self.use_linear_projection: # True
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                # 输出之前重新改回来 bz inner h w torch.Size([8, 320, 128, 96])
                # print(f'输出之前重新改回来 bz inner h w {hidden_states.shape}')
                hidden_states = (
                    self.proj_out(hidden_states, scale=lora_scale) # 选这个
                    if not USE_PEFT_BACKEND # True
                    else self.proj_out(hidden_states) 
                )
                # proj_out 之后torch.Size([8, 320, 128, 96]) 没变
                # print(f'{not USE_PEFT_BACKEND} proj_out 之后{hidden_states.shape}')

            output = hidden_states + residual
            # print(f'transformer model 输出 ？？{output.shape}')

        if not return_dict: # True
            return (output,), spatial_attn_inputs, spatial_attn_idx
        # 这行不会执行
        return Transformer2DModelOutput(sample=output), spatial_attn_inputs, spatial_attn_idx
