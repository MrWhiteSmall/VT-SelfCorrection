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
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .unet_vton_2d_blocks import (
    UNetMidBlock2D, 
    UNetMidBlock2DCrossAttn,  # 只用到了这个，并且只用了一次
    UNetMidBlock2DSimpleCrossAttn,
    get_down_block,
    get_up_block,
)

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import (
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None


'''
FrozenDict([('sample_size', 64),
 ('in_channels', 8), 
 ('out_channels', 4), 
 ('center_input_sample', False), 
 ('flip_sin_to_cos', True), 
 ('freq_shift', 0), 
 ('down_block_types', ['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D']), 
 ('mid_block_type', 'UNetMidBlock2DCrossAttn'), 
 ('up_block_types', ['UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D']), 
 ('only_cross_attention', False), 
 ('block_out_channels', [320, 640, 1280, 1280]), 
 ('layers_per_block', 2), 
 ('downsample_padding', 1), 
 ('mid_block_scale_factor', 1), 
 ('dropout', 0.0), 
 ('act_fn', 'silu'), 
 ('norm_num_groups', 32), 
 ('norm_eps', 1e-05), 
 ('cross_attention_dim', 768), 
 ('transformer_layers_per_block', 1), 
 ('reverse_transformer_layers_per_block', None), 
 ('encoder_hid_dim', None), 
 ('encoder_hid_dim_type', None), 
 ('attention_head_dim', 8), 
 ('num_attention_heads', None), 
 ('dual_cross_attention', False), 
 ('use_linear_projection', False), 
 ('class_embed_type', None), 
 ('addition_embed_type', None), 
 ('addition_time_embed_dim', None), 
 ('num_class_embeds', None), 
 ('upcast_attention', False), 
 ('resnet_time_scale_shift', 'default'), 
 ('resnet_skip_time_act', False), 
 ('resnet_out_scale_factor', 1.0), 
 ('time_embedding_type', 'positional'), 
 ('time_embedding_dim', None), 
 ('time_embedding_act_fn', None), 
 ('timestep_post_act', None), 
 ('time_cond_proj_dim', None), 
 ('conv_in_kernel', 3), 
 ('conv_out_kernel', 3), 
 ('projection_class_embeddings_input_dim', None), 
 ('attention_type', 'default'), 
 ('class_embeddings_concat', False), 
 ('mid_block_only_cross_attention', None), 
 ('cross_attention_norm', None), 
 ('addition_embed_type_num_heads', 64), 
 ('_class_name', 'UNetVton2DConditionModel'), 
 ('_diffusers_version', '0.24.0.dev0'), 
 ('_name_or_path', '/public/home/yangzhe/ltt/lsj/git_workspace/OOTDiffusion-old/checkpoints/ootd/ootd_hd/checkpoint-36000')])

'''
class UNetVton2DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None, # 64
        in_channels: int = 4, # 8
        out_channels: int = 4, # 4
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", 
                                      "CrossAttnUpBlock2D", 
                                      "CrossAttnUpBlock2D", 
                                      "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280, #768
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,
    ):
        super().__init__()

        # 64
        self.sample_size = sample_size



        # 如果 `num_attention_heads` 未定义（这在大多数模型中是这样），
        # 它将默认为 `attention_head_dim`。乍一看这有些奇怪，确实如此。
        # 这样做的原因是为了纠正该库创建时引入的变量命名错误。
        # 这个命名错误在很久以后才被发现，见 https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # 将 `attention_head_dim` 改为 `num_attention_heads` 会影响到超过 40,000 个配置，
        # 因此在这里我们通过命名修正来解决这个问题。

        num_attention_heads = num_attention_heads or attention_head_dim
        # 8
        # print(f'num_attention_heads:{num_attention_heads}')

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        # time_embedding_dim :: None 选后面那个
        # time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
        time_embed_dim =  block_out_channels[0] * 4

        # flip_sin_to_cos：True 、 freq_shift：0
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act, # None
            cond_proj_dim=time_cond_proj_dim, # None
        )

        
        self.encoder_hid_proj = None

        # class embedding
        self.class_embedding = None

        # addition_embed_type可以是'None'或text或text_img
        # if addition_embed_type == "text":
        #     if encoder_hid_dim is not None:
        #         text_time_embedding_from_dim = encoder_hid_dim
        #     else:
        #         text_time_embedding_from_dim = cross_attention_dim

        #     self.add_embedding = TextTimeEmbedding(
        #         text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
        #     )
        # elif addition_embed_type == "text_image":
        #     # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
        #     # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
        #     # case when `addition_embed_type == "text_image"` (Kadinsky 2.1)`
        #     self.add_embedding = TextImageTimeEmbedding(
        #         text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
        #     )

        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else: # 选这个
            self.time_embed_act = get_activation(time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool): # True
            mid_block_only_cross_attention = only_cross_attention
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(num_attention_heads, int): # True
            # (1,)*3=(1,1,1,) 所以这里同理
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int): # 8
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int): # 768
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int): # 2
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int): # 1
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        if class_embeddings_concat: # False
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else: # 选这个
            blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
            transformer_layers_per_block=transformer_layers_per_block[-1],
            in_channels=block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            attention_type=attention_type,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_num_groups is not None: # 32 True
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

            self.conv_act = get_activation(act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        # 这里是为了让进入unet之前的维度在处理后输出后保持不变
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, 
            kernel_size=conv_out_kernel, padding=conv_out_padding
        )


    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor, _remove_lora=True)

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module splits the input tensor in slices to compute attention in
        several steps. This is useful for saving some memory in exchange for a small decrease in speed.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def enable_freeu(self, s1, s2, b1, b2):
        r"""Enables the FreeU mechanism from https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stage blocks where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of values that
        are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate the "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism."""
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)

    def forward(
        self,
        sample: torch.FloatTensor,
        spatial_attn_inputs,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        # print(f'''
        # {sample.shape} 8 8 128 96
        # {type(spatial_attn_inputs)} list 16个8 12288 40？
        # {timestep.shape} 空的
        # {encoder_hidden_states.shape} 8 2 768
        # 下面全是None
        # {type(class_labels)}
        # {type(timestep_cond)}
        # {type(attention_mask)}
        # {type(cross_attention_kwargs)}
        # {type(down_block_additional_residuals)}
        # {type(mid_block_additional_residual)}
        # {type(down_intrablock_additional_residuals)}
        # {type(encoder_attention_mask)}
        # ''')
        
        # default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        # 0. center input if necessary
        # Faslse
        # print(f'self.config.center_input_sample??{self.config.center_input_sample}')
        # if self.config.center_input_sample:
        #     sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep

        timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        
        ################## 预处理 ##########################
        # 2. pre-process
        # 8 8 128 96
        # print(f'unet预处理vton sample经过conv前，{sample.shape}')
        sample = self.conv_in(sample)
        # 8 320 128 96
        # print(f'unet预处理vton sample经过conv后，{sample.shape}')

       #################### 注意力 ########################
        # for spatial attention
        # 空间注意力
        spatial_attn_idx = 0

        # 3. down
        # 1.0
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        is_adapter = down_intrablock_additional_residuals is not None # False
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other


        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            # AttributeError: 'DownBlock2D' object has no attribute 'has_cross_attention'
            # 3 CrossAttn + 1 Downsample
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}

                sample, res_samples, spatial_attn_inputs, spatial_attn_idx = downsample_block(
                    hidden_states=sample,
                    spatial_attn_inputs=spatial_attn_inputs,
                    spatial_attn_idx=spatial_attn_idx,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)
                if is_adapter and len(down_intrablock_additional_residuals) > 0: # False
                    sample += down_intrablock_additional_residuals.pop(0)
            #  3=2+1 3 3 2
            # print('down......',len(res_samples),'\t',res_samples[1].shape)
            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample, spatial_attn_inputs, spatial_attn_idx = self.mid_block(
                    sample,
                    spatial_attn_inputs=spatial_attn_inputs,
                    spatial_attn_idx=spatial_attn_idx,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)
            # print('mid....',sample.shape)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            # False True True True
            # print('*'*10,'有无注意力？',hasattr(upsample_block, "has_cross_attention"))
            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                # 输出最后一次的attention
                sample, spatial_attn_inputs, spatial_attn_idx = upsample_block(
                    hidden_states=sample,
                    spatial_attn_inputs=spatial_attn_inputs,
                    spatial_attn_idx=spatial_attn_idx,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    scale=lora_scale,
                )
            # print('up....',sample.shape)

        # 6. post-process
        # GroupNorm(32, 320, eps=1e-05, affine=True)
        # print('self.conv_norm_out',self.conv_norm_out)
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        # print(f'unet 后处理 conv前：：{sample.shape}') # torch.Size([8, 320, 128, 96])
        sample = self.conv_out(sample)
        # print(f'unet 后处理 conv后：：{sample.shape}') # torch.Size([8, 4, 128, 96])

        # 的确是False
        # print(f'再确认一下USE_PEFT_BACKEND??{USE_PEFT_BACKEND}')

        if not return_dict: # True
            return (sample,)

        # sample就是每次UNet采样的结果 noise score(xt,t)
        return UNet2DConditionOutput(sample=sample)
