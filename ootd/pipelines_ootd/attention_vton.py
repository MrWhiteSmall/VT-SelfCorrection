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
from typing import Any, Dict, Optional

import torch
from torch import nn

from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRACompatibleLinear
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormZero


@maybe_allow_in_graph
class GatedSelfAttentionDense(nn.Module):


    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        # self.attn(hidden_states,encoder_hidden_states=None,attn_mask=None)
        hidden_states = self.norm1(torch.cat([x, objs], dim=1))
        x = x + self.alpha_attn.tanh() * self.attn(
            hidden_states
            )[:, :n_visual, :]

        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):

    # @register_to_config
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
    ):
        super().__init__()

        # print(f'''
        # dim??{dim}  320 = 8*40  640=8*80 1280=8*160 好像就是把channel这个维度再划分一下？
        # num_attention_heads??{num_attention_heads} 8
        # attention_head_dim??{attention_head_dim} 40
        # cross_attention_dim??{cross_attention_dim} 768
        # activation_fn??{activation_fn} geglu
        # num_embeds_ada_norm??{num_embeds_ada_norm} None
        # attention_bias??{attention_bias} False
        # only_cross_attention??{only_cross_attention} False
        # double_self_attention??{double_self_attention} False
        # upcast_attention??{upcast_attention} False
        # norm_elementwise_affine??{norm_elementwise_affine} True
        # norm_type??{norm_type} layer_norm
        # norm_eps??{norm_eps} 1e-5
        # final_dropout??{final_dropout} False
        # attention_type??{attention_type} default
        # positional_embeddings??{positional_embeddings} None
        # num_positional_embeddings??{num_positional_embeddings} None
        # ''')
        # False
        self.only_cross_attention = only_cross_attention

        # False False False True
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single" # False
        self.use_layer_norm = norm_type == "layer_norm" # True


        self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        # print('self-attn 自注意力！！！！')  16次！！！
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        ################# attn1 Attention ####################
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        # print(f'''
        # 是否存在 attn2 ？？？
        # cross_attention_dim :True
        # double_self_attention : False
        # ''')
        self.norm2 = (
            nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        )
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim if not double_self_attention else None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )  # is self-attn if encoder_hidden_states is none


        # 3. Feed-forward
        # True
        # print(f'not self.use_ada_layer_norm_single??{not self.use_ada_layer_norm_single}')
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0


    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        spatial_attn_inputs = [],
        spatial_attn_idx = 0,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        # print(f'传入encoder hidden应该就是交叉注意力，否则就是自注意力，传入的时候，Vton和garm做了concat')
        # print(f'传入block时的hidden？？？{hidden_states.shape}')
        # if encoder_hidden_states:
        #     print(f'传入block时的encoder？？？{encoder_hidden_states.shape}')
        # else:
        #     print(f'传入的encoder是空的....')
        # batch_size = hidden_states.shape[0]

        spatial_attn_input = spatial_attn_inputs[spatial_attn_idx]
        spatial_attn_idx += 1
        # 这里 vton的输入和garm的输入 concat了一下 8 12288*2 320
        # 回顾一下  8 320 128 96 -> 8 128*96 320
        hidden_states = torch.cat((hidden_states, spatial_attn_input), dim=1)

        # 8 12288*2 320
        norm_hidden_states = self.norm1(hidden_states)

        # if self.use_ada_layer_norm: # False
        #     norm_hidden_states = self.norm1(hidden_states, timestep)
        # elif self.use_ada_layer_norm_zero:
        #     norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
        #         hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        #     )
        # elif self.use_layer_norm: # True
        #     norm_hidden_states = self.norm1(hidden_states)
        # elif self.use_ada_layer_norm_single: # False
        #     shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        #         self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        #     ).chunk(6, dim=1)
        #     norm_hidden_states = self.norm1(hidden_states)
        #     norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        #     norm_hidden_states = norm_hidden_states.squeeze(1)
        # else:
        #     raise ValueError("Incorrect norm used")

        # False
        # print(f'self.pos_embed???{self.pos_embed is not None}')
        # if self.pos_embed is not None: # False
        #     norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        # 1.0
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        # 空字典{}
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        # gligen_kwargs = cross_attention_kwargs.pop("gligen", None) # None
        # print(f'''
        # cross_attention_kwargs:{cross_attention_kwargs}
        # gligen_kwargs:{gligen_kwargs}
        # ''')
        
        # attn1前hidden ？？ torch.Size([8, 24576, 320]) 
        # print(f'attn1前hidden ？？ {norm_hidden_states.shape}')
        # print(f'only cross控制的是encoder_hidden??{self.only_cross_attention},这里应该是False吧？')
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        # attn1后 attn_output??torch.Size([8, 24576, 320]) 不变
        # print(f'attn1后 attn_output??{attn_output.shape}')

        # 指示是否使用某种自适应层归一化方法
        # gate_msa 表示多头自注意力（Multi-Head Self-Attention, MSA）的门控值。
        # if self.use_ada_layer_norm_zero: # False
        #     attn_output = gate_msa.unsqueeze(1) * attn_output
        # elif self.use_ada_layer_norm_single: # False
        #     attn_output = gate_msa * attn_output
        # print(f'自适应后，attn_output=attn1:{attn_output.shape}')

        # print(f'hidden chunk前:{hidden_states.shape}')
        # hidden states与attn_output同一个维度
        hidden_states = attn_output + hidden_states
        # attn_output + hidden_states之后 torch.Size([8, 24576, 320])
        # print(f'attn_output + hidden_states之后 {hidden_states.shape}')
        # 第2维分成两块 bz,ch/2,h,w
        hidden_states, _ = hidden_states.chunk(2, dim=1)
        # hidden chunk后:torch.Size([8, 12288, 320])
        # print(f'hidden chunk后:{hidden_states.shape}')

        # ndim返回几维数据，（bz,c,h,w） 则是4维 ndim=4
        # print(f'hidden ndim??{hidden_states.ndim}')
        # if hidden_states.ndim == 4: # False  hidden ndim??3
        #     hidden_states = hidden_states.squeeze(1)
        # 那现在呢？？torch.Size([8, 12288, 320])
        # print(f'那现在呢？？{hidden_states.shape}')

        # 2.5 GLIGEN Control
        # False
        # print(f'''gligen_kwargs::{gligen_kwargs is not None}''')
        # torch.Size([8, 12288, 320])
        # print(f'''hidden_states:::{hidden_states.shape}''')
        # if gligen_kwargs is not None: # False
        #     hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        # 交叉注意力attn2？？::True
        # print(f'''交叉注意力attn2？？::{self.attn2 is not None}''')
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)

            # if self.use_ada_layer_norm: # False
            #     norm_hidden_states = self.norm2(hidden_states, timestep)
            # elif self.use_ada_layer_norm_zero or self.use_layer_norm: # True
            #     norm_hidden_states = self.norm2(hidden_states)
            # elif self.use_ada_layer_norm_single:
            #     # For PixArt norm2 isn't applied here:
            #     # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
            #     norm_hidden_states = hidden_states
            # else:
            #     raise ValueError("Incorrect norm")

            # print(f'''
            # self.pos_embed is not None ::{self.pos_embed is not None} False
            # self.use_ada_layer_norm_single is False {self.use_ada_layer_norm_single is False} True
            # ''')
            # False and True = False
            # if self.pos_embed is not None and self.use_ada_layer_norm_single is False: # False
            #     norm_hidden_states = self.pos_embed(norm_hidden_states)

            ############# 使用交叉注意力 #####################
            # hidden_state=norm_hidden_states
            # attn2之前？？torch.Size([8, 12288, 320])
            # print(f'attn2之前？？{norm_hidden_states.shape}')
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            # attn2之后？？torch.Size([8, 12288, 320]) 不变
            # print(f'attn2之后？？{attn_output.shape}')
            
            # print(f'此时hidden与attn保持相同维度')
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # if not self.use_ada_layer_norm_single: # False
        #     norm_hidden_states = self.norm3(hidden_states)

        # if self.use_ada_layer_norm_zero:  # False
        #     norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        # if self.use_ada_layer_norm_single: #　False
        #     norm_hidden_states = self.norm2(hidden_states)
        #     norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # print(f'self._chunk_size is not None??{self._chunk_size is not None}')
        # if self._chunk_size is not None: # False
            
        #     # 这段代码的目的是将一个张量沿指定维度分割成多个块，对每个块应用一个前向传递（forward pass）函数，
        #     # 然后将这些块重新拼接起来
        #     num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
        #     ff_output = torch.cat(
        #         [
        #             self.ff(hid_slice, scale=lora_scale)
        #             for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
        #         ],
        #         dim=self._chunk_dim,
        #     )
        # else:
        ff_output = self.ff(norm_hidden_states, scale=lora_scale)
        # False 不存在
        # print(f'chunk size 是否存在:{self._chunk_size is not None}')
        # print(f'ff_output:{ff_output.shape}')

        # if self.use_ada_layer_norm_zero:
        #     ff_output = gate_mlp.unsqueeze(1) * ff_output
        # elif self.use_ada_layer_norm_single:
        #     ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        
        # 输出时 hidden？？torch.Size([8, 12288, 320])
        # print(f'输出时 hidden？？{hidden_states.shape}')
        # print(f'传入的spatial空间注意力并没有被改变，索引值+1')
        return hidden_states, spatial_attn_inputs, spatial_attn_idx


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = LoRACompatibleLinear if not USE_PEFT_BACKEND else nn.Linear

        act_fn = GEGLU(dim, inner_dim)

        # if activation_fn == "gelu": # False
        #     act_fn = GELU(dim, inner_dim)
        # if activation_fn == "gelu-approximate":
        #     act_fn = GELU(dim, inner_dim, approximate="tanh")
        # elif activation_fn == "geglu":
        #     act_fn = GEGLU(dim, inner_dim)
        # elif activation_fn == "geglu-approximate":
        #     act_fn = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(linear_cls(inner_dim, dim_out))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        compatible_cls = (GEGLU,) if USE_PEFT_BACKEND else (GEGLU, LoRACompatibleLinear)
        for module in self.net:
            if isinstance(module, compatible_cls):
                hidden_states = module(hidden_states, scale)
            else:
                hidden_states = module(hidden_states)
        return hidden_states
