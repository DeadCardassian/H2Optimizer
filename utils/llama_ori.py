from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import time
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

import pdb
import types
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRotaryEmbedding,
    LlamaForCausalLM,
)
from utils.cache import Cache, HHCache, StaticCache
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPast

cnt_for = 0
cnt_tok = 0
curr_token_idx = 0
time_seq_1 = []
time_seq_2 = []
time_seq_3 = []
time_seq_4 = []
logger = logging.get_logger(__name__)

__all__ = ["H2OLlamaForCausalLM"]


def prepend_constant_block(X: torch.Tensor, M: int) -> torch.Tensor:
    # X 的形状为 (A, B, C, N)
    # 截取最后一维的第一个元素，形状为 (A, B, C, 1)
    first_element = X[..., 0:1]
    
    # 将这个单一元素沿最后一维重复 M 次，得到 (A, B, C, M)
    # 这里使用 repeat 来复制
    M_block = first_element.repeat(1, 1, 1, M)

    # 拼接 M_block 与 X
    X_expanded = torch.cat([M_block, X], dim=-1)
    return X_expanded


def _make_causal_mask(
    bsz: int, tgt_len: int, past_key_values_length: int, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def apply_rotary_pos_emb_single(x, cos, sin, position_ids=None, unsqueeze_dim=1):

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)

    return x_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class H2OLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.positional_rolling = config.enable_position_rolling
        self.token_cnt = 0

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #print("kvcache shape:", past_key_value.key_cache[0].shape)
        if(self.layer_idx == 100):
            global curr_token_idx
            curr_token_idx += 1
            print(curr_token_idx)
            try:
                #[batch_size, num_heads, seq_len, head_dim]
                print("kvcache shape:", past_key_value.key_cache[5].shape)
                #key_l5 = past_key_value.key_cache[5]
                #masked_positions = (key_l5 == 0).all(dim=-1)  # 检查最后一个维度是否全为0

                # 打印掩码信息
                #for b in range(key_l5.size(0)):
                #    for h in range(key_l5.size(1)):
                #        print(f"Batch {b}, Head {h}: Masked Tokens")
                #        print(torch.nonzero(masked_positions[b, h], as_tuple=True))  # 打印被掩码的位置
            except IndexError as e:
                print("IndexError: key_cache is empty or does not have enough elements.")
                print(f"Current key_cache length: {len(past_key_value.key_cache)}")
                print("Exception details:", str(e))
        #print("num token shape:", past_key_value.num_hh_tokens.shape)
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if not self.positional_rolling:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

            kv_seq_len = past_key_value.get_seq_length(self.layer_idx) if past_key_value is not None else key_states.shape[-2]

            if not position_ids.nelement() > 1:
                # decoding stage
                key_position_ids = torch.arange(kv_seq_len, device=hidden_states.device).unsqueeze(0)
                query_position_ids = key_position_ids[:, -1].unsqueeze(0)
            elif not kv_seq_len == position_ids.shape[-1]:
                # prefilling stage with evicting
                query_position_ids = position_ids
                key_position_ids = torch.arange(kv_seq_len, device=hidden_states.device).unsqueeze(0)
            else:
                # prefilling stage
                query_position_ids = position_ids
                key_position_ids = position_ids

            key_cos, key_sin = self.rotary_emb(value_states, key_position_ids)
            query_cos, query_sin = self.rotary_emb(value_states, query_position_ids)

            query_states = apply_rotary_pos_emb_single(query_states, query_cos, query_sin)
            key_states = apply_rotary_pos_emb_single(key_states, key_cos, key_sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        #fa_flag = 0
        st1 = time.time()
        st2 = time.time()
        st3 = time.time()
        st4 = time.time()
        token_len = 64 * 20
        near_end = (token_len-2) * 32
        
        if self.token_cnt % 8 == 0 or 1:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            #q1, q2, q3, _ = query_states.shape
            #_, _, k3, _ = key_states.shape
            #attn_weights = torch.rand(q1, q2, q3, k3)
            #attn_weights: (bsz,num_heads,q_len,kv_len)
            if attention_mask is not None:  # no matter the length, we just slice it
                #print(attention_mask.device)
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            #st2 = time.time()
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # Update KV Cache based on Heavy-Hitter Oracle
            #print("h2o shape:", attn_weights.shape)
            #st3 = time.time()
            #global cnt_for, cnt_tok
            #if past_key_value is not None:
            #    past_key_value.update_slimmingH(attn_weights, self.num_key_value_groups, self.layer_idx)
            if past_key_value is not None and 1==1:

                max_gen_len = 64
                if self.token_cnt == max_gen_len:  # max gen len
                    self.token_cnt = 0

                # 每32次forward等于完成1个token的处理
                #print(self.forward_call_count % 32)

                #print(self.forward_call_count)

                # 若生成的token数达到指定间隔，则执行一次H2O筛选
                if self.token_cnt % 8 == 0:
                    past_key_value.update_slimmingH(attn_weights, self.num_key_value_groups, self.layer_idx)
                self.token_cnt += 1

            #st4 = time.time()
            #fa_flag = 0
            #if not fa_flag:

            #attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        else:
            #FLASH ATTENTION 闪光注意力
            dropout_p = 0.0
            softmax_scale = None
            causal = True
            window_size = (-1, -1)
            softcap = 0.0
            alibi_slopes = None
            deterministic = False
            return_softmax = False

            #len_k = key_states.size(2)
            #print(len_k)
            query_states = query_states.permute(0, 2, 1, 3)
            key_states = key_states.permute(0, 2, 1, 3)
            value_states = value_states.permute(0, 2, 1, 3)

            attn_output = flash_attn_func(query_states, key_states, value_states, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, deterministic, return_softmax)
            #out: (batch_size, seqlen, nheads, headdim)

            #attn_output = attn_output.permute(0, 2, 3, 1)
            #if attn_output.size(3) < 1873:
            #attn_output = prepend_constant_block(attn_output, len_k)
            #(batch_size, nheads, headdim, seqlen)
            #print(attn_output.shape)
            #if past_key_value is not None:
                #.update_slimming(attn_output, self.num_key_value_groups, self.layer_idx)

            attn_output = attn_output.reshape(bsz, q_len, -1)
            #attn_output: (batch_size, query_length, hidden_size)
            
            #FLASH ATTENTION 闪光注意力
        ed = time.time()
        #print(f"fa: {bool(fa_flag)}")
        #print("attention run time: {:.5f} seconds".format(ed - st))
        time_seq_1.append(ed-st1)
        time_seq_2.append(ed-st2)
        time_seq_3.append(ed-st3)
        time_seq_4.append(ed-st4)

        if (len(time_seq_1)==near_end):
            ave_total = sum(time_seq_1) / len(time_seq_1)
            ave_4 = sum(time_seq_4) / len(time_seq_4)
            ave_3 = sum(time_seq_3) / len(time_seq_3)
            ave_2 = sum(time_seq_2) / len(time_seq_2)
            print("AVE ATTENTION TIME:")
            print("{:.10f}".format(ave_total))
            print("{:.10f}".format(ave_3-ave_4))
            print("times V ratio:")
            print("{:.2%}".format(ave_4 / ave_total))
            print("kvcache ratio:")
            print("{:.2%}".format((ave_3 - ave_4) / ave_total))
            print("softmax ratio:")
            print("{:.2%}".format((ave_2 - ave_3) / ave_total))
            print("QK ratio:")
            print("{:.2%}".format((ave_total - ave_2) / ave_total))


        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


def enable_h2ocache_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    past_seen_tokens = 0
    if use_cache:  # kept for BC (cache positions)
        if not isinstance(past_key_values, StaticCache):
            past_key_values = HHCache.from_legacy_cache(self.num_window_length, self.num_heavy_hitter_tokens, past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

    if cache_position is None:
        if isinstance(past_key_values, StaticCache):
            raise ValueError("cache_position is a required argument when using StaticCache.")
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        )
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

class H2OLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].self_attn = H2OLlamaAttention(config, layer_idx)

        self.model.forward = types.MethodType(enable_h2ocache_forward, self.model)
        self.model.num_heavy_hitter_tokens = config.num_heavy_hitter_tokens
        self.model.num_window_length = config.num_window_length
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if

        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(self.model.layers[0].self_attn, "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0]
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_key_values.get_seq_length()

            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                past_length = cache_position[0]
                cache_length = past_key_values[0].shape[2] # length = num_layers * 3 (3 -> key, value, score)
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
