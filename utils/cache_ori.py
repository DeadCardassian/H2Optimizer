import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)
time_seqc_1 = []
time_seqc_2 = []
time_seqc_3 = []
time_seqc_4 = []

time_seq_H1 = []

@dataclass
class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        raise NotImplementedError("Make sure to implement `get_seq_length` in a subclass.")

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        raise NotImplementedError("Make sure to implement `get_max_length` in a subclass.")

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Cache without size limit -> all cache is usable
        # Cache with size limit -> if the length cache plus the length of the new inputs is larger the maximum cache
        #   length, we will need to evict part of the cache (and thus not all cache is usable)
        max_length = self.get_max_length()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length + new_seq_length > max_length:
            return max_length - new_seq_length
        return previous_seq_length

    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_tokens` attribute is deprecated and will be removed in v4.41. Use the `cache_position` "
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.
    """

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


class SinkCache(Cache):
    """
    A cache that as described in the [Attention Sinks paper](https://arxiv.org/abs/2309.17453). It allows the model to
    generate beyond the length of its context window, without losing fluency in the conversation. As it discards past
    tokens, the model will lose the ability to generate tokens that depend on the context that was discarded.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_sink_tokens (`int`):
            The number of sink tokens. See the original paper for more information.
    """

    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cos_sin_cache = {}
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_key_rotary_pos_emb(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        rotated_key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)
        return rotated_key_states

    def _get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[-2] not in self.cos_sin_cache:
            # Upcast to float32 temporarily for better accuracy
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
            original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            self.cos_sin_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        return self.cos_sin_cache[key_states.shape[-2]]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.window_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # Optional kwargs for `SinkCache` -- needed on models using RoPE. `partial_rotation_size` is used on models
        # with partially rotated position embeddings, like Phi or Persimmon.
        sin = cache_kwargs.get("sin")
        cos = cache_kwargs.get("cos")
        partial_rotation_size = cache_kwargs.get("partial_rotation_size")
        using_rope = cos is not None and sin is not None

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        else:
            # Shifting cache
            keys_to_keep = self.key_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
            ]

            # On RoPE models, we need to recompute the Key rotation as the tokens are shifted
            if using_rope:
                rerotation_cos, rerotation_sin = self._get_rerotation_cos_sin(
                    key_states, cos[: self.window_length], sin[: self.window_length]
                )
                if partial_rotation_size is not None:
                    keys_to_keep, keys_pass = (
                        keys_to_keep[..., :partial_rotation_size],
                        keys_to_keep[..., partial_rotation_size:],
                    )
                keys_to_keep = self._apply_key_rotary_pos_emb(keys_to_keep, rerotation_cos, rerotation_sin)
                if partial_rotation_size is not None:
                    keys_to_keep = torch.cat((keys_to_keep, keys_pass), dim=-1)

            # Concatenate sink tokens, shifted & rotated tokens (if needed), and new tokens
            sink_keys = self.key_cache[layer_idx][:, :, : self.num_sink_tokens]
            self.key_cache[layer_idx] = torch.cat([sink_keys, keys_to_keep, key_states], dim=-2)

            sink_values = self.value_cache[layer_idx][:, :, : self.num_sink_tokens]
            values_to_keep = self.value_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
            ]
            self.value_cache[layer_idx] = torch.cat([sink_values, values_to_keep, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))


class HHCache(Cache):
    """
    A cache that apply heavy-hitter oracle (https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf).
    Only the heavy-hitter and the recent tokens are stored in the cache.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_hh_tokens (`int`):
            The number of heavy hitter tokens. See the original paper for more information.
    """

    def __init__(self, window_length: int, num_hh_tokens: int) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_length = window_length
        self.num_hh_tokens = num_hh_tokens
        self.accumulated_attention_scores: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx], self.accumulated_attention_scores[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.accumulated_attention_scores[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.window_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        accumulated_attention_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens

        if accumulated_attention_scores is not None:
            self.accumulated_attention_scores.append(accumulated_attention_scores)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_slimming(
        self,
        attention_scores: torch.Tensor,
        num_kv_groups: int,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Slimming the cache based on accumulated attention scores, only keep heavy-hitters + local tokens.

        Parameters:
            attention_scores (`torch.Tensor`):
                Attention_scores for current steps.
            num_kv_groups (`int`):
                The number of kv groups in repeat kv.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        #print("window_length: ", self.window_length)
        #print("num_kv_groups: ", num_kv_groups)
        #print("num_hh_tokens: ", self.num_hh_tokens)
        # Update score metrics (Accumulated attention scores)
        #attn_weights: (bsz,num_heads,q_len,kv_len)
        stc1 = time.perf_counter()
        if len(self.accumulated_attention_scores) <= layer_idx:
            self.accumulated_attention_scores.append(attention_scores.sum(2)[:,::num_kv_groups, :]) # [bs, num_heads, key_len]
        else:
            num_new_tokens = attention_scores.shape[2]  # q_len = 1, num_kv_groups = 4
            updated_attention_scores = attention_scores.sum(2)[:,::num_kv_groups, :] # [bs, num_heads, key_len]
            updated_attention_scores[:, :, :-num_new_tokens] += self.accumulated_attention_scores[layer_idx]
            self.accumulated_attention_scores[layer_idx] = updated_attention_scores
        # 存储更新每个token的累积注意力分数
        stc2 = time.perf_counter()
        # Update KV Cache
        #print(f"{self.get_seq_length(layer_idx)} vs {self.window_length}") 257>256
        if self.get_seq_length(layer_idx) > self.window_length:
            
            seq_scores = self.accumulated_attention_scores[layer_idx][:, :, :-self.window_length + self.num_hh_tokens]
            _, keep_hh_index = torch.topk(seq_scores, self.num_hh_tokens, dim=-1)  #保留了前面用于筛选H2的序列
            keep_hh_index = keep_hh_index.sort().values  # [batch_size, num_heads, num_hh_tokens]
            stc3 = time.perf_counter()
            keep_local_index = torch.arange(self.get_seq_length(layer_idx) - self.window_length + self.num_hh_tokens, self.get_seq_length(layer_idx), device=keep_hh_index.device).repeat(keep_hh_index.shape[0], keep_hh_index.shape[1], 1)
            keep_index = torch.cat([keep_hh_index, keep_local_index], dim=-1) #合并H2和最近索引

            mask = torch.zeros(self.accumulated_attention_scores[layer_idx].shape, dtype=torch.bool).to(keep_hh_index.device)
            mask = mask.scatter(-1, keep_index, 1)
            stc4 = time.perf_counter()
            bsz, num_heads, _, head_dim = self.key_cache[layer_idx].shape
            self.key_cache[layer_idx] = self.key_cache[layer_idx][mask].view(bsz, num_heads, -1, head_dim)
            self.value_cache[layer_idx] = self.value_cache[layer_idx][mask].view(bsz, num_heads, -1, head_dim)
            #stc4 = time.perf_counter()
            #print(f"key_cache updated shape: {self.key_cache[layer_idx].shape}")
            #print(f"key_cache updated memory: {self.key_cache[layer_idx].element_size() * self.key_cache[layer_idx].nelement() / 1024 / 1024:.2f} MB")
            self.accumulated_attention_scores[layer_idx] = self.accumulated_attention_scores[layer_idx][mask].view(bsz, num_heads, -1)

            #print(f"accumulated_attention_scores updated shape: {self.accumulated_attention_scores[layer_idx].shape}")
            #print(f"accumulated_attention_scores updated memory: {self.accumulated_attention_scores[layer_idx].element_size() * self.accumulated_attention_scores[layer_idx].nelement() / 1024 / 1024:.2f} MB")
        
        edc = time.perf_counter()

        time_seqc_1.append(edc-stc1)
        time_seqc_2.append(edc-stc2)
        time_seqc_3.append(edc-stc3)
        time_seqc_4.append(edc-stc4)
        token_len = 64
        near_end = (token_len-2) * 32
        if (len(time_seqc_1)==near_end):
            ave_total = sum(time_seqc_1) / len(time_seqc_1)
            ave_4 = sum(time_seqc_4) / len(time_seqc_4)
            ave_3 = sum(time_seqc_3) / len(time_seqc_3)
            ave_2 = sum(time_seqc_2) / len(time_seqc_2)
            print("AVE update TIME:")
            print("{:.10f}".format(ave_total))
            #print("{:.10f}".format(ave_3-ave_4))
            print("c4 ratio:")
            print("{:.2%}".format(ave_4 / ave_total))
            print("c3 ratio:")
            print("{:.2%}".format((ave_3 - ave_4) / ave_total))
            print("c2 ratio:")
            print("{:.2%}".format((ave_2 - ave_3) / ave_total))
            print("c1 ratio:")
            print("{:.2%}".format((ave_total - ave_2) / ave_total))


    def update_slimmingH(
        self,
        attention_scores: torch.Tensor,
        num_kv_groups: int,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stH1 = time.perf_counter()
        """
        Slimming the cache based on accumulated attention scores, only keep heavy-hitters + local tokens.
        """
        # 将当前step的注意力分数按num_kv_groups进行降采样并求和
        new_scores = attention_scores.sum(2)[:, ::num_kv_groups, :]

        # 更新累计的注意力分数
        if len(self.accumulated_attention_scores) <= layer_idx:
            self.accumulated_attention_scores.append(new_scores)
        else:
            old_scores = self.accumulated_attention_scores[layer_idx]
            num_new_tokens = new_scores.shape[-1]

            updated_attention_scores = new_scores.clone()
            old_len = old_scores.shape[-1]
            new_len = updated_attention_scores.shape[-1]

            # 检查是否有需要相加的部分（若new_len == num_new_tokens，则 :-num_new_tokens会产生空切片）
            if new_len > num_new_tokens:
                expected_old_len = new_len - num_new_tokens
                if old_len == expected_old_len:
                    updated_attention_scores[:, :, :-num_new_tokens] += old_scores
                else:
                    # 如果old_scores长度与预期不一致，说明逻辑或数据有问题
                    # 可以打印信息或进行相应的回退处理，这里简单跳过
                    pass

            self.accumulated_attention_scores[layer_idx] = updated_attention_scores

        # 如果缓存序列长度超过window_length，则进行精简
        if self.get_seq_length(layer_idx) > self.window_length:
            acc_scores = self.accumulated_attention_scores[layer_idx]
            total_len = acc_scores.shape[-1]

            # Heavy-hitter分数范围（老旧上下文中选择）
            hh_end = total_len - self.window_length + self.num_hh_tokens
            if hh_end > 0:  # 避免负数/无效索引
                seq_scores = acc_scores[:, :, :hh_end]
                # topk选出heavy-hitters的索引
                _, keep_hh_index = torch.topk(seq_scores, self.num_hh_tokens, dim=-1, largest=True, sorted=True)
            else:
                # 如果hh_end <=0，表示window_length超过当前acc长度，不需要筛选，直接使用全部
                keep_hh_index = torch.arange(total_len, device=acc_scores.device)
                keep_hh_index = keep_hh_index.unsqueeze(0).unsqueeze(0).expand(
                    acc_scores.size(0), acc_scores.size(1), -1
                )

            # local tokens索引（最新的token）
            start_local = max(total_len - self.window_length + self.num_hh_tokens, 0)
            end_local = total_len
            keep_local_index = torch.arange(start_local, end_local, device=keep_hh_index.device)
            keep_local_index = keep_local_index.unsqueeze(0).unsqueeze(0).expand(
                keep_hh_index.size(0), keep_hh_index.size(1), -1
            )

            # 合并heavy-hitters与local indices
            keep_index = torch.cat([keep_hh_index, keep_local_index], dim=-1)  # [bs, num_heads, keep_len]

            # 使用gather直接索引保留的token
            bsz, num_heads, seq_len, head_dim = self.key_cache[layer_idx].shape
            gather_index = keep_index.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            self.key_cache[layer_idx] = torch.gather(self.key_cache[layer_idx], 2, gather_index)
            self.value_cache[layer_idx] = torch.gather(self.value_cache[layer_idx], 2, gather_index)
            self.accumulated_attention_scores[layer_idx] = torch.gather(acc_scores, 2, keep_index)

        edH = time.perf_counter()
        time_seq_H1.append(edH-stH1)
        token_len = 64
        near_end = (token_len-2) * 32
        if (len(time_seq_H1)==near_end):
            ave_total = sum(time_seq_H1) / len(time_seq_H1)
            #ave_4 = sum(time_seqc_4) / len(time_seqc_4)
            #ave_3 = sum(time_seqc_3) / len(time_seqc_3)
            #ave_2 = sum(time_seqc_2) / len(time_seqc_2)
            print("AVE updateH TIME:")
            print("{:.10f}".format(ave_total))


    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx], self.accumulated_attention_scores[layer_idx],))
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, window_length: int, num_hh_tokens: int, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls(window_length, num_hh_tokens)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values) // 3):
                key_states = past_key_values[layer_idx * 3]
                value_states = past_key_values[layer_idx * 3 + 1]
                accumulated_attention_scores = past_key_values[layer_idx * 3 + 2]
                cache.update(key_states, value_states, layer_idx, accumulated_attention_scores=accumulated_attention_scores)
        return cache

    def evict_for_space(self, space_needed: int):
        num_layers = len(self.key_cache)

        # Update score metrics (Accumulated attention scores)
        if len(self.accumulated_attention_scores) < num_layers:
            raise ValueError("The accumulated_attention_scores should be updated before evicting the cache.")

        for layer_idx in range(num_layers):
            # Update KV Cache, Evict for new coming prompts
            if self.get_seq_length(layer_idx) + space_needed > self.window_length:
                if self.window_length - self.num_hh_tokens <= space_needed:
                    raise ValueError("The space_needed should be less than the window_length - num_hh_tokens.")

                seq_scores = self.accumulated_attention_scores[layer_idx][:, :, :-self.window_length + self.num_hh_tokens + space_needed]
                _, keep_hh_index = torch.topk(seq_scores, self.num_hh_tokens, dim=-1)
                keep_hh_index = keep_hh_index.sort().values

                keep_local_index = torch.arange(self.get_seq_length(layer_idx) - self.window_length + self.num_hh_tokens + space_needed, self.get_seq_length(layer_idx), device=keep_hh_index.device).repeat(keep_hh_index.shape[0], keep_hh_index.shape[1], 1)
                keep_index = torch.cat([keep_hh_index, keep_local_index], dim=-1)

                mask = torch.zeros(self.accumulated_attention_scores[layer_idx].shape, dtype=torch.bool).to(keep_hh_index.device)
                mask = mask.scatter(-1, keep_index, 1)

                bsz, num_heads, _, head_dim = self.key_cache[layer_idx].shape
                self.key_cache[layer_idx] = self.key_cache[layer_idx][mask].view(bsz, num_heads, -1, head_dim)
                self.value_cache[layer_idx] = self.value_cache[layer_idx][mask].view(bsz, num_heads, -1, head_dim)
                self.accumulated_attention_scores[layer_idx] = self.accumulated_attention_scores[layer_idx][mask].view(bsz, num_heads, -1)




class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the `max_position_embeddings`, `hidden_size` and `num_attention_heads`
            required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        self.key_cache: torch.Tensor = torch.zeros(cache_shape, dtype=self.dtype, device=device)
        self.value_cache: torch.Tensor = torch.zeros(cache_shape, dtype=self.dtype, device=device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for. Kept for backward compatibility
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` just needs the `q_len`
                to know how much of the cache it should overwrite.

        Return:
            A tuple containing the updated key and value states.
        """
        new_cache_positions = cache_kwargs.get("cache_position")
        k_out = self.key_cache
        v_out = self.value_cache

        k_out[:, :, new_cache_positions] = key_states
        v_out[:, :, new_cache_positions] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model. `layer_idx` kept for BC"""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: This is error prone, a filled cache may be `0.0`. Let's use a stateless integer instead, after
        # https://github.com/pytorch/pytorch/issues/120248 is fixed
        return (self.key_cache[0, 0].any(dim=-1)).sum()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return self.max_cache_len

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        device = self.key_cache.device
        self.key_cache = self.key_cache.index_select(0, beam_idx.to(device))
        device = self.value_cache.device
        self.value_cache = self.value_cache.index_select(0, beam_idx.to(device))

    def to_legacy_cache(self):
        """Dummy function for BC. We have to keep it because otherwise the call in the forward of models will break it"""
        return None

