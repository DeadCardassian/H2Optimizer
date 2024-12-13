from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import torch
import numpy as np
torch.set_printoptions(sci_mode=False)


def baseline_attention(query_states, key_states, value_states, causal=True):
    """
    使用普通矩阵乘法计算注意力机制，并输出 (batch_size, query_length, hidden_size) 的形状。
    
    Args:
        query_states: Q 张量，形状 (batch_size, num_heads, seq_len_q, head_dim)
        key_states: K 张量，形状 (batch_size, num_heads, seq_len_k, head_dim)
        value_states: V 张量，形状 (batch_size, num_heads, seq_len_k, head_dim)
        causal: 是否使用因果掩码，默认为 True
    
    Returns:
        softmax_A: softmax(QK^T) 的结果
        attention_output: QKV 的最终输出，形状 (batch_size, query_length, hidden_size)
    """
    # 计算 QK^T
    scores = torch.matmul(query_states, key_states.transpose(-1, -2))  # (batch_size, num_heads, seq_len_q, seq_len_k)
    softmax_scale = 1.0 / np.sqrt(query_states.size(-1))
    scores *= softmax_scale  # 缩放

    # 如果是因果注意力，应用上三角掩码
    if causal:
        seq_len_q, seq_len_k = scores.size(-2), scores.size(-1)
        causal_mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=scores.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(causal_mask, float('-inf'))

    # 计算 softmax(QK^T)
    softmax_A = torch.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

    # 计算 QKV
    attention_output = torch.matmul(softmax_A, value_states)  # (batch_size, num_heads, seq_len_q, head_dim)

    # 从已知信息计算 hidden_size
    num_heads = query_states.size(1)  # 获取注意力头的数量
    head_dim = query_states.size(-1)  # 每个头的维度
    hidden_size = num_heads * head_dim  # 计算隐藏层维度

    # 转换到 (batch_size, query_length, hidden_size)
    attention_output = attention_output.transpose(1, 2).contiguous()  # (batch_size, seq_len_q, num_heads, head_dim)
    attention_output = attention_output.view(attention_output.size(0), attention_output.size(1), hidden_size)

    return softmax_A, attention_output




device = torch.device("cuda")
'''
batch_size = 16
seqlen = 2048  #token序列长度
num_heads = 40
head_dim = 128  # 单个注意力的话是词嵌入维度
embed_dim = num_heads * head_dim  # 确保 embed_dim 是 num_heads 和 head_dim 的乘积
'''
batch_size = 1               # 通常为 1，因为问答任务多是逐条处理。 /k
num_heads = 40               # 对于 LLaMA-13B，每层有 40 个注意力头。 
num_key_value_heads = 40     # 问答任务中，通常 Q、K、V 的头数一致。
seq_len_q = 512              # 查询序列长度（一般为输入的 token 数，例如 128）。 /k
seq_len_k = 512              # 键序列长度（通常为上下文窗口大小，例如 2048）。
head_dim = 128               # 每个头的维度，对于 LLaMA-13B，head_dim = 5120 / 40 = 128。


q = torch.randn(batch_size, num_heads, seq_len_q, head_dim, 
                               device=device, dtype=torch.float16)
k = torch.randn(batch_size, num_key_value_heads, seq_len_k, head_dim, 
                              device=device, dtype=torch.float16)
v = torch.randn(batch_size, num_key_value_heads, seq_len_k, head_dim, 
                                device=device, dtype=torch.float16)

softmax_A, attention_output = baseline_attention(q, k, v, True)  #基准attention计算

# 输出结果
#print("benchmark softmax(QK^T) shape:", softmax_A.shape)
#print(softmax_A)
print("#######################")
print("benchmark Attention output (QKV) shape:", attention_output.shape)
print(attention_output)


# 设置其他参数
dropout_p = 0.0
softmax_scale = None
causal = True
window_size = (-1, -1)
softcap = 0.0
alibi_slopes = None
deterministic = False
return_softmax = False

q = q.permute(0, 2, 1, 3)
k = k.permute(0, 2, 1, 3)
v = v.permute(0, 2, 1, 3)

out = flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size, softcap, alibi_slopes, deterministic, return_softmax)
#out: (batch_size, seqlen, nheads, headdim)
out = out.reshape(batch_size, seq_len_q, -1)
#print(out.shape)

print("Output shape:", out.shape)
print(out)

result = torch.allclose(attention_output, out, atol=1e-2)
print(result)