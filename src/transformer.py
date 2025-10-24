import torch
import math
import src.utils as utils
import einops

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2/(in_features+out_features))
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype),
                mean=0.0,
                std=std,
                a=-3*std,
                b=3*std,
            ))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oi,...i->...o", self.weight, x)

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                mean=0.0,
                std=1,
                a=-3,
                b=3,
            ))
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True)/x.shape[-1]+self.eps)
        norm = torch.einsum("...i,i->...i", x, self.weight) / rms
        return norm.to(in_type)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.einsum("...i,...i->...i", 
            silu(self.w1(x)), 
            self.w3(x)
        ))

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        thetas = 1.0 / (theta ** (torch.arange(0.0, d_k, 2, device=device) / d_k)) # (d_k/2,)
        idxs = torch.arange(max_seq_len, device=device) # (max_seq_len,)
        theta_iks = torch.einsum("i,j->ij", idxs, thetas) # (max_seq_len, d_k/2)

        full_theta_iks = torch.zeros(max_seq_len, d_k, device=device) # (max_seq_len, d_k)
        for i in range(d_k//2):
            full_theta_iks[:, 2*i] = theta_iks[:, i]
            full_theta_iks[:, 2*i+1] = theta_iks[:, i]
            
        self.cos_iks = torch.cos(full_theta_iks) # (max_seq_len, d_k)
        self.sin_iks = torch.sin(full_theta_iks) # (max_seq_len, d_k)
    
    def swap_neg(self, x: torch.Tensor) -> torch.Tensor:
        x_ = torch.zeros_like(x, device=x.device)
        # for i in range(0, x.shape[-1], 2):
        #     x_[..., i] = -x[..., i+1]
        #     x_[..., i+1] = x[..., i]
        x_[..., 0::2] = -x[..., 1::2]  # Even indices get negated odd values
        x_[..., 1::2] = x[..., 0::2]   # Odd indices get even values
        return x_

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        return x * self.cos_iks[token_positions] + self.swap_neg(x) * self.sin_iks[token_positions]

def scaled_dot_product_attention(
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
    float_mask = torch.zeros_like(mask.float()).masked_fill_(torch.logical_not(mask), float('inf'))
    pre_softmax = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(Q.shape[-1]) - float_mask
    return torch.einsum("...qk,...kd->...qd", utils.softmax(pre_softmax, dim=-1), V)

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding=None):
        super().__init__()
        d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_k = d_k
        self.q_proj = Linear(d_k * num_heads, d_model)
        self.k_proj = Linear(d_k * num_heads, d_model)
        self.v_proj = Linear(d_k * num_heads, d_model)
        self.output_proj = Linear(d_model, d_k * num_heads)
        self.rope = rope
    
    def forward(self, x, token_positions=None) -> torch.Tensor:
        q_proj = self.q_proj(x) # (batch_size, seq_len, d_model)
        k_proj = self.k_proj(x) # (batch_size, seq_len, d_model)
        v_proj = self.v_proj(x) # (batch_size, seq_len, d_model)

        num_heads, d_k = self.num_heads, self.d_k

        # Reshape for multi-head attention
        q = einops.rearrange(q_proj, '... s (h d_k) -> ... h s d_k', h=num_heads, d_k=d_k) # (batch_size, num_heads, seq_len, d_k)
        k = einops.rearrange(k_proj, '... s (h d_k) -> ... h s d_k', h=num_heads, d_k=d_k) # (batch_size, num_heads, seq_len, d_k)
        v = einops.rearrange(v_proj, '... s (h d_v) -> ... h s d_v', h=num_heads, d_v=d_k) # (batch_size, num_heads, seq_len, d_k)

        if self.rope:
            assert token_positions is not None
        #     for h in range(self.num_heads):
        #         q_proj[..., h*d_k:(h+1)*d_k] = self.rope(
        #             q_proj[..., h*d_k:(h+1)*d_k], 
        #             token_positions,
        #         )
        #         k_proj[..., h*d_k:(h+1)*d_k] = self.rope(
        #             k_proj[..., h*d_k:(h+1)*d_k], 
        #             token_positions,
        #         )
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Create mask
        seq_len = x.shape[-2]
        mask = torch.logical_not(torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)).bool()
        
        # Vectorized attention computation
        # out = torch.zeros_like(v_proj)
        # mask = torch.logical_not(torch.triu(torch.ones(x.shape[-2], x.shape[-2]), diagonal=1)).bool()
        # for h in range(self.num_heads):
        #     out[..., :, h*d_k:(h+1)*d_k] = scaled_dot_product_attention(
        #         q_proj[..., h*d_k:(h+1)*d_k],
        #         k_proj[..., h*d_k:(h+1)*d_k],
        #         v_proj[..., h*d_k:(h+1)*d_k],
        #         mask,
        #     )
        out = scaled_dot_product_attention(q, k, v, mask)
        
        # Reshape back to original format
        out = einops.rearrange(out, '... h s d_k -> ... s (h d_k)') # (batch_size, seq_len, d_model)
        
        return self.output_proj(out)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x) -> torch.Tensor:
        token_positions = torch.arange(x.shape[-2])
        h = self.attn(self.ln1(x), token_positions) + x
        h = self.ffn(self.ln2(h)) + h
        return h

class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        rope = RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length)
        self.layers = torch.nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, indices) -> torch.Tensor:
        h = self.token_embeddings(indices)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_final(h)
        h = self.lm_head(h)
        return h
