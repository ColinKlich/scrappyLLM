# llama.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super().__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))
        self.eps = eps

    def forward(self, x):
        norm_x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_normed = x * norm_x
        return self.scale * x_normed

class RoPEMaskedAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_k = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.w_v = nn.Linear(config['d_model'], config['d_model'], bias=False)
        self.register_buffer('R', self.get_rotary_matrix(config['context_window'], config['d_model']))

    @staticmethod
    def get_rotary_matrix(context_window, embedding_dim):
        R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
        for position in range(context_window):
            for i in range(embedding_dim//2):
                theta = 10000. ** (-2.*(i - 1) / embedding_dim)
                m_theta = position * theta
                R[position, 2*i,2*i] = np.cos(m_theta)
                R[position, 2*i,2*i+1] = -np.sin(m_theta)
                R[position, 2*i+1,2*i] = np.sin(m_theta)
                R[position, 2*i+1,2*i+1] = np.cos(m_theta)
        return R
    
    def forward(self, x, return_attn_weights=False):
        b, m, d = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        
        q_rotated = (torch.bmm(q.transpose(0, 1), self.R[:m, ...])).transpose(0, 1)
        k_rotated = (torch.bmm(k.transpose(0, 1), self.R[:m, ...])).transpose(0, 1)

        activations = F.scaled_dot_product_attention(
            q_rotated, k_rotated, v, dropout_p=0.0, is_causal=True
        )

        if return_attn_weights:
            attn_mask = torch.tril(torch.ones((m,m)), diagonal=0)
            attn_weights = torch.bmm(q_rotated, k_rotated.transpose(1,2)) / np.sqrt(d) + attn_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            return activations, attn_weights
        return activations
    
class RoPEMaskedMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([
            RoPEMaskedAttentionHead(config) for _ in range(config['n_heads'])
        ])
        self.linear = nn.Linear(config['n_heads'] * config['d_model'], config['d_model'])
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x): 
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out

class LlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rms = RMSNorm((config['d_model'],))
        self.attention = RoPEMaskedMultiheadAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
        )

    def forward(self, x):
        x_norm = self.rms(x)
        x = x + self.attention(x_norm)
        x_norm = self.rms(x)
        x = x + self.feedforward(x_norm)
        return x

class LlamaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )
        self.rms_final = RMSNorm((config['d_model'],))
        self.last_linear = nn.Linear(config['d_model'], config['vocab_size'])
        
        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LlamaModel initialized with {total_params:,} parameters")
        
    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        x = self.llama_blocks(x)
        x = self.rms_final(x)
        logits = self.last_linear(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config['vocab_size']), 
                targets.reshape(-1),
                ignore_index=0
            )
            return logits, loss
        else:
            return logits
    
    def generate(self, prompt_ids, max_new_tokens=100, temperature=0.8, top_k=50):
        """
        Generate text from a prompt.
        
        Args:
            prompt_ids: Tensor of shape (batch_size, seq_len) with input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter (0 = no top-k filtering)
            
        Returns:
            Tensor of shape (batch_size, seq_len + max_new_tokens) with generated tokens
        """
        self.eval()
        generated = prompt_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get context window
                context = generated[:, -self.config['context_window']:]
                
                # Forward pass
                logits = self(context)
                
                # Get last token logits
                last_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = last_logits < torch.topk(last_logits, top_k)[0][..., -1, None]
                    last_logits[indices_to_remove] = -float('Inf')
                
                # Sample from distribution
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append token
                generated = torch.cat([generated, next_token], dim=-1)
        
        self.train()
        return generated