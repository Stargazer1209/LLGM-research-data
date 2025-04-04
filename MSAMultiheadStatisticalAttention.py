import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiheadStatisticalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads 
        self.softmax = nn.Softmax(dim=1)
        self.to_single = nn.Linear(dim, dim, bias=qkv_bias)
        self.tau = nn.Parameter(torch.ones(num_heads, 1))
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, output_attentions=False):
        weighted_representation = rearrange(self.to_single(query), 'b n (h d) -> b h n d', h=self.num_heads)
        probability_distribution = self.softmax(torch.sum(F.normalize(weighted_representation, dim=-2) ** 2, dim=-1) * self.tau)
        
        attention_scores = 1. / (1 + torch.matmul((probability_distribution / (probability_distribution.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), weighted_representation ** 2))
        attention_scores = self.dropout(attention_scores)
        final_output = rearrange(-torch.mul(weighted_representation.mul(probability_distribution.unsqueeze(-1)), attention_scores), 'b h n d -> b n (h d)')
        
        return self.to_out(final_output)