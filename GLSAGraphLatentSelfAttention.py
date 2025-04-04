import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GraphLatentSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
        t_emb_size=0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.t_emb_size = t_emb_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Parameter setup
        self.encoder_decoder_attention = encoder_decoder_attention
        self.q_lora_rank = 0  # Can be configured as needed
        self.kv_lora_rank = embed_dim // 4  # Example value
        self.t_emb_lora_rank = min(t_emb_size, 16)  # Low-rank for type embeddings
        
        # Query projection (using LoRA if enabled)
        if self.q_lora_rank == 0:
            self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        else:
            self.wq_a = nn.Linear(embed_dim, self.q_lora_rank, bias=bias)
            self.q_norm = nn.LayerNorm(self.q_lora_rank)
            self.wq_b = nn.Linear(self.q_lora_rank, num_heads * self.head_dim, bias=bias)
        
        # Key-Value projection with LoRA
        self.wkv_a = nn.Linear(embed_dim, self.kv_lora_rank, bias=bias)
        self.kv_norm = nn.LayerNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, num_heads * self.head_dim, bias=bias)
        self.t_emb_lookup = nn.Embedding(self.t_emb_size, self.t_emb_lora_rank)
        self.t_emb_norm = nn.LayerNorm(self.t_emb_lora_rank)
        self.t_emb_projection = nn.Linear(self.t_emb_lora_rank, 1, bias=bias)
        self.t_emb_scale = nn.Parameter(torch.ones(1) * 0.1)  # Scaling factor for type embeddings
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)
        
        # Cache related
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"
        self.softmax_scale = self.head_dim ** -0.5

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions=False,
        adj_matrix_emb=None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention; tgt_len, bsz, embed_dim = query.size(); assert embed_dim == self.embed_dim and list(query.size()) == [tgt_len, bsz, embed_dim]
        
        # Handle previous states for incremental decoding
        saved_state, layer_state, key = (s := layer_state.get(self.cache_key, {}), layer_state, None if "prev_key" in s and static_kv else key) if layer_state is not None else (None, {}, key)

        # Calculate query projection and reshape for multi-head attention
        q = (self.q_proj(query) if self.q_lora_rank == 0 else self.wq_b(self.q_norm(self.wq_a(query)))).view(tgt_len, bsz, self.num_heads, self.head_dim)
        
        # Determine key and value vectors
        key_source = key if static_kv and key is not None else query
        if key_source is not None:
            # Apply LoRA projection for both k and v together
            kv_latent = self.kv_norm(self.wkv_a(key_source))
            kv_proj = self.wkv_b(kv_latent)
            
            # Reshape for multi-head attention
            shape = (key_source.size(0), key_source.size(1), self.num_heads, self.head_dim)
            k = v = kv_proj.view(*shape)
        else:
            k = v = None
        
        # Reshape tensors and handle cache state
        q = q.permute(1, 2, 0, 3).contiguous().view(bsz * self.num_heads, tgt_len, self.head_dim)
        if k is not None: k = k.permute(1, 2, 0, 3).contiguous().view(bsz * self.num_heads, -1, self.head_dim)
        if v is not None: v = v.permute(1, 2, 0, 3).contiguous().view(bsz * self.num_heads, -1, self.head_dim)
        if saved_state is not None: k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache state  
        src_len = k.size(1)
        layer_state[self.cache_key] = {"prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
                                       "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
                                       "prev_key_padding_mask": key_padding_mask if not static_kv else None}

        # Calculate attention scores
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.softmax_scale; assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        # Reshape attention weights for batch processing
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            T = torch.where(attn_mask == 60, torch.zeros_like(attn_mask), attn_mask).long()
            T_emb = self.t_emb_projection(self.t_emb_norm(self.t_emb_lookup(T))).squeeze(-1) * torch.sigmoid(self.t_emb_scale)
            mask = torch.where(attn_mask == 60, torch.tensor(-10000.0, device=attn_mask.device), 
                              torch.zeros_like(attn_mask, dtype=torch.float))
            attn_weights = attn_weights + T_emb.unsqueeze(1) + mask.unsqueeze(1)

        # Handle key padding mask
        if key_padding_mask is not None and key_padding_mask.dim() > 0:
            assert key_padding_mask.size()[:2] == (bsz, src_len)
            attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # Reshape back to original dimensions
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # Apply attention mechanism
        attn_probs = F.dropout(F.softmax(attn_weights, dim=-1), p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)

        # Reshape and project output
        attn_output = self.out_proj(attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, self.num_heads * self.head_dim))

        # Format attention weights for return value
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) if output_attentions else None
            
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # Process saved key and value states
        for name, tensor in [("key", k), ("value", v)]:
            prev = saved_state.get(f"prev_{name}")
            if prev is not None:
                prev = prev.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    locals()[name[0]] = prev  # Set k or v
                else:
                    locals()[name[0]] = torch.cat([prev, locals()[name[0]]], dim=1) if locals()[name[0]] is not None else None
        
        # Process key padding mask
        prev_mask = saved_state.get("prev_key_padding_mask")
        if prev_mask is not None:
            key_padding_mask = prev_mask if static_kv else torch.cat([prev_mask, key_padding_mask], dim=1)
        elif key_padding_mask is not None and not static_kv:
            filler = torch.zeros(bsz, k.size(1) - key_padding_mask.size(1), 
                                device=key_padding_mask.device, dtype=key_padding_mask.dtype)
            key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
            
        return k, v, key_padding_mask
