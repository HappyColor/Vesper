
import torch.nn as nn
import torch.nn.functional as F
from modules import MultiheadAttention
from modules.activation import _get_activation_fn

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        ffn_embed_dim, 
        num_heads, 
        activation, 
        dropout, 
        bias, 
        normalize_before, 
        has_relative_attention_bias: bool = False,
        num_buckets: int = 32,
        max_distance: int = 128,
        gru_rel_pos: bool = False,
        qk_norm: bool = False
    ):
        super().__init__()
        self.dropout = dropout
        self.normalize_before = normalize_before

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn = MultiheadAttention(
            embed_dim, num_heads, None, None, dropout, bias, has_relative_attention_bias, num_buckets, max_distance, gru_rel_pos, qk_norm
        )

        # Feed-Forward Network
        self.activation_fn = _get_activation_fn(activation)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None, pos_bias=None, need_weights=False):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn, pos_bias = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            position_bias=pos_bias,
            need_weights=need_weights,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn, pos_bias

