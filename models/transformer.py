
import torch.nn as nn
from modules import TransformerEncoderLayer, Classifier, make_conv_pos

class Transformer(nn.Module):
    def build_encoder(self, args):
        layer = TransformerEncoderLayer(
            embed_dim=args.embed_dim, 
            ffn_embed_dim=args.ffn_embed_dim,
            num_heads=args.num_heads,
            activation=args.activation,
            dropout=args.dropout,
            bias=args.bias,
            normalize_before=args.normalize_before
        )
        return layer

    def __init__(self, args):
        super().__init__()
        self.normalize_before = args.normalize_before
        
        self.pos_conv = make_conv_pos(
                args.embed_dim,
                args.conv_pos,
                args.conv_pos_groups,
            )

        self.layer_norm = nn.LayerNorm(args.embed_dim)
        self.layers = nn.ModuleList(
            [self.build_encoder(args) for _ in range(args.num_encoder_layers)]
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = Classifier(args.embed_dim, args.num_classes, args.dropout, args.activation)

    def extract_features(self, x, key_padding_mask):
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.normalize_before:
            x = self.layer_norm(x)

        layer_results = []
        for i, layer in enumerate(self.layers):
            x, attn, _ = layer(x, key_padding_mask=key_padding_mask, need_weights=False)
            layer_results.append((x, attn))

        return x, layer_results
    
    def forward(self, x, pad_mask=None):
        x, layer_results = self.extract_features(x, pad_mask)

        if self.normalize_before:
            x = self.layer_norm(x)

        x = self.avgpool(x.transpose(-1, -2)).squeeze(dim=-1)
        pred = self.classifier(x)

        return pred

