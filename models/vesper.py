
import torch
from torch import Tensor, BoolTensor, FloatTensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import re
from typing import List, Optional
from einops.layers.torch import Rearrange

from modules import TransformerEncoderLayer, _get_activation_fn, make_conv_pos, MultiheadAttention

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)

def init_with_wavlm(model: nn.Module, num_layers: int=24, ckpt: str='PATH/TO/WavLM_CHECKPOINT', need_mask_emb: bool=True, style: list=['random'], info: str=''):    
    assert ckpt is not None
    data = torch.load(ckpt)
    state_dict = data['model']
    num_wavlm_layers = data['cfg']['encoder_layers']
    
    pop_dict = {}
    for key in state_dict.keys():
        if key.startswith('encoder.layers.') and not 'relative_attention_bias' in key:
            pop_dict[key] = state_dict[key]

    for key in pop_dict.keys():
        state_dict.pop(key)
    encoder_layers_modules = set([re.search(r'(?<=\d\.).*', key).group(0) for key in pop_dict.keys()])
    
    if isinstance(style, str):
        style = [style]
    if style[0] == 'uniform_average':
        assert num_wavlm_layers % num_layers == 0
        merge_rate = num_wavlm_layers // num_layers

        for module in encoder_layers_modules:
            for i in range(num_layers):
                state_dict[f'encoder.layers.{i}.{module}'] = (
                    torch.mean(
                    torch.stack(
                        [
                            pop_dict[f'encoder.layers.{i*merge_rate+j}.{module}'] for j in range(merge_rate)
                        ], dim=0), dim=0)
                )
    elif style[0] == 'custom_average':
        custom = style[1]
        assert len(custom) == num_layers
        
        for module in encoder_layers_modules:
            for i in range(num_layers):
                state_dict[f'encoder.layers.{i}.{module}'] = (
                    torch.mean(
                    torch.stack(
                        [
                            pop_dict[f'encoder.layers.{j}.{module}'] for j in range(custom[i][0], custom[i][1]+1)
                        ], dim=0), dim=0)
                )
    elif style[0] == 'uniform_extract':
        interval = num_wavlm_layers // num_layers

        for module in encoder_layers_modules:
            for i in range(num_layers):
                state_dict[f'encoder.layers.{i}.{module}'] = pop_dict[f'encoder.layers.{i*interval}.{module}']
    elif style[0] == 'custom_extract':
        custom = style[1]
        assert len(custom) == num_layers

        for module in encoder_layers_modules:
            for i in range(num_layers):
                state_dict[f'encoder.layers.{i}.{module}'] = pop_dict[f'encoder.layers.{custom[i]}.{module}']
    elif style[0] == 'identity_mapping':
        for module in encoder_layers_modules:
            for i in range(num_layers):
                state_dict[f'encoder.layers.{i}.{module}'] = pop_dict[f'encoder.layers.{i}.{module}']
    elif style[0] == 'random':
        state_dict = model.state_dict()
    else:
        raise NotImplementedError

    if not need_mask_emb:
        state_dict.pop('mask_emb')
        model.mask_emb = None
    
    # we remove the layer_normalization in the output of encoder
    state_dict.pop('encoder.layer_norm.weight')
    state_dict.pop('encoder.layer_norm.bias')
    
    print(f'vesper/{info}: Initialize with WavLM (style: {style}).')
    model.load_state_dict(state_dict)

    del state_dict
    del pop_dict

def init_with_ckpt(model: nn.Module, ckpt: str='PATH/TO/CHECKPOINT', name: str='vesper', need_mask_emb: bool=True, info: str='', device: str='cuda'):
    assert ckpt is not None

    if ckpt == '':
        print(f'{name}/{info}: No checkpoint found.')
        return
    
    if not need_mask_emb and hasattr(model, 'mask_emb'):
        model.mask_emb = None
    state_dict = torch.load(ckpt, map_location=device)['model']

    dit = {}
    for k, v in state_dict.items():
        if k.startswith(name):
            dit[k[len(name)+1:]] = v
    
    if not need_mask_emb and 'mask_emb' in dit.keys():
        dit.pop('mask_emb')

    # we remove the layer_normalization in the output of encoder
    dit.pop('encoder.layer_norm.weight', None)
    dit.pop('encoder.layer_norm.bias', None)

    if dit is None:
        print(f'{name}/{info}: No matching keys found in checkpoint: {ckpt}')
    else:
        print(f'{name}/{info}: Initialize with checkpoint: {ckpt}')
        model.load_state_dict(dit)

    del state_dict
    del dit

def apply_mask(x: Tensor, mask: BoolTensor, fill_value: Tensor, clone: bool=False):
    _x = x.clone() if clone else x
    _x[mask] = fill_value
    return _x

@torch.no_grad()
def get_rms(x: Tensor, frame_length: int = 2048, hop_length: int = 512):
    '''
    Inputs:
        x: (B, T), ``Tensor``, T dedotes the length of the time series.
    Outputs:
        rms: (B, Tf), ``Tensor``, Tf denotes the number of frames.
    '''
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.dim() == 1:
        x = x.unsqueeze(dim=0)

    n_frames = 1 + (x.shape[-1] - frame_length) // hop_length
    strides = torch.tensor(x.stride())

    shape = list(x.shape)[:-1] + [frame_length, n_frames]
    strides = list(strides) + [hop_length] #  * new_stride

    frame = torch.as_strided(x, size=shape, stride=strides)
    rms = torch.sqrt(torch.mean(torch.abs(frame)**2, dim=1, keepdim=False))

    return rms

@torch.no_grad()
def space_indices(indices: Tensor, space: int=1, maximum: int=1, already_sorted: bool=True):
    if not already_sorted:
        indices, _ = torch.sort(indices, descending=False)
    for i in range(0, len(indices)-1):
        if indices[i+1] - indices[i] < space:
            indices[i+1] = indices[i] + space
        if indices[i+1] > maximum:
            indices = indices[:i+1]
            break
    return indices

@torch.no_grad()
def get_random_mask(
        fea: Tensor,
        span: int=8, 
        max_num_span: int=10, 
        span_space: int=1, 
        real_length: Tensor=None, 
        max_mask_percentage: float=0.5
    ):
    mask = torch.full(fea.shape[:2], False, dtype=torch.bool, device=fea.device)

    if real_length is not None:
        num_span_per_sample = (real_length * max_mask_percentage / span).tolist()
        num_span_per_sample = [math.floor(s) if s < max_num_span else max_num_span for s in num_span_per_sample]
        valid_length = (real_length - span).tolist()
    else:
        valid_length = [fea.shape[1] - span] * fea.shape[0]
        num_span_per_sample = [max_num_span] * fea.shape[0]

    span_start = []
    for i, (valid) in enumerate(valid_length):
        num_span = num_span_per_sample[i]
        indices = torch.randperm(valid)[:num_span]
        
        indices = space_indices(indices, space=span+span_space, maximum=valid, already_sorted=False)

        if len(indices) < num_span:
            indices = torch.cat((indices, torch.randperm(valid, device=indices.device)))[:num_span]
        
        if (not num_span) or (not len(indices)):
            indices = torch.randperm(valid)[0].unsqueeze(dim=0)
            span_start.append(indices)
            mask[i][indices:real_length[i]] = True
        else: 
            span_start.append(indices)

            indices = torch.as_tensor(
                    [
                        indices[j] + offset
                        for j in range(num_span)
                        for offset in range(span)
                    ]
            )

            mask[i][indices] = True
    
    return mask, span_start

@torch.no_grad()
def get_rms_mask(
        rms: Tensor, 
        h_up: float=1.0, 
        h_down: float=0.5, 
        l_up: float=0.49, 
        l_down: float=0.2, 
        span: int=8, 
        max_num_span: int=10, 
        span_space: int=1, 
        real_length: Tensor=None, 
        max_mask_percentage: float=0.5
    ):
    mask = torch.full(rms.shape, False, dtype=torch.bool, device=rms.device)

    if real_length is not None:
        num_span_per_sample = (real_length * max_mask_percentage / span).tolist()
        num_span_per_sample = [math.floor(s) if s < max_num_span else max_num_span for s in num_span_per_sample]
        valid_length = (real_length - span).tolist()
    else:
        valid_length = [rms.shape[-1] - span] * rms.shape[0]
        num_span_per_sample = [max_num_span] * rms.shape[0]

    span_start = []
    for i, (row, valid) in enumerate(zip(rms, valid_length)):
        row = row[:valid]
        max_val = torch.max(row)
        h_down = h_down * max_val
        h_up = h_up * max_val
        l_down = l_down * max_val
        l_up = l_up * max_val
        h_mask = torch.logical_and(row >= h_down, row <= h_up)
        l_mask = torch.logical_and(row >= l_down, row <= l_up)
        h_indices = torch.nonzero(h_mask, as_tuple=False).squeeze(dim=1)
        l_indices = torch.nonzero(l_mask, as_tuple=False).squeeze(dim=1)
        
        num_span = num_span_per_sample[i]
        h_indices = h_indices[torch.randperm(len(h_indices))][:num_span//2]
        l_indices = l_indices[torch.randperm(len(l_indices))][:num_span-len(h_indices)]
        
        h_indices = space_indices(h_indices, space=span+span_space, maximum=valid)
        l_indices = space_indices(l_indices, space=span+span_space, maximum=valid)

        if len(h_indices) + len(l_indices) < num_span:
            indices = torch.cat((h_indices, l_indices, torch.randperm(valid, device=h_indices.device)))[:num_span]
        else:
            indices =torch.cat((h_indices, l_indices))
        
        if (not num_span) or (not len(indices)):
            indices = torch.randperm(valid)[0].unsqueeze(dim=0)
            span_start.append(indices)
            mask[i][indices:real_length[i]] = True
        else: 
            span_start.append(indices)

            indices = torch.as_tensor(
                    [
                        indices[j] + offset
                        for j in range(num_span)
                        for offset in range(span)
                    ]
            )

            mask[i][indices] = True
    
    return mask, span_start

@torch.no_grad()
def expand_mask(
        mask: Tensor, 
        expanded_span: int=40, 
        span_start: Tensor=None, 
        max_num_expanded_span: int=2, 
        span_space: int=1, 
        real_length: Tensor=None, 
        max_mask_percentage: float=0.5
    ):
    mask = torch.full_like(mask, False)

    if real_length is not None:
        num_span_per_sample = (real_length * max_mask_percentage / expanded_span).tolist()
        num_span_per_sample = [math.floor(s) if s < max_num_expanded_span else max_num_expanded_span for s in num_span_per_sample]
        valid_length = (real_length - expanded_span).tolist()
    else:
        valid_length = [mask.shape[-1] - expanded_span] * mask.shape[0]
        num_span_per_sample = [max_num_expanded_span] * mask.shape[0]

    expanded_span_start = []
    for i, (indices, valid) in enumerate(zip(span_start, valid_length)):
        indices = indices[indices < valid]
        num_expanded_span = num_span_per_sample[i]
        
        indices = space_indices(indices, space=expanded_span+span_space, maximum=valid, already_sorted=False)
        
        if len(indices) < num_expanded_span:
            indices = torch.cat((indices, torch.randperm(valid, device=indices.device)))[:num_expanded_span]
        else:
            indices = indices[torch.randperm(len(indices))][:num_expanded_span]
        
        if (not num_expanded_span) or (not len(indices)):
            indices = span_start[i][0].unsqueeze(dim=0)
            expanded_span_start.append(indices)
            mask[i][indices:real_length[i]] = True
        else:
            expanded_span_start.append(indices)

            indices = torch.as_tensor(
                    [
                        indices[j] + offset
                        for j in range(num_expanded_span)
                        for offset in range(expanded_span)
                    ]
            )

            mask[i][indices] = True

    return mask, expanded_span_start

def normalize(x: Tensor, p: int=2, dim: int=-1):
    return F.normalize(x, p, dim)

def masked_select(x: Tensor, mask: BoolTensor):
    '''
    Inputs:
        x: (B, T, C), ``Tensor``
        mask: (B, T), ```BoolTensor`
    Output:
        x: (-1, C),  `` Tensor``
    '''
    return x.masked_select(mask.unsqueeze(dim=-1)).view(-1, x.size(-1))

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: list = [(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2,
            dropout: float = 0.0,
            conv_bias: bool = False,
            mode: str = "default"
    ):
        super().__init__()

        def block(
                n_in,
                n_out,
                k,
                stride,
                conv_bias=False,
                is_layer_norm=False,
                is_group_norm=False
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        Rearrange("b c t -> b t c"),
                        nn.LayerNorm(dim, elementwise_affine=True),
                        Rearrange("b c t -> b t c"),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    conv_bias=conv_bias,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.frame_length = args.frame_length
        self.hop_length = args.hop_length
        self.h_up = args.h_up
        self.h_down = args.h_down
        self.l_up = args.l_up
        self.l_down = args.l_down
        self.small_span = args.small_span
        self.num_small_span = args.num_small_span
        self.large_span = args.large_span
        self.num_large_span = args.num_large_span
        self.span_space = args.span_space
        self.max_mask_percentage = args.max_mask_percentage
        self.encoder_layers = args.encoder_layers
        self.dropout = args.dropout
        self.pos_conv = make_conv_pos(args.encoder_embed_dim, args.conv_pos, args.conv_pos_groups)
        self.mask_depend_on_rms = args.mask_depend_on_rms

        self.relative_position_embedding = args.relative_position_embedding
        self.num_buckets = args.num_buckets
        self.max_distance = args.max_distance

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=args.encoder_embed_dim, 
                    ffn_embed_dim=args.ffn_embed_dim,
                    num_heads=args.num_heads,
                    activation=args.activation,
                    dropout=args.dropout,
                    bias=args.bias,
                    normalize_before=True,
                    has_relative_attention_bias=(self.relative_position_embedding and i == 0),
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    gru_rel_pos=args.gru_rel_pos,
                    qk_norm=args.qk_norm
                )
                for i in range(args.encoder_layers)
            ]
        )

        # self.layer_norm = nn.LayerNorm(args.encoder_embed_dim)

        self.apply(init_bert_params)
    
    def forward(self, x: Tensor, padding_mask=None, layer=None, student_pretraining=False, waveform=None, mask_emb=None):
        if student_pretraining:
            if padding_mask is not None:
                real_length = torch.sum(~padding_mask, dim=-1, dtype=torch.int)
            else:
                real_length = torch.full((x.size(0),), fill_value=x.size(1), device=x.device, dtype=torch.int)
            
            if self.mask_depend_on_rms:
                rms = get_rms(waveform, frame_length=self.frame_length, hop_length=self.hop_length)
                small_span_mask, span_start = get_rms_mask(
                    rms, 
                    self.h_up, 
                    self.h_down,
                    self.l_up,
                    self.l_down,
                    self.small_span,
                    self.num_small_span,
                    self.span_space,
                    real_length,
                    self.max_mask_percentage
                )
            else:
                small_span_mask, span_start = get_random_mask(
                    x,
                    self.small_span,
                    self.num_small_span,
                    self.span_space,
                    real_length,
                    self.max_mask_percentage
                )
            large_span_mask, expanded_span_start = expand_mask(
                small_span_mask,
                self.large_span,
                span_start,
                self.num_large_span,
                self.span_space,
                real_length,
                self.max_mask_percentage
            )
            interlayer = self.encoder_layers//2
            x, layer_results = self.extract_features(
                x,
                padding_mask,
                None,
                student_pretraining,
                interlayer,
                small_span_mask,
                large_span_mask,
                mask_emb
            )
        else:
            x, layer_results = self.extract_features(x, padding_mask, layer)

        # if layer is None:
        #     x = self.layer_norm(x)

        if student_pretraining:
            return x, layer_results, real_length, interlayer, small_span_mask, large_span_mask
        else:
            return x, layer_results

    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        student_pretraining=False,
        interlayer=0,
        small_span_mask=None,
        large_span_mask=None,
        mask_emb=None
    ):
        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_results = []
        attn_weights = None
        layer_results.append((x, attn_weights))
        pos_bias = None
        
        if student_pretraining:
            x = apply_mask(x, small_span_mask, mask_emb, clone=True)
            for i, layer in enumerate(self.layers):
                if i == interlayer:
                    x = apply_mask(x, large_span_mask, mask_emb, clone=True)
                x, attn_weights, pos_bias = layer(x, key_padding_mask=padding_mask, need_weights=True, pos_bias=pos_bias)
                layer_results.append((x, attn_weights))
        else:
            for i, layer in enumerate(self.layers):
                x, attn_weights, pos_bias = layer(x, key_padding_mask=padding_mask, need_weights=True, pos_bias=pos_bias)
                layer_results.append((x, attn_weights))
                if i == tgt_layer:
                    break

        return x, layer_results
    
class PredictionHead(nn.Module):
    '''A simple feed-forward network.

    Inputs:
        x: (B, T, input_dim), ``Tensor``
    Outputs:
        x: (B, T, output_dim), ``Tensor``
    '''
    def __init__(self, input_dim: int, output_dim: int, activation: str, norm_input: bool=True):
        super().__init__()
        self.norm_input = norm_input
        self.simple_ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            _get_activation_fn(activation, module=True),
            nn.Linear(input_dim//2, output_dim)
        )

    def forward(self, x: Tensor):
        if self.norm_input:
            x = F.layer_norm(x, [x.shape[-1]])
        return self.simple_ffn(x)

class Vesper(nn.Module):
    def __init__(self, args):
        super().__init__()
        feature_enc_layers = eval(args.conv_feature_layers)
        conv_embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(feature_enc_layers, mode=args.extractor_mode)
        self.layer_norm = nn.LayerNorm(conv_embed)
        self.post_extract_proj = nn.Linear(conv_embed, args.encoder_embed_dim)
        self.dropout_input = nn.Dropout(args.dropout_input)
        
        self.encoder = TransformerEncoder(args)

        self.mask_emb = nn.Parameter(FloatTensor(args.encoder_embed_dim).uniform_(), requires_grad=True)
        self.padding_mask = None
        self.normalize = args.normalize
        self.freeze_cnn = args.freeze_cnn

    def forward_padding_mask(self, features: Tensor, padding_mask: Tensor) -> Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        self.padding_mask = padding_mask.all(-1)
    
    def get_padding_mask(self):
        return self.padding_mask

    def forward(
            self, 
            waveform: Tensor, 
            padding_mask: Optional[Tensor]=None, 
            output_layer: Optional[int]=None,
            ret_layer_results: bool=False,
            student_pretraining=False
        ):
        '''
        Inputs:
            waveform: (B, T_audio), ``Tensor``
            padding_mask: (B, T_audio), ``BoolTensor``, key padding mask.
            output_layer: ``int``, varies between [1, 24].
            ret_layer_results: ``bool``, default False.
        Outputs:
            features: (B, T, C), ``Tensor``
            layers_rep: [feature_encoder_output, layer_1_output, layer_2_output, ..., layer_n_output], ``list``
        '''
        if self.normalize:
            waveform = F.layer_norm(waveform, [waveform.shape[-1]])

        if self.freeze_cnn:
            with torch.no_grad():
                features = self.feature_extractor(waveform)
        else:
            features = self.feature_extractor(waveform)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        features = self.post_extract_proj(features)
        features = self.dropout_input(features)

        if padding_mask is not None:
            self.forward_padding_mask(features, padding_mask)
        else:
            self.padding_mask = None

        if student_pretraining:
            features, layer_results, real_length, interlayer, small_span_mask, large_span_mask = self.encoder(
                features,
                padding_mask=self.padding_mask,
                layer=None,
                student_pretraining=True,
                waveform=waveform,
                mask_emb=self.mask_emb
            )
            return features, layer_results, real_length, interlayer, small_span_mask, large_span_mask
        else:
            features, layer_results = self.encoder(
                features, 
                padding_mask=self.padding_mask, 
                layer=None if output_layer is None else output_layer - 1
            )

            if ret_layer_results:
                features = (features, layer_results)
            return features

class VesperFinetuneWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.vesper = Vesper(args)

        if args.init_with_ckpt:
            init_with_ckpt(self.vesper, args.path_to_vesper, 'vesper', need_mask_emb=False)
        elif args.init_with_wavlm:
            init_with_wavlm(self.vesper, args.encoder_layers, args.path_to_wavlm, need_mask_emb=False, style=args.init_style)
        else:
            print('No initialization method specified. Initializing with random weights.')

        self.projector = nn.Linear(args.encoder_embed_dim, args.projector_dim)
        self.classifier = nn.Linear(args.projector_dim, args.num_classes)

        self.freeze_upstream = args.freeze_upstream
        # self.normalize = args.normalize

        if args.output_rep == 'weighted_sum':
            self.weights = nn.Parameter(torch.zeros(args.encoder_layers+1))
            print(f'Using weighted sum of {list(self.weights.shape)} representations as output representation.')
        elif args.output_rep == 'last_layer':
            self.weights = None
            print('Using last layer representation as output representation.')
        else:
            raise NotImplementedError(f'output_rep {args.output_rep} is not implemented.')

    def _weighted_sum(self, layer_results: list):
        stacked_feature = torch.stack(layer_results, dim=0)

        # if self.normalize:
        #     stacked_feature = F.layer_norm(stacked_feature, (stacked_feature.shape[-1],))

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(len(layer_results), -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature
    
    def forward(
            self, 
            waveform: Tensor, 
            padding_mask: Optional[Tensor]=None
        ):
        if self.freeze_upstream:
            with torch.no_grad():
                fea, layer_results = self.vesper(
                    waveform=waveform, padding_mask=padding_mask, ret_layer_results=True, student_pretraining=False)
        else:
            fea, layer_results = self.vesper(
                waveform=waveform, padding_mask=padding_mask, ret_layer_results=True, student_pretraining=False)

        if self.weights is not None:
            # layer_results = [layer_results[i+1][0] for i in range(len(layer_results)-1)]
            layer_results = [layer_results[i][0] for i in range(len(layer_results))]
            fea = self._weighted_sum(layer_results)
        
        padding_mask = self.vesper.get_padding_mask()
        if padding_mask is not None:
            real_length = torch.sum(~padding_mask, dim=-1, keepdim=True)
            fea[padding_mask] = 0.0
        else:
            real_length = torch.full((fea.size(0),1), fill_value=fea.size(1), dtype=fea.dtype, device=fea.device)
        
        fea = self.projector(fea)
        pooled_output = fea.sum(dim=1) / real_length
        pred = self.classifier(pooled_output)

        return pred

class Vesper_PretrainWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.vesper = Vesper(args)
        if args.init_with_ckpt:
            init_with_ckpt(self.vesper, args.path_to_vesper, 'vesper', info='student', device=args.device)
        elif args.init_with_wavlm:
            init_with_wavlm(self.vesper, args.encoder_layers, args.path_to_wavlm, style=args.init_style)
        else:
            print('No initialization method specified. Initializing with random weights.')

        self.l_predictor = PredictionHead(
            input_dim=args.encoder_embed_dim,
            output_dim=args.encoder_embed_dim,
            activation=args.activation,
            norm_input=True
        ) if args.enable_l_predictor else None

        self.h_predictor = PredictionHead(
            input_dim=args.encoder_embed_dim,
            output_dim=args.encoder_embed_dim,
            activation=args.activation,
            norm_input=True
        ) if args.enable_h_predictor else None

        self.x_predictor = PredictionHead(
            input_dim=args.encoder_embed_dim,
            output_dim=args.encoder_embed_dim,
            activation=args.activation,
            norm_input=True
        ) if args.enable_x_predictor else None

        if args.init_with_ckpt:
            if self.l_predictor is not None:
                init_with_ckpt(self.l_predictor, args.path_to_vesper, 'l_predictor', False)
            if self.h_predictor is not None:
                init_with_ckpt(self.h_predictor, args.path_to_vesper, 'h_predictor', False)
            if self.x_predictor is not None:
                init_with_ckpt(self.x_predictor, args.path_to_vesper, 'x_predictor', False)
        
        self.loss = nn.MSELoss()
    
    def cal_loss(self, pred: Tensor, target: Tensor, apply_norm: bool=True):
        if apply_norm:
            loss = self.loss(normalize(pred), normalize(target))
        else:
            loss = self.loss(pred, target)
        return loss

    def forward(
            self, 
            waveform: Tensor, 
            padding_mask: Optional[Tensor]=None,
            l_target: Tensor=None,
            h_target: Tensor=None,
        ):
        fea, layer_results, _, interlayer, small_span_mask, large_span_mask = self.vesper(
            waveform=waveform, padding_mask=padding_mask, ret_layer_results=True, student_pretraining=True)
        
        if self.l_predictor is not None:
            s_fea_l = masked_select(layer_results[interlayer][0], mask=small_span_mask)
            s_fea_l_pred = self.l_predictor(s_fea_l)
            t_fea_l = masked_select(l_target, mask=small_span_mask)
            l_loss = self.cal_loss(s_fea_l_pred, t_fea_l, apply_norm=False)
        else:
            l_loss = torch.zeros(1).to(fea.device)

        if self.h_predictor is not None:
            s_fea_h = masked_select(layer_results[-1][0], mask=large_span_mask)
            s_fea_h_pred = self.h_predictor(s_fea_h)
            t_fea_h = masked_select(h_target, mask=large_span_mask)
            h_loss = self.cal_loss(s_fea_h_pred, t_fea_h, apply_norm=False)
        else:
            h_loss = torch.zeros(1).to(fea.device)

        if self.x_predictor is not None:
            s_fea_x = fea.view(-1, fea.size(-1))
            s_fea_x_pred = self.x_predictor(s_fea_x)
            t_fea_x = l_target.view(-1, l_target.size(-1))
            x_loss = self.cal_loss(s_fea_x_pred, t_fea_x, apply_norm=False)
        else:
            x_loss = torch.zeros(1).to(fea.device)

        return l_loss, h_loss, x_loss

