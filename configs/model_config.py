
from yacs.config import CfgNode as CN
_C = CN(new_allowed=True)

###############
# Transformer #
###############
_C.Transformer = CN(new_allowed=True)

_C.Transformer.num_encoder_layers = 4
_C.Transformer.embed_dim = 1024
_C.Transformer.ffn_embed_dim = 512
_C.Transformer.num_heads = 8
_C.Transformer.activation = 'gelu'
_C.Transformer.dropout = 0.1
_C.Transformer.bias = True
_C.Transformer.normalize_before = True

# positional embeddings
_C.Transformer.conv_pos = 128
_C.Transformer.conv_pos_groups = 16

##########
# Vesper #
##########
_C.Vesper = CN(new_allowed=True)

# mainstream model
_C.Vesper.encoder_layers= 4
_C.Vesper.encoder_embed_dim = 1024
_C.Vesper.ffn_embed_dim = 4096
_C.Vesper.num_heads = 16
_C.Vesper.activation = 'gelu'
_C.Vesper.dropout = 0.1
_C.Vesper.bias = True
_C.Vesper.normalize = True
_C.Vesper.normalize_before = True
_C.Vesper.relative_position_embedding = True
_C.Vesper.qk_norm = False  # query/key (QK) normalization

# predictor
_C.Vesper.enable_l_predictor = True
_C.Vesper.enable_h_predictor = True
_C.Vesper.enable_x_predictor = True

# FinetuneWrapper
_C.Vesper.projector_dim = 256
_C.Vesper.output_rep = 'weighted_sum' # 'weighted_sum' / 'last_layer'

# initiliaze with wavlm
_C.Vesper.init_with_wavlm = True
_C.Vesper.init_style = ['uniform_extract']   # ['custom_average', [(0, 1), (2, 5), (6, 13), (14, 23)]], ['custom_extract', [0, 5, 11, 17]]
_C.Vesper.path_to_wavlm = ['/148Dataset/data-chen.weidong/pre_trained_model/wavlm/WavLM-Base.pt', '/148Dataset/data-chen.weidong/pre_trained_model/wavlm/WavLM-Large.pt']

# initiliaze with other pre-trained model
_C.Vesper.init_with_ckpt = False
_C.Vesper.path_to_vesper = ''

# rms-based mask
_C.Vesper.mask_depend_on_rms = True
_C.Vesper.frame_length = 400 # 16000 * 0.025
_C.Vesper.hop_length = 320   # 16000 * 0.020
_C.Vesper.span_space = 1
_C.Vesper.h_up = 1.0
_C.Vesper.h_down = 0.5
_C.Vesper.l_up = 0.49
_C.Vesper.l_down = 0.2
_C.Vesper.small_span = 8
_C.Vesper.num_small_span = 20
_C.Vesper.large_span = 40
_C.Vesper.num_large_span = 4
_C.Vesper.max_mask_percentage = 0.64

# positional embedding
_C.Vesper.conv_pos = 128
_C.Vesper.conv_pos_groups = 16

# bucket relative position embedding
_C.Vesper.num_buckets = 320
_C.Vesper.max_distance = 800
_C.Vesper.gru_rel_pos = True

# feature encoder
_C.Vesper.extractor_mode = 'layer_norm'   # 'default' / 'layer_norm'
_C.Vesper.conv_feature_layers = '[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2'
_C.Vesper.dropout_input = 0.0
