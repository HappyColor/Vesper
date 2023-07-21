
from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

###############
# Transformer #
###############
_C.Transformer = CN(new_allowed=True)
_C.Transformer.EPOCH = 120
_C.Transformer.batch_size = 32
_C.Transformer.lr = 0.0005

##########
# Vesper #
##########
_C.Vesper_pretrain = CN(new_allowed=True)
_C.Vesper_pretrain.EPOCH = 100
_C.Vesper_pretrain.batch_size = 256
_C.Vesper_pretrain.lr = 0.005
_C.Vesper_pretrain.optimizer = 'AdamW' # 'AdamW' / 'sgd'
_C.Vesper_pretrain.weight_decay = 0.01
_C.Vesper_pretrain.freeze_cnn = True
_C.Vesper_pretrain.loss_weight_l = 1.0
_C.Vesper_pretrain.loss_weight_h = 0.1
_C.Vesper_pretrain.loss_weight_x = 1.0

_C.Vesper_finetune = CN(new_allowed=True)
_C.Vesper_finetune.EPOCH = 50
_C.Vesper_finetune.batch_size = 16
_C.Vesper_finetune.lr = 0.0001
_C.Vesper_finetune.optimizer = 'sgd' # 'AdamW' / 'sgd'
_C.Vesper_finetune.weight_decay = 0.01
_C.Vesper_finetune.freeze_cnn = True
_C.Vesper_finetune.freeze_upstream = True
