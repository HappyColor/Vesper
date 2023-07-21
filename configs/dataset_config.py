
from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

###########
# IEMOCAP #
###########
_C.iemocap = CN(new_allowed=True)
_C.iemocap.num_classes = 4
_C.iemocap.meta_csv_file = '/148Dataset/data-chen.weidong/iemocap/feature/name_label_text.csv'
_C.iemocap.wavdir = '/148Dataset/data-chen.weidong/iemocap/wav_all_sentences'
_C.iemocap.batch_length = 104000 # 16000 * 6.5
_C.iemocap.evaluate = ['accuracy', 'recall']
_C.iemocap.folds = [1, 2, 3, 4, 5]
_C.iemocap.f1 = 'weighted'
_C.iemocap.have_test_set = False

########
# MELD #
########
_C.meld = CN(new_allowed=True)
_C.meld.num_classes = 7
_C.meld.meta_csv_file = '/148Dataset/data-chen.weidong/meld/label/official'
_C.meld.wavdir = '/148Dataset/data-chen.weidong/meld/audio_16k'
_C.meld.batch_length = 72000 # 16000 * 4.5
_C.meld.evaluate = ['f1']
_C.meld.folds = [1]
_C.meld.f1 = 'weighted'
_C.meld.have_test_set = True

###########
# CREMA-D #
###########
_C.crema = CN(new_allowed=True)
_C.crema.num_classes = 6
_C.crema.meta_csv_file = '/148Dataset/data-chen.weidong/CREMA-D/CREMA-D.csv'
_C.crema.wavdir = '/148Dataset/data-chen.weidong/CREMA-D/AudioWAV'
_C.crema.batch_length = 48000 # 16000 * 3.0
_C.crema.evaluate = ['accuracy', 'recall']
_C.crema.folds = [1]
_C.crema.f1 = 'weighted'
_C.crema.have_test_set = False

#########
# LSSED #
#########
_C.lssed = CN(new_allowed=True)
_C.lssed.num_classes = 4
_C.lssed.meta_csv_file = '/148Dataset/data-chen.weidong/lssed_all/metadata_english_all.csv'
_C.lssed.wavdir = '/148Dataset/data-chen.weidong/lssed_all/wav_all'
_C.lssed.batch_length = 80000 # 16000*5
_C.lssed.evaluate = ['accuracy', 'recall']
_C.lssed.folds = [1]
_C.lssed.f1 = 'weighted'
_C.lssed.have_test_set = True

_C.lssed.target_length = 249
_C.lssed.l_target_dir = '/148Dataset/data-chen.weidong/lssed_all/feature/wavlm_large_L12_mat'
_C.lssed.h_target_dir = '/148Dataset/data-chen.weidong/lssed_all/feature/wavlm_large_L24_mat'
