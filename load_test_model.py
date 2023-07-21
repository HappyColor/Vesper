
import os
import torch
from models.vesper import Vesper, init_with_ckpt
from configs import model_cfg

def load_model(ckpt_path, device='cuda'):
    model_cfg['Vesper']['freeze_cnn'] = True
    model_cfg['Vesper']['device'] = device
    model = Vesper(model_cfg['Vesper']).to(device)
    init_with_ckpt(model, ckpt_path, 'vesper', device=device)

    return model

def extract_hiddens(model, waveform, padding_mask=None):
    with torch.no_grad():
        fea, layer_results = model(
            waveform=waveform, padding_mask=padding_mask, ret_layer_results=True, student_pretraining=False)
    layer_results = [layer_results[i+1][0] for i in range(len(layer_results)-1)]
    return layer_results

if __name__ == '__main__':
    device = 'cuda'  # 'cuda' or 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ckpt_path = 'exp/Vesper/lssed_e100_b32_lr0.0005_None_mark_L12_wloss/fold_1/checkpoint/model_best.pt'

    model = load_model(ckpt_path, device)
    waveform = torch.randn(1, 16000).to(device)
    padding_mask = torch.zeros(1, 16000).eq(1).to(device)

    hiddens = extract_hiddens(model, waveform, padding_mask)
    
    print(hiddens[0].shape) # 每一层的输出特征的形状（B, T, C）
    print(len(hiddens))     # 12, 总共有12层
