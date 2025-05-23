import json
import torch
import os
import pandas as pd
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import argparse
from data_loader import PowerLoader
from train import model_dict, calculate_accuracy


def model_builder(configs):
    test_data = PowerLoader(configs, 'test')
    configs.seq_len = test_data.max_seq_len
    configs.pred_len = 0
    configs.enc_in = test_data.all_data[0][0].shape[1]
    configs.num_class = test_data.num_class if not configs.is_loc else configs.seq_len+1
    model = model_dict[configs.model].Model(configs).float()
    if configs.use_multi_gpu and configs.use_gpu:
        model = nn.DataParallel(model, device_ids=configs.device_ids)
    return model


def process_power_data(configs):
    data_dir_path = configs.test_root_path
    is_loc = configs.is_loc
    list_data = []
    convert = {'a': 1, 'b': 2, 'c': 4, 'ab': 3, 'ac': 5, 'bc': 6, 'abc': 7, 'normal': 0}
    for root, dirs, files in os.walk(data_dir_path):
        for file in files:
            file_name = os.path.splitext(file)[0]
            ext = os.path.splitext(file)[-1].upper()
            file_path = os.path.join(root, file)
            if ext == '.CSV':
                df = pd.read_csv(file_path)
                if is_loc:
                    label = int(file_name.split('_')[-3])
                    list_data.append((file_name, df.to_numpy(), np.array([label])))
                else:
                    label_name = os.path.basename(os.path.dirname(file_path))
                    list_data.append((file_name, df.to_numpy(), np.array([convert[label_name]])))
    return list_data


def predict(configs):
    convert = {1: 'a相故障', 2: 'b相故障', 4: 'c相故障', 3: 'ab两相故障', 5: 'ac两相故障', 6: 'bc两相故障', 7: 'abc三相故障', 0: '正常'}
    all_outs = []
    all_gts = []
    if not os.path.exists(configs.result_save_path):
        os.makedirs(configs.result_save_path, exist_ok=True)
    json_path = os.path.join(configs.result_save_path, configs.model+'.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        results = []
    current_len = len(results)
    if current_len > 0:
        all_outs.extend([i.get('output') for i in results])
        all_gts.extend([j.get('ground_truth') for j in results])
    model_path = os.path.join(configs.checkpoints, f'{configs.model}/best_model.pth')
    model = model_builder(configs)
    model.load_state_dict(torch.load(model_path))
    model.to('cpu')
    model.eval()
    test_data = process_power_data(configs)
    with torch.no_grad():
        for index, (file_name, inp, label) in enumerate(test_data[current_len:]):
            gt = label.tolist()[0]
            all_gts.append(gt)
            mask = torch.from_numpy(np.array([1]*inp.shape[0])).float().unsqueeze(0).to('cpu')
            inp = torch.from_numpy(inp).float().unsqueeze(0).to('cpu')
            ss = time.time()
            out = torch.argmax(F.softmax(model(inp, mask, None, None), dim=-1), dim=-1).item()
            ee = time.time()
            all_outs.append(out)
            results.append(
                {'file': file_name, 'output': out, 'ground_truth': gt, 'infer_time': f'{(ee-ss):.6f}'}
            )
            if (index+1) % 50 == 0:
                with open(json_path, 'w', encoding='utf-8') as ff:
                    json.dump(results, ff, ensure_ascii=False, indent=4)
            print(f'{configs.model}:[{current_len+index+1}/{len(test_data)}], spend_time={(ee-ss):.6f}s, is_true={out==gt}, file_name={file_name}')
        with open(json_path, 'w', encoding='utf-8') as ff:
            json.dump(results, ff, ensure_ascii=False, indent=4)
    print(f'Result: ACC={calculate_accuracy(all_outs, all_gts, configs.is_loc)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PowerGCT')
    parser.add_argument('--task_name', type=str, required=False, default='classification', help='task name')
    parser.add_argument('--test_root_path', type=str, required=False, default='./datasets/test',
                        help='test_data_dirpath')
    parser.add_argument('--result_save_path', type=str, default='./results')
    parser.add_argument('--model', type=str, required=False, default='PowerGCT', help='model name')
    parser.add_argument('--num_class', type=int, default=8, help='num classes')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--is_loc', action='store_true', help='is loc or not')
    args = parser.parse_args()

    predict(args)

