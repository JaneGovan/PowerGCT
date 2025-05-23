import argparse
import random
import torch
import numpy as np
import os
from tqdm import tqdm
import time
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader import PowerLoader, collate_fn
from models import PowerGCT, PowerGCT_wo_rope, PowerGCT_wo_chunk, PowerGCT_wo_rope_and_chunk


def set_seed(seed=1234):
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU时设置所有种子
        torch.backends.cudnn.deterministic = True  # 确保CUDA卷积结果确定
        torch.backends.cudnn.benchmark = False     # 关闭优化（可能牺牲速度）

    # NumPy 和 Python 随机库
    np.random.seed(seed)
    random.seed(seed)


model_dict = {
            'PowerGCT': PowerGCT,
            'PowerGCT_wo_rope': PowerGCT_wo_rope,
            'PowerGCT_wo_chunk': PowerGCT_wo_chunk,
            'PowerGCT_wo_rope_and_chunk': PowerGCT_wo_rope_and_chunk
}


def build_model(configs):
    train_data = PowerLoader(configs)
    configs.seq_len = train_data.max_seq_len
    configs.pred_len = 0
    configs.enc_in = train_data.all_data[0][0].shape[1]
    configs.num_class = train_data.num_class if not configs.is_loc else configs.seq_len+1
    print(configs)
    model = model_dict[configs.model].Model(configs).float()
    if configs.use_multi_gpu and configs.use_gpu:
        model = nn.DataParallel(model, device_ids=configs.device_ids)
    return model


def calculate_accuracy(true_labels, predicted_labels, is_loc=False):
    if len(true_labels) != len(predicted_labels):
        raise ValueError("两个列表的长度必须相同")

    def compare(a, b):
        if is_loc:
            return abs(int(a)-int(b)) < 3
        else:
            return a == b
    # 计算正确预测的数量
    correct_predictions = sum(compare(t, p) for t, p in zip(true_labels, predicted_labels))

    # 计算准确度
    accuracy = correct_predictions / len(true_labels)

    return accuracy


def train_model(train_args):
    train_data = PowerLoader(train_args, 'train', is_loc=train_args.is_loc)
    train_loader = DataLoader(
        train_data,
        shuffle=train_args.shuffle,
        batch_size=train_args.batch_size,
        num_workers=train_args.num_workers,
        drop_last=True,
        collate_fn=lambda x: collate_fn(x, max_len=train_args.seq_len)
    )
    test_data = PowerLoader(train_args, 'test', is_loc=train_args.is_loc)
    test_loader = DataLoader(
        test_data,
        shuffle=False,
        batch_size=1,
        drop_last=False,
        collate_fn=lambda x: collate_fn(x, max_len=train_args.seq_len)
    )

    model = build_model(train_args).to(train_args.device)
    # print([i for i in model.named_parameters()])

    checkpoint = os.path.join(train_args.checkpoints, train_args.model)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    # model_optim = optim.Adam(model.parameters(), lr=train_args.learning_rate)
    model_optim = optim.RAdam(model.parameters(), lr=train_args.learning_rate)

    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    best_model_path = os.path.join(checkpoint, 'best_model.pth')
    log_file_path = os.path.join(checkpoint, f'log_{train_args.model}.txt')

    with open(log_file_path, 'w', encoding='utf-8') as log:
        log.truncate(0)
        for epoch in range(train_args.train_epochs):
            train_ground = []
            train_out = []
            train_loss = []
            model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in tqdm(enumerate(train_loader), leave=False, total=len(train_loader), desc="Training"):
                batch_x = batch_x.float().to(train_args.device)
                padding_mask = padding_mask.float().to(train_args.device)

                label = label.to(train_args.device)
                train_ground.extend(label.squeeze(-1).flatten().tolist())

                outputs = model(batch_x, padding_mask, None, None)

                train_out.extend(torch.argmax(F.softmax(outputs, dim=-1), dim=-1).flatten().tolist())

                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
                model_optim.step()
                model_optim.zero_grad()

            list_ground = []
            list_out = []
            model.eval()
            with torch.no_grad():
                for j, (batch_y, label, padding_mask_y) in tqdm(enumerate(test_loader), leave=False, total=len(test_loader), desc="Evaluation"):
                    batch_y = batch_y.float().to(train_args.device)
                    padding_mask_y = padding_mask_y.float().to(train_args.device)
                    label = label.to(train_args.device)
                    ground_true = label.squeeze(0).item()
                    output = torch.argmax(F.softmax(model(batch_y, padding_mask_y, None, None), dim=-1), dim=-1).item()
                    list_ground.append(ground_true)
                    list_out.append(output)
            if (epoch+1) % (train_args.train_epochs // train_args.limit_save) == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint, f'checkpoint_epoch_{epoch+1}.pth'))
            current_accuracy = calculate_accuracy(list_ground, list_out, train_args.is_loc)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                torch.save(model.state_dict(), best_model_path)

            log.write(f'Epoch: [{epoch + 1}/{train_args.train_epochs}], Train_Loss: {train_loss[-1]:.2f}, Train_Acc: {calculate_accuracy(train_out, train_ground, train_args.is_loc):.4f},  Val_Acc: {current_accuracy:.4f}, Epoch_Time: {time.time() - epoch_time:.2f}\n')
            print(f'{train_args.model} -> Epoch: [{epoch + 1}/{train_args.train_epochs}], Train_Loss: {train_loss[-1]:.2f}, Train_Acc: {calculate_accuracy(train_out, train_ground, train_args.is_loc):.4f}, Val_Acc: {current_accuracy:.4f}, Epoch_Time: {time.time() - epoch_time:.2f}')

        log.write(f'###Accuracy of best model {train_args.model} is {best_accuracy}###\n')
        print(f'###Accuracy of best model {train_args.model} is {best_accuracy}###')


if __name__ == '__main__':
    # 调用函数设置种子
    set_seed(1234)
    parser = argparse.ArgumentParser(description='PowerGCT')
    parser.add_argument('--task_name', type=str, required=False, default='classification', help='task name')
    parser.add_argument('--train_root_path', type=str, required=False, default='./datasets/train', help='train_data_dirpath')
    parser.add_argument('--test_root_path', type=str, required=False, default='./datasets/test', help='test_data_dirpath')
    parser.add_argument('--model', type=str, required=False, default='PowerGCT', help='model name')
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001)
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--train_epochs', type=int, required=False, default=30)
    parser.add_argument('--limit_save', type=int, default=2)
    parser.add_argument('--shuffle', type=bool, default=True, help='train data shuffle')
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
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        args.device = torch.device("cpu")
        print('Using CPU')
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    train_model(args)


