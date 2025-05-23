from comtrade import Comtrade, get_file_encoding, clean_invalid_elem
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import traceback
import torch
import random
import shutil
import numpy as np
from glob import glob
from torch.utils.data import Dataset
from typing import List, Tuple
random.seed(42)


class PowerData:
    def __init__(self, cfg_file_path, dat_file_path):
        self._dir_name = os.path.dirname(cfg_file_path)
        self._file_name = os.path.splitext(os.path.basename(cfg_file_path))[0]
        self._cfg_file_path = cfg_file_path
        self._dat_file_path = dat_file_path
        if os.path.exists(os.path.join(self._dir_name, self._file_name + '.hdr')):
            self._hdr_file_path = os.path.join(self._dir_name, self._file_name + '.hdr')
        elif os.path.exists(os.path.join(self._dir_name, self._file_name + '.HDR')):
            self._hdr_file_path = os.path.join(self._dir_name, self._file_name + '.HDR')
        else:
            self._hdr_file_path = None
        if os.path.exists(os.path.join(self._dir_name, self._file_name + '.tmk')):
            self._tmk_file_path = os.path.join(self._dir_name, self._file_name + '.tmk')
        elif os.path.exists(os.path.join(self._dir_name, self._file_name + '.TMK')):
            self._tmk_file_path = os.path.join(self._dir_name, self._file_name + '.TMK')
        else:
            self._tmk_file_path = None
        self._save_dir_path = 'extracted_data_csv'
        self._rec = Comtrade().load(self._cfg_file_path, self._dat_file_path)
        self._time_samples = self.get_time_samples()
        self._cfg = self._rec.cfg
        self._cfg_analog_channels = self._rec.cfg.analog_channels

    def get_hdr_file_path(self):
        return self._hdr_file_path

    def get_tmk_file_path(self):
        return self._tmk_file_path

    def get_rec(self):
        return self._rec

    def get_cfg(self):
        return self._cfg

    def get_time_samples(self):
        return len(self._rec.analog[0])

    @staticmethod
    def match_line_by_name(substring: str, parentstring: str):
        substring = re.sub('[线回]', '', substring)
        # Ⅰ Ⅱ Ⅲ Ⅳ Ⅴ Ⅵ Ⅶ Ⅷ Ⅸ Ⅹ Ⅺ Ⅻ Ⅼ Ⅽ Ⅾ Ⅿ
        convert = {'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV', 'Ⅴ': 'V', 'Ⅵ': 'VI', 'Ⅶ': 'VII', 'Ⅷ': 'VIII', 'Ⅸ': 'IX',
                   'Ⅹ': 'X', 'Ⅺ': 'XI', 'Ⅻ': 'XII', 'Ⅼ': 'L', 'Ⅽ': 'C', 'Ⅾ': 'D', 'Ⅿ': 'M'}
        alist = re.findall(r'[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅬⅭⅮⅯ]', substring)
        for s in set(alist):
            substring = re.sub(f'{s}', f'{convert[s]}', substring)
        blist = re.findall(r'[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅬⅭⅮⅯ]', parentstring)
        for s in set(blist):
            parentstring = re.sub(f'{s}', f'{convert[s]}', parentstring)
        roman = re.search('((VIII)|(XII)|(VII)|(III)|(VI)|(IV)|(IX)|(XI)|(II)|I|V|X|L|C|D|M)', substring)
        str_zh = re.sub('(VIII)|(XII)|(VII)|(III)|(VI)|(IV)|(IX)|(XI)|(II)|I|V|X|L|C|D|M', '', substring)
        flag = True
        for s in str_zh:
            if s not in parentstring:
                flag = False
        if not roman:
            return flag
        else:
            if roman.group(1) in parentstring and flag:
                return True
            else:
                return False

    def get_info_fault_point_by_tmk(self):
        """
        返回类型 ：(故障电压A相所在通道, 故障电流A相所在通道, 故障类型, 故障开始时刻的采样点, 故障结束时刻的采样点)
        """
        binary2fault = {1: 'A相', 2: 'B相', 3: 'AB相', 4: 'C相', 5: 'AC相', 6: 'BC相', 7: 'ABC相'}
        tmk_file_path = self._tmk_file_path
        with open(tmk_file_path, 'r', encoding=get_file_encoding(tmk_file_path)) as f:
            ctx = f.read()
            fault_xb = int(re.search(r'(FAULT_XB) *= *(\d+)', ctx, flags=re.IGNORECASE).group(2))
            fault_start = int(re.search(r'(FAULT_POS) *= *(\d+)', ctx, flags=re.IGNORECASE).group(2))
            fault_end = int(re.search(r'(FAULT_TZ_POS) *= *(\d+)', ctx, flags=re.IGNORECASE).group(2))
            fault_u_channel = int(re.search(r'(BUS_ACHAN) *= *(\d+)', ctx, flags=re.IGNORECASE).group(2))
            fault_i_channel = int(re.search(r'(CURR_ACHAN) *= *(\d+)', ctx, flags=re.IGNORECASE).group(2))

        return fault_u_channel - 1, fault_i_channel - 1, binary2fault.get(fault_xb, None), fault_start, fault_end

    def get_info_fault_point_by_hdr(self):
        """
        返回类型 ：[(故障电压A相所在通道, 故障电流A相所在通道, 故障类型, 故障开始时刻的采样点, 故障结束时刻的采样点), ...]
        """
        hdr_file_path = self._hdr_file_path
        valid_u_channel = self.get_valid_voltage_channel_by_self()
        valid_i_channel = self.get_valid_current_channel_by_hdr()
        info_points = []
        analog_channel_ids = self._rec.analog_channel_ids
        with open(hdr_file_path, 'r', encoding=get_file_encoding(hdr_file_path)) as f:
            ctx = f.readlines()
            fault_line = [re.sub(r'[0-9a-zA-HJ-Z]', '', clean_invalid_elem(c.split(' '))[-1]) for c in ctx if
                          re.search(r'故障间隔名', c)]
            fault_line_name = fault_line[0]
            fault_type = [clean_invalid_elem(c.split(' '))[-1] for c in ctx if re.search(r'故障相别', c)]
            fault_starttime = [''.join(clean_invalid_elem(c.split(' '))[1:]) for c in ctx if re.search(r'故障时刻', c)]
            fault_endtime = [''.join(clean_invalid_elem(c.split(' '))[1:]) for c in ctx if re.search(r'跳闸时刻', c)]
        if len(fault_type) != len(fault_starttime) or len(fault_type) != len(fault_endtime) or len(
                fault_endtime) != len(fault_starttime):
            raise Exception(f'{hdr_file_path}: number of fault error')
        for x, y, z in zip(fault_type, fault_starttime, fault_endtime):
            triple = []
            triple.append(valid_u_channel[0]) if valid_u_channel else triple.append(None)
            triple.append(valid_i_channel[0]) if valid_i_channel else triple.append(None)
            if 'A' in x.upper() and 'B' in x.upper() and 'C' in x.upper():
                if '接地' in x:
                    triple.append('ABC相接地故障')
                else:
                    triple.append('ABC相故障')
            elif 'A' in x.upper() and 'B' in x.upper() and 'C' not in x.upper():
                if '接地' in x:
                    triple.append('AB相接地故障')
                else:
                    triple.append('AB相故障')
            elif 'A' in x.upper() and 'B' not in x.upper() and 'C' in x.upper():
                if '接地' in x:
                    triple.append('AC相接地故障')
                else:
                    triple.append('AC相故障')
            elif 'A' not in x.upper() and 'B' in x.upper() and 'C' in x.upper():
                if '接地' in x:
                    triple.append('BC相接地故障')
                else:
                    triple.append('BC相故障')
            elif 'A' in x.upper():
                if '接地' in x:
                    triple.append('A相接地故障')
                else:
                    triple.append('A相故障')
            elif 'B' in x.upper():
                if '接地' in x:
                    triple.append('B相接地故障')
                else:
                    triple.append('B相故障')
            elif 'C' in x.upper():
                if '接地' in x:
                    triple.append('C相接地故障')
                else:
                    triple.append('C相故障')
            else:
                raise Exception(f'{hdr_file_path}: abnormal type error')
            start_time_match = re.search(r'(-?\d+(\.\d+)?)(s|ms)', y, flags=re.IGNORECASE)
            if start_time_match.group(3).lower() == 'ms':
                start_fault_time = float(start_time_match.group(1)) / 1000
            else:
                start_fault_time = float(start_time_match.group(1))
            end_time_match = re.search(r'(-?\d+(\.\d+)?)(s|ms)', z, flags=re.IGNORECASE)
            if end_time_match.group(3).lower() == 'ms':
                end_fault_time = float(end_time_match.group(1)) / 1000
            else:
                end_fault_time = float(end_time_match.group(1))
            print(start_fault_time, end_fault_time)
            for idx, t in enumerate(self.dat_reader().iloc[:, -1]):
                if float(t) >= start_fault_time:
                    triple.append(idx)
                    break
            else:
                raise Exception(f'{hdr_file_path}: abnormal start point error')
            for idx, t in enumerate(self.dat_reader().iloc[:, -1]):
                if float(t) >= end_fault_time:
                    triple.append(idx)
                    break
            else:
                raise Exception(f'{hdr_file_path}: abnormal end point error')
            info_points.append(tuple(triple))
        return info_points

    def get_valid_voltage_channel_by_self(self):
        df = self.dat_reader()
        all_u_channels = self.get_voltage_channels()
        for group_u in all_u_channels:
            v_channel_ph_a = group_u[0]
            if (min(df.iloc[:, v_channel_ph_a]) <= -35 and max(df.iloc[:, v_channel_ph_a]) >= 35) and (
                    min(df.iloc[:, v_channel_ph_a + 1]) <= -35 and max(df.iloc[:, v_channel_ph_a + 1]) >= 35) and (
                    min(df.iloc[:, v_channel_ph_a + 2]) <= -35 and max(df.iloc[:, v_channel_ph_a + 2]) >= 35):
                return v_channel_ph_a, v_channel_ph_a + 1, v_channel_ph_a + 2, v_channel_ph_a + 3
        else:
            return None

    def get_valid_current_channel_by_hdr(self):
        hdr_file_path = self._hdr_file_path
        analog_channel_ids = self._rec.analog_channel_ids
        with open(hdr_file_path, 'r', encoding=get_file_encoding(hdr_file_path)) as f:
            ctx = f.readlines()
            fault_line = [re.sub(r'[0-9a-zA-HJ-Z]', '', clean_invalid_elem(c.split(' '))[-1]) for c in ctx if
                          re.search(r'故障间隔名', c)]
            fault_line_name = fault_line[0]
        for ix, name in enumerate(analog_channel_ids):
            if PowerData.match_line_by_name(fault_line_name, name) and self._cfg.analog_channels[ix].uu == 'A':
                return ix, ix + 1, ix + 2, ix + 3
        else:
            return None

    def get_valid_current_channel(self):
        i_channel_ph_a = self.get_info_fault_point_by_tmk()[1]
        return i_channel_ph_a, i_channel_ph_a + 1, i_channel_ph_a + 2, i_channel_ph_a + 3

    def get_valid_voltage_channel(self):
        df = self.dat_reader()
        v_channel_ph_a = self.get_info_fault_point_by_tmk()[0]
        if (min(df.iloc[:, v_channel_ph_a]) <= -35 and max(df.iloc[:, v_channel_ph_a]) >= 35) and (
                min(df.iloc[:, v_channel_ph_a + 1]) <= -35 and max(df.iloc[:, v_channel_ph_a + 1]) >= 35) and (
                min(df.iloc[:, v_channel_ph_a + 2]) <= -35 and max(df.iloc[:, v_channel_ph_a + 2]) >= 35):
            return v_channel_ph_a, v_channel_ph_a + 1, v_channel_ph_a + 2, v_channel_ph_a + 3
        else:
            voltage_channels = [v_channel_ph_a]
            for idx, name_line in enumerate(self._rec.analog_channel_ids):
                if self._cfg.analog_channels[idx].uu == 'V' and self._cfg.analog_channels[idx].ph == 'B':
                    if min(df.iloc[:, idx]) <= -35 and max(df.iloc[:, idx]) >= 35:
                        voltage_channels.append(idx)
                        break

            for idx, name_line in enumerate(self._rec.analog_channel_ids):
                if self._cfg.analog_channels[idx].uu == 'V' and self._cfg.analog_channels[idx].ph == 'C':
                    if min(df.iloc[:, idx]) <= -35 and max(df.iloc[:, idx]) >= 35:
                        voltage_channels.extend([idx, idx + 1])
                        break

            assert len(voltage_channels) == 4, 'length of voltage_channels is error'
            return voltage_channels[0], voltage_channels[1], voltage_channels[2], voltage_channels[3]

    def get_voltage_channels(self):
        voltage_channels = []
        for idx, name_line in enumerate(self._rec.analog_channel_ids):
            if self._cfg.analog_channels[idx].uu == 'V' and self._cfg.analog_channels[idx].ph == 'A':
                voltage_channels.append((idx, idx + 1, idx + 2, idx + 3))
        return voltage_channels

    def get_current_channels(self):
        current_channels = []
        for idx, name_line in enumerate(self._rec.analog_channel_ids):
            if self._cfg.analog_channels[idx].uu == 'A' and self._cfg.analog_channels[idx].ph == 'A':
                current_channels.append((idx, idx + 1, idx + 2, idx + 3))
        return current_channels

    def dat_reader(self, is_save=False):
        analog_data_list = self._rec.analog
        analog_channel_ids = self._rec.analog_channel_ids
        # print(rec.cfg_summary())
        all_moment = list()
        before_num_sample = 0
        before_moment = 0.0
        for i in range(self._rec.cfg.nrates):
            rate, samples = self._rec.cfg.sample_rates[i]
            if i != 0 and samples <= self._rec.cfg.sample_rates[i - 1][-1]:
                all_moment.clear()
                before_num_sample = 0
                before_moment = 0.0
                break
            for j in range(before_num_sample, samples):
                if j == 0:
                    all_moment.append(0.0)
                    before_moment = 0.0
                else:
                    all_moment.append(before_moment + 1 / rate)
                    before_moment = before_moment + 1 / rate
            before_num_sample = len(all_moment)
        if before_num_sample == 0:
            rate = self._rec.cfg.sample_rates[-1][0]
            for k in range(self._rec.cfg.sample_rates[-1][-1]):
                if k == 0:
                    all_moment.append(0.0)
                    before_moment = 0.0
                else:
                    all_moment.append(before_moment + 1 / rate)
                    before_moment = before_moment + 1 / rate
        all_data = dict()
        for idx, channel_id in enumerate(analog_channel_ids):
            all_data[channel_id + f'_{idx + 1}'] = analog_data_list[idx].tolist()
        all_data['相对时间'] = all_moment
        df = pd.DataFrame(all_data)

        save_file_path = os.path.join(self._save_dir_path,
                                      os.path.splitext(os.path.basename(self._dat_file_path))[0] + '.csv')
        if is_save and not os.path.exists(save_file_path):
            if not os.path.exists(self._save_dir_path):
                os.mkdir(self._save_dir_path)
            df.to_csv(save_file_path, index=False, encoding='utf-8-sig')
        return df

    def extreme_values(self, feature_channels: List[Tuple[int, int, int, int]]) -> Tuple[float, float]:
        df = self.dat_reader()
        value_min = min([min(df.iloc[:, i]) for i_feature_channel in feature_channels for i in i_feature_channel])
        value_max = max([max(df.iloc[:, j]) for j_feature_channel in feature_channels for j in j_feature_channel])
        return value_min, value_max

    def draw_analog_channel(self, u_feature_channel, i_feature_channel):
        # u_feature_channel = self.get_valid_voltage_channel()[0]
        df = self.dat_reader()
        t_min = 0.0
        t_max = df.iloc[-1, -1]
        u_min, u_max = self.extreme_values(self.get_voltage_channels())
        i_min, i_max = self.extreme_values(self.get_current_channels())
        u_surround = max([abs(u_max), abs(u_min)])
        i_surround = max([abs(i_max), abs(i_min)])
        fig, axs = plt.subplots(len(u_feature_channel + i_feature_channel), 1, figsize=(13.5, 8.5))
        for idx, col in enumerate(u_feature_channel + i_feature_channel):
            axs[idx].plot(df.iloc[:, -1], df.iloc[:, col], label=df.columns[idx], linestyle='-', linewidth=0.5)
            axs[idx].set_ylabel(df.columns[col], rotation=0, labelpad=50, fontdict={'family': 'SimHei', 'size': 10})
            axs[idx].set_xlim(t_min, t_max)
            if col in u_feature_channel:
                axs[idx].set_ylim(-u_surround, u_surround)
            else:
                axs[idx].set_ylim(-i_surround, i_surround)
        plt.tight_layout()
        plt.show()


def get_total_data(directory):
    total_data = dict()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.dat') or file.endswith('.DAT'):
                file_name = os.path.splitext(file)[0]
                if total_data.get(file_name) is not None:
                    continue
                if os.path.exists(os.path.join(root, file_name + '.cfg')):
                    cfg_file_path = os.path.join(root, file_name + '.cfg')
                elif os.path.exists(os.path.join(root, file_name + '.CFG')):
                    cfg_file_path = os.path.join(root, file_name + '.CFG')
                else:
                    continue
                total_data[file_name] = PowerData(cfg_file_path, os.path.join(root, file)).dat_reader()
    return total_data


def check_total_data(directory):
    num = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_name, ext = os.path.splitext(file)
            file_path = os.path.join(root, file)
            if ext.upper() == '.TMK':
                cfg_file_path = os.path.join(root, file_name + '.cfg') if os.path.exists(
                    os.path.join(root, file_name + '.cfg')) else os.path.join(root, file_name + '.CFG')
                dat_file_path = os.path.join(root, file_name + '.dat') if os.path.exists(
                    os.path.join(root, file_name + '.dat')) else os.path.join(root, file_name + '.DAT')
                try:
                    power = PowerData(cfg_file_path, dat_file_path)
                    tmk_file_path = power.get_tmk_file_path()
                    print(tmk_file_path)
                    u_channel = power.get_valid_voltage_channel()
                    i_channel = power.get_valid_current_channel()
                    power.draw_analog_channel(u_channel, i_channel)
                    # for i in range(power.get_cfg().nrates):
                    #     rate, samples = power.get_cfg().sample_rates[i]
                    #     if i != 0 and samples <= power.get_cfg().sample_rates[i - 1][-1]:
                    #         # continue
                    #         print(file_path)
                    #         print(power.get_voltage_channels())
                    #         print(power.get_valid_voltage_channel())
                    #         print(power.get_info_fault_point_by_tmk())
                    print('=' * 50)
                    num += 1
                except Exception:
                    continue
                # power.dat_reader(is_save=True)

            elif ext.upper() == '.HDR':
                cfg_file_path = os.path.join(root, file_name + '.cfg') if os.path.exists(
                    os.path.join(root, file_name + '.cfg')) else os.path.join(root, file_name + '.CFG')
                dat_file_path = os.path.join(root, file_name + '.dat') if os.path.exists(
                    os.path.join(root, file_name + '.dat')) else os.path.join(root, file_name + '.DAT')
                try:
                    power = PowerData(cfg_file_path, dat_file_path)
                    hdr_file_path = power.get_hdr_file_path()
                    print(hdr_file_path)
                    with open(hdr_file_path, 'r', encoding=get_file_encoding(hdr_file_path)) as f:
                        li = f.readlines()
                    for i in li:
                        if '故障相别' in i:
                            print(i)
                    u_channel = power.get_valid_voltage_channel_by_self()
                    i_channel = power.get_valid_current_channel_by_hdr()
                    power.draw_analog_channel(u_channel, i_channel)
                    # for i in range(power.get_cfg().nrates):
                    #     rate, samples = power.get_cfg().sample_rates[i]
                    #     if i != 0 and samples <= power.get_cfg().sample_rates[i - 1][-1]:
                    #         print(file_path)
                    #         print(power.get_info_fault_point_by_hdr())
                    #         uu = power.get_valid_voltage_channel_by_self()
                    #         ii = power.get_valid_current_channel_by_hdr()
                    #         power.draw_analog_channel(uu, ii)
                    print('=' * 50)
                    num += 1
                except Exception:
                    continue
                # power.dat_reader(is_save=True)
    return num


def generate_dataset_by_tmk(directory, save_dir='./all_dataset'):
    for label in ['a', 'b', 'c', 'ab', 'ac', 'bc', 'abc', 'normal']:
        sub_save = os.path.join(save_dir, f'class_8/{label}')
        if not os.path.exists(sub_save):
            os.makedirs(sub_save, exist_ok=True)
    normal, p_a, p_b, p_c, p_ab, p_ac, p_bc, p_abc = 0, 0, 0, 0, 0, 0, 0, 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_name = os.path.splitext(file)[0]
            cfg_file_path = os.path.join(root, file_name + '.cfg') if os.path.exists(
                os.path.join(root, file_name + '.cfg')) else os.path.join(root, file_name + '.CFG')
            dat_file_path = os.path.join(root, file_name + '.dat') if os.path.exists(
                os.path.join(root, file_name + '.dat')) else os.path.join(root, file_name + '.DAT')
            ext = os.path.splitext(file)[-1].upper()
            # file_path = os.path.join(root, file)
            if ext == '.TMK':
                try:
                    electric_data = PowerData(cfg_file_path, dat_file_path)
                except Exception:
                    continue
                df = electric_data.dat_reader()
                info_fault = list(electric_data.get_info_fault_point_by_tmk())
                for i in range(electric_data.get_cfg().nrates):
                    rate, samples = electric_data.get_cfg().sample_rates[i]
                    if i != 0 and samples <= electric_data.get_cfg().sample_rates[i - 1][-1]:
                        continue
                if info_fault[3] <= 0 or info_fault[2] is None or electric_data.get_time_samples() < info_fault[4]:
                    continue
                channels_8 = list(electric_data.get_valid_voltage_channel() + electric_data.get_valid_current_channel())
                if np.isnan(df.iloc[:, channels_8].to_numpy()).any() or np.isinf(
                        df.iloc[:, channels_8].to_numpy()).any():
                    continue

                # # check and fix fault point
                # if info_fault[3] < 50:
                #     continue
                # print(info_fault)
                # check_channels(df.iloc[info_fault[3]-50: info_fault[4]+50, channels_8])
                # state = input("跳过(c) or 纠正(v) or 不改变(other): ")
                # if state == 'c':
                #     continue
                # elif state == 'v':
                #     info_fault[3] = int(input("开始点纠正为: "))
                #     info_fault[4] = int(input("结束点纠正为: "))
                # else:
                #     print("不改变")

                if info_fault[3] > 200:
                    normal_sample = list(range(info_fault[3] - 200, info_fault[3] - 150))
                    for i in random.sample(normal_sample, len(normal_sample) // 25):
                        df.iloc[i: i + 100, channels_8].to_csv(
                            os.path.join(save_dir, f'class_8/normal/{file_name}_{0}_normal_{normal}.csv'), index=False,
                            encoding='utf-8-sig')
                        normal += 1

                # if info_fault[4] - info_fault[3] < 95:
                #     end_sample = info_fault[4]
                # else:
                #     end_sample = info_fault[3] + 95
                end_sample = info_fault[3] + 100

                if info_fault[3] < 100:
                    start_sample = 0
                else:
                    start_sample = info_fault[3] - 100
                if end_sample - start_sample < 100:
                    continue

                range_sample = list(range(start_sample, end_sample - 100))
                if info_fault[2] == 'A相':
                    candidates = random.sample(range_sample, len(range_sample) // 26)
                    for i in candidates:
                        df.iloc[i:i + 100, channels_8].to_csv(
                            os.path.join(save_dir,
                                         f'class_8/a/{file_name}_{info_fault[3] - i}_a_{p_a}.csv'),
                            index=False,
                            encoding='utf-8-sig')
                        p_a += 1

                elif info_fault[2] == 'B相':
                    candidates = random.sample(range_sample, len(range_sample) // 20)
                    for i in candidates:
                        df.iloc[i:i + 100, channels_8].to_csv(
                            os.path.join(save_dir,
                                         f'class_8/b/{file_name}_{info_fault[3] - i}_b_{p_b}.csv'),
                            index=False,
                            encoding='utf-8-sig')
                        p_b += 1

                elif info_fault[2] == 'C相':
                    candidates = random.sample(range_sample, len(range_sample) // 24)
                    for i in candidates:
                        df.iloc[i:i + 100, channels_8].to_csv(
                            os.path.join(save_dir,
                                         f'class_8/c/{file_name}_{info_fault[3] - i}_c_{p_c}.csv'),
                            index=False,
                            encoding='utf-8-sig')
                        p_c += 1

                elif info_fault[2] == 'AB相':
                    candidates = random.sample(range_sample, len(range_sample) // 6)
                    for i in candidates:
                        df.iloc[i:i + 100, channels_8].to_csv(
                            os.path.join(save_dir,
                                         f'class_8/ab/{file_name}_{info_fault[3] - i}_ab_{p_ab}.csv'),
                            index=False,
                            encoding='utf-8-sig')
                        p_ab += 1

                elif info_fault[2] == 'AC相':
                    candidates = random.sample(range_sample, len(range_sample) // 4)
                    for i in candidates:
                        df.iloc[i:i + 100, channels_8].to_csv(
                            os.path.join(save_dir,
                                         f'class_8/ac/{file_name}_{info_fault[3] - i}_ac_{p_ac}.csv'),
                            index=False,
                            encoding='utf-8-sig')
                        p_ac += 1

                elif info_fault[2] == 'BC相':
                    candidates = random.sample(range_sample, len(range_sample) // 6)
                    for i in candidates:
                        df.iloc[i:i + 100, channels_8].to_csv(
                            os.path.join(save_dir,
                                         f'class_8/bc/{file_name}_{info_fault[3] - i}_bc_{p_bc}.csv'),
                            index=False,
                            encoding='utf-8-sig')
                        p_bc += 1

                elif info_fault[2] == 'ABC相':
                    candidates = random.sample(range_sample, len(range_sample) // 2)
                    for i in candidates:
                        df.iloc[i:i + 100, channels_8].to_csv(
                            os.path.join(save_dir,
                                         f'class_8/abc/{file_name}_{info_fault[3] - i}_abc_{p_abc}.csv'),
                            index=False,
                            encoding='utf-8-sig')
                        p_abc += 1

                print(file_name)
                # print(info_fault)
                print('=' * 50)
    return normal, p_a, p_b, p_c, p_ab, p_ac, p_bc, p_abc


def check_channels(df):
    u_min = min([min(df.iloc[:, i]) for i in range(4)])
    u_max = max([max(df.iloc[:, j]) for j in range(4)])
    i_min = min([min(df.iloc[:, i]) for i in range(4, 8)])
    i_max = max([max(df.iloc[:, j]) for j in range(4, 8)])
    u_surround = max([abs(u_max), abs(u_min)])
    i_surround = max([abs(i_max), abs(i_min)])
    fig, axs = plt.subplots(8, 1, figsize=(13.5, 8.5))
    for idx, col in enumerate(df.columns):
        axs[idx].plot(list(range(len(df.iloc[:, 0]))), df.iloc[:, idx], label=col, linestyle='-', linewidth=0.5)
        axs[idx].set_ylabel(col, rotation=0, labelpad=50, fontdict={'family': 'SimHei', 'size': 10})
        # axs[idx].set_xlim(t_min, t_max)
        if idx < 4:
            axs[idx].set_ylim(-u_surround, u_surround)
        else:
            axs[idx].set_ylim(-i_surround, i_surround)
        axs[idx].set_xticks(np.arange(0, len(df.iloc[:, 0]), 2))
        axs[idx].grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def fix_dataset(dataset_dir_path, out_dir='./dataset/all'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    list_csv_file = glob(f'{dataset_dir_path}/**/*.csv', recursive=True)
    list_out_filename = [os.path.splitext(os.path.basename(i))[0] for i in glob(f'{out_dir}/**/*.csv', recursive=True)]

    for csv in list_csv_file:
        print("#" * 50)
        filename = os.path.splitext(os.path.basename(csv))[0]
        if filename in list_out_filename:
            continue
        split_filename = filename.split('_')
        fault_type = split_filename[-2]
        fault_point = split_filename[-3]
        new_dir = os.path.join(out_dir, fault_type)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir, exist_ok=True)
        print(f'故障文件：{filename}\n故障类型：{fault_type}\n故障位置：{fault_point}')
        df = pd.read_csv(csv)
        check_channels(df)
        state = input("删除(D/d) or 修改(C/c) or 跳过(Enter/other): ")
        if state == 'D' or state == 'd':
            try:
                os.remove(csv)
                print(f"已删除故障文件：{filename}")
            except Exception as e:
                traceback.print_exc()
                print('删除失败')
        elif state == 'C' or state == 'c':
            new_point = int(input("新的故障位置："))
            split_filename[-3] = str(new_point)
            new_filename = '_'.join(split_filename)
            new_csv = os.path.join(os.path.dirname(csv), new_filename + '.csv')
            try:
                os.rename(csv, new_csv)
                copy_dir = os.path.join(new_dir, new_filename + '.csv')
                shutil.copy(new_csv, copy_dir)
                # os.rename(csv, new_csv)
                print(
                    f"{os.path.basename(csv)}\nv\n{os.path.basename(new_csv)}\n故障位置从{fault_point}改为{new_point}")
            except Exception as e:
                traceback.print_exc()
                print('修改失败')
        else:
            try:
                copy_dir = os.path.join(new_dir, filename + '.csv')
                shutil.copy(csv, copy_dir)
                print("跳到下一个")
            except Exception as e:
                traceback.print_exc()
                print("跳过失败")
        print(f"故障{fault_type}已修改{len(glob(f'{new_dir}/**/*.csv', recursive=True))}个样本")
        print(f"还有{540 - len(glob(f'{new_dir}/**/*.csv', recursive=True))}个样本要check")


def draw_dataset(csv_file_path, save_dir='./view'):
    filename = os.path.splitext(os.path.basename(csv_file_path))[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_file_path)
    u_min = min([min(df.iloc[:, i]) for i in range(4)])
    u_max = max([max(df.iloc[:, j]) for j in range(4)])
    i_min = min([min(df.iloc[:, i]) for i in range(4, 8)])
    i_max = max([max(df.iloc[:, j]) for j in range(4, 8)])
    u_surround = max([abs(u_max), abs(u_min)])
    i_surround = max([abs(i_max), abs(i_min)])
    fig, axs = plt.subplots(8, 1, figsize=(13.5, 8.5))
    for idx, col in enumerate(df.columns):
        axs[idx].plot(list(range(len(df.iloc[:, 0]))), df.iloc[:, idx], label=col, linestyle='-', linewidth=0.5)
        axs[idx].set_ylabel(col, rotation=0, labelpad=50, fontdict={'family': 'SimHei', 'size': 10})
        # axs[idx].set_xlim(t_min, t_max)
        if idx < 4:
            axs[idx].set_ylim(-u_surround, u_surround)
        else:
            axs[idx].set_ylim(-i_surround, i_surround)
        axs[idx].set_xticks(np.arange(0, len(df.iloc[:, 0]), 5))
        axs[idx].grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, filename + '.png'))
    plt.close(fig)


def view_all_data(dir_path, save_dir='./view'):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_name, ext = os.path.splitext(file)
            file_path = os.path.join(root, file)
            draw_dataset(file_path, save_dir)


def split_dataset(data_directory, split_rate, save_dir_path='./dataset'):
    assert len(split_rate) == 3 and sum(split_rate) == 1, "Value of split_rate is unreasonable!"
    len_classes = []
    for sub in os.listdir(data_directory):
        len_classes.append(len(os.listdir(os.path.join(data_directory, sub))))
    len_one_class = min(len_classes)
    num_val = int(len_one_class * split_rate[1])
    num_test = int(len_one_class * split_rate[2])
    num_train = len_one_class - num_test - num_val
    for sub_dir in os.listdir(data_directory):
        class_dir_path = os.path.join(data_directory, sub_dir)
        list_file_train = os.listdir(class_dir_path)[:num_train]
        list_file_val = os.listdir(class_dir_path)[num_train:num_train + num_val]
        list_file_test = os.listdir(class_dir_path)[-num_test:]
        for i in list_file_train:
            source_file_path = os.path.join(class_dir_path, i)
            des_dir_path = os.path.join(save_dir_path, f'train/{sub_dir}')
            if not os.path.exists(des_dir_path):
                os.makedirs(des_dir_path, exist_ok=True)
            des_file_path = os.path.join(des_dir_path, i)
            shutil.copy(source_file_path, des_file_path)
        for k in list_file_val:
            source_file_path = os.path.join(class_dir_path, k)
            des_dir_path = os.path.join(save_dir_path, f'val/{sub_dir}')
            if not os.path.exists(des_dir_path):
                os.makedirs(des_dir_path, exist_ok=True)
            des_file_path = os.path.join(des_dir_path, k)
            shutil.copy(source_file_path, des_file_path)
        for j in list_file_test:
            source_file_path = os.path.join(class_dir_path, j)
            des_dir_path = os.path.join(save_dir_path, f'test/{sub_dir}')
            if not os.path.exists(des_dir_path):
                os.makedirs(des_dir_path, exist_ok=True)
            des_file_path = os.path.join(des_dir_path, j)
            shutil.copy(source_file_path, des_file_path)
    return num_train, num_val, num_test


class PowerLoader(Dataset):
    def __init__(self, args, flag='train', is_loc=False):
        self.data_dir_path = args.train_root_path if flag.lower() == 'train' else args.test_root_path
        self.all_data = self.process_all_data()
        self.max_seq_len = max([i[0].shape[0] for i in self.all_data])
        self.num_class = self.max_seq_len+1 if is_loc else args.num_class
        self.is_loc = is_loc

    def process_all_data(self):
        list_data = []
        convert = {'a': 1, 'b': 2, 'c': 4, 'ab': 3, 'ac': 5, 'bc': 6, 'abc': 7, 'normal': 0}
        temp_seq_len = 0
        for root, dirs, files in os.walk(self.data_dir_path):
            for file in files:
                file_name = os.path.splitext(file)[0]
                ext = os.path.splitext(file)[-1].upper()
                file_path = os.path.join(root, file)
                if ext == '.CSV':
                    fault_id = int(file_name.split('_')[-3])
                    df = pd.read_csv(file_path)
                    if len(df.iloc[:, 0]) > temp_seq_len:
                        temp_seq_len = len(df.iloc[:, 0])
                    label_name = os.path.basename(os.path.dirname(file_path))
                    list_data.append((df.to_numpy(), np.array([convert[label_name]]), np.array([fault_id])))
        return list_data

    @staticmethod
    def instance_norm(case):
        mean = case.mean(0, keepdim=True)
        case = case - mean
        stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
        case /= stdev
        return case

    def __getitem__(self, idx):
        if self.is_loc:
            return torch.from_numpy(self.all_data[idx][0]), torch.from_numpy(self.all_data[idx][2])
        else:
            return torch.from_numpy(self.all_data[idx][0]), torch.from_numpy(self.all_data[idx][1])

    def __len__(self):
        return len(self.all_data)


def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def collate_fn(data, max_len=None):
    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks


def del_identification(data_dir_path):
    new_headers = ['Ua', 'Ub', 'Uc', '3U0', 'Ia', 'Ib', 'Ic', '3I0']
    for root, dirs, files in os.walk(data_dir_path):
        for file in files:
            ext = os.path.splitext(file)[-1].upper()
            if ext == '.CSV':
                file_name = os.path.splitext(file)[0]
                # print(file_name)
                info_list = file_name.split('_')
                new_file_name = '_'.join(['power', info_list[-3], info_list[-2], info_list[-1]])
                # print(new_file_name)
                file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, new_file_name + '.csv')
                # print(new_file_path)
                df = pd.read_csv(file_path)
                if len(df.columns) != 8:
                    raise ValueError(f"文件列数不是8列，实际有{len(df.columns)}列")
                # print(df.columns)
                df.columns = new_headers
                # print(df.columns)
                try:
                    df.to_csv(new_file_path, index=False)
                    os.remove(file_path)
                except Exception as e:
                    print(f"处理文件时出错: {e}")
                    return None
                print(f'{file_name} -> {new_file_name}')
