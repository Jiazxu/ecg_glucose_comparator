# -*- coding: utf-8 -*-
# Tools for data processing
# raw files[.csv] --> processed files[.csv] --> DataLoader[MyDataset]
#                                        or --> dataset[.pt]

import pandas as pd
import numpy as np
import neurokit2 as nk
import os
import time

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")


# raw data masking, .csv to .csv
# extract raw files from raw_dir, delete private info, and save them in new_dir.
def raw_data_masking(raw_dir, new_dir):
    
    labels = ['fast1', 'fast2', 'glucose1', 'glucose2']
    date_list = os.listdir(raw_dir) 

    for date in date_list:
        date_path_raw = os.path.join(raw_dir, date)
        date_path_new = os.path.join(new_dir, date)
        if not os.path.isdir(date_path_new):
                os.mkdir(date_path_new)
        for label in labels:
            fold_label_raw = os.path.join(date_path_raw, label)
            fold_label_new = os.path.join(date_path_new, label)
            if not os.path.isdir(fold_label_new):
                    os.mkdir(fold_label_new)    
            raw_data_path_list = os.listdir(fold_label_raw)
            
            for wave_name in raw_data_path_list:
                df = pd.read_csv(os.path.join(fold_label_raw, wave_name), delim_whitespace=True, index_col=False)    
                df = df[9:]
                df.to_csv(os.path.join(fold_label_new, wave_name), index = False, header = False)
            
            print(date, ': csv_to_csv completed.')  


# raw data process, .csv to .csv
# categorize and save data in '/fast1', '/fast2', '/glucose1', '/glucose2' folders.
def csv_to_csv(raw_dir, date_list, new_dir):
    
    labels = ['fast1', 'fast2', 'glucose1', 'glucose2']

    for date in date_list:
        date_path = os.path.join(raw_dir, date)

        for label in labels:
            label_fold = os.path.join(date_path, label)
            new_fold = os.path.join(new_dir, label)
            raw_data_path_list = os.listdir(label_fold)
            
            for wave_name in raw_data_path_list:
                df = pd.read_csv(os.path.join(label_fold, wave_name), delim_whitespace=True, index_col=False)    
                df = df[9:]
                df.to_csv(os.path.join(new_fold, wave_name), index = False, header = False)
            
        print(date, ': csv_to_csv completed.')  

# Detect ECG_R_Peaks from .csv files, and transform them into np.array
# Each sample include ONE wave + HR_value
def one_wave_process(file_path, clean=False, pre_R=128, post_R=255):
    
    data = pd.read_csv(file_path, delim_whitespace=True, index_col=False)
    data = np.array(list(data.values))
    data = data.reshape(-1)
    # To avoid artifacts when ecg record starts, discard first 3 second data - 1536 points
    data = data[1535:]
    if clean:
        data = nk.ecg_clean(data, sampling_rate=512, method="neurokit")
    signals, info = nk.ecg_process(data, sampling_rate=512, method="neurokit")
    #print(info['ECG_R_Peaks'])
    
    R_peak = info['ECG_R_Peaks']
    last_R = R_peak[-1]
    waves = []

    # Add the HR into the data directly
    # Reverse the list and discard the last R point
    for point in R_peak[::-1][1:]:
        start = point - pre_R
        end = point + post_R
        hr_value = last_R - point
        last_R = point
            
        if start > 0:
            wave = np.append(data[start:end],hr_value)
            waves.append(wave)
 
    # total points = 128 + 255 + 1 = 384    
    # shape: [waves, 384(383points + 1 hr_value)] 
    return np.array(waves)[::-1]



# Detect ECG_R_Peaks from .csv files, and transform them into np.array
# Each sample include TWO waves.
def two_wave_process(file_path, clean=False, pre_R=512, post_R=255):
    
    data = pd.read_csv(file_path, delim_whitespace=True, index_col=False)
    data = np.array(list(data.values))
    data = data.reshape(-1)
    # To avoid artifacts when ecg record starts, discard first 3 second data - 1536 points
    data = data[1535:]
    if clean:
        data = nk.ecg_clean(data, sampling_rate=512, method="neurokit")
    signals, info = nk.ecg_process(data, sampling_rate=512, method="neurokit")
    #print(info['ECG_R_Peaks'])
    
    R_peak = info['ECG_R_Peaks']
    last_R = R_peak[-1]
    waves = []
    
    # Reverse the list and discard the last R point
    # No HR value
    for point in R_peak[::-1][1:]:
        start = point - pre_R
        end = point + post_R
        last_R = point
            
        if start > 0:
            waves.append(data[start:end+1])
 
    # total points = 512 + 256 = 768    
    # shape: [batch size, 768] 
    return np.array(waves)[::-1]



# Process csv. filse, combine f+f and f+g waves into a new np.array (f=fast data, g=glucose data)
# 'path' is a folder, including four folders: 'fast1', 'fast2', 'glucose1', 'glucose2'
# one_wave=True: waves points 384(383points + 1heart_rate)
# mode = ['stack', 'subtract', 'individual', 'not_combine']
def preprocess_ecg_data(path, meal=False, one_wave=True, mode='not_combine'):

    fast_dir = path+'/fast1'
    fast_ecg_list = [os.path.join(fast_dir, ecg) for ecg in os.listdir(fast_dir)] 
    glucose_dir = path+'/glucose1'
    glucose_ecg_list = [os.path.join(glucose_dir, ecg) for ecg in os.listdir(glucose_dir)] 
        
    # For 'fast2' and 'glucose2'
    if meal:
        fast2_dir = path+'/fast2'
        fast2_ecg_list = [os.path.join(fast2_dir, ecg) for ecg in os.listdir(fast2_dir)] 
        fast_ecg_list = fast_ecg_list + fast2_ecg_list

        glucose2_dir = path+'/glucose2'
        glucose2_ecg_list = [os.path.join(glucose2_dir, ecg) for ecg in os.listdir(glucose2_dir)] 
        glucose_ecg_list = glucose_ecg_list + glucose2_ecg_list
        
    if one_wave:
        # shape: [batch size, 384(383points + 1heart_rate)] 
        fast_waves = np.vstack([one_wave_process(file_path) for file_path in fast_ecg_list])
        glucose_waves = np.vstack([one_wave_process(file_path) for file_path in glucose_ecg_list])
    else:
        # shape: [batch size, 768] 
        fast_waves = np.vstack([two_wave_process(file_path) for file_path in fast_ecg_list])
        glucose_waves = np.vstack([two_wave_process(file_path) for file_path in glucose_ecg_list])

    fast_num = len(fast_waves)
    glucose_num = len(glucose_waves)

    ff_waves,fg_waves = [], []
    
    if mode == 'not_combine':
        ff_waves,fg_waves = fast_waves, glucose_waves

    else:
        for i in range(fast_num):
            for ii in range(i+1, fast_num):
                if mode == 'stack':
                    ff_wave = np.row_stack([fast_waves[i], fast_waves[ii]]).reshape(-1)
                    ff_waves.append(ff_wave)    

                elif mode == 'subtract':
                    ff_wave = fast_waves[ii] - fast_waves[i]
                    ff_waves.append(ff_wave)
                    
                elif mode == 'individual':
                    ff_waves = fast_waves
                    break   

            for g in range(glucose_num):    

                if mode == 'stack':
                    fg_wave = np.row_stack([fast_waves[i], glucose_waves[g]]).reshape(-1)
                    fg_waves.append(fg_wave)    

                elif mode == 'subtract':
                    fg_wave = glucose_waves[g] - fast_waves[i]
                    fg_waves.append(fg_wave)    

                elif mode == 'individual':
                    fg_waves = glucose_waves
                    break   

    # mode='stack': output: [batch size, 768 points] or [batch size, 1536 points]
    # mode='subtract': output: [batch size, 768 points] or [batch size, 1536 points]
    return np.vstack([ff_waves,fg_waves]), len(ff_waves), len(fg_waves)

def normalize_data(data):

    scaler = MinMaxScaler()
    normalized_dataset = scaler.fit_transform(data.t())
    data = normalized_dataset.T
    data = torch.tensor(data)
    
    return data


# My ecg dataset
class MyDataset(Dataset):

    def __init__(self, data_dir, meal=False, one_wave=True, mode='not_combine'):
        super().__init__() 

        self.data_dir = data_dir
        self.meal = meal
        self.one_wave = one_wave
        self.mode = mode
        data, self.ff_num, self.fg_num = preprocess_ecg_data(
                self.data_dir, self.meal, self.one_wave, self.mode)
        #scaler = MinMaxScaler()
        #normalized_data = scaler.fit_transform(data.T)
        #data = normalized_data.T
        self.processed_data = data

    def __len__(self):
        return self.ff_num + self.fg_num

    def __getitem__(self, idx):
        
        # wave [points]
        wave = self.processed_data[idx]
        label = 0 if idx < self.ff_num else 1
        
        # Return the processed data and any labels if available
        # wave [points], label [1]
        return wave, label

    def get_shape(self):
        wave = self.processed_data[0]
        return wave.shape



# Build and save a dataset(conclude ff and fg waves) in a .pt file.
def csv_to_pt(root, meal=False, one_wave=True, mode='not_combine', file_path='dataset.pt'):

    print('==> Dataset building starts...')
    start = time.time()

    dataset = MyDataset(root, meal, one_wave, mode)
    print('==> Building completed, saving starts...')

    torch.save(dataset, file_path)
    
    end = time.time()
    print('==> Saving completed! Total Time: %.2f seconds' % (end-start))
    print('==> Total waves: {}. FF: {}, FG: {}'.format(len(dataset), dataset.ff_num, dataset.fg_num))
    print('==> Sample shape:', dataset.get_shape())



# Return DataLoader(s)
def ecg_loader(root, load_from_pt=True, batch_size=8, shuffle=True, 
               NUM_WORKERS=1, PIN_MEMORY=True, DROP_LAST=True):

    # Load dataset from a .pt file
    if load_from_pt:
        print('==> Loading data...')  
        dataset = torch.load(root)

    # Build dataset from scratch in a folder
    else:
        print('==> Dataset building starts...')
        start = time.time()
        
        dataset = MyDataset(root, meal=False, one_wave=True, mode='not_combine')
        end = time.time()
        print('==> Building completed! Total Time: %.2f seconds' % (end-start))

    total_num = len(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST)
        
    print('==> DataLoader completed!')
    print('==> Data total samples:', total_num)
        
    return dataloader


# input_label: [batch_size]
# new_label: [batch_size / 2, 1]
def remake_labels(input_label):

    assert len(input_label) % 2 == 0, 'Batch num is odd: {}'.format(int(len(input_label)))
    
    l_l = torch.split(input_label, int(len(input_label)/2), dim=0)
    
    label_1, label_2 = l_l[0], l_l[1]
    
    new_label = []
    for i in range(len(label_1)):
        if label_1[i]==label_2[i]:
            new_label.append([0])
        else:
            new_label.append([1])

    new_label = torch.tensor(new_label).float() 
    
    return new_label


   