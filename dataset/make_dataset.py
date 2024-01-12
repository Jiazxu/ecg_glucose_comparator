# -*- coding: utf-8 -*-
# make_dataset.py
# Establish training set, validation set, test set by date randomly


import os, sys
sys.path.append("..")
import random
import tkinter as tk
from tkinter import messagebox

import warnings
warnings.filterwarnings("ignore")

from utils import data_utils


# dataset
if __name__ == '__main__':

    RAW_DIR = '../dataset/data_raw'
    NEW_DIR = '../dataset'
    ONE_WAVE_PT = NEW_DIR+'/training_data/PT_DIR/one_wave.pt'
    dataloader = data_utils.ecg_loader(ONE_WAVE_PT, load_from_pt=True, batch_size=4, shuffle=True)

    import numpy as np
    import neurokit2 as nk
    import matplotlib.pyplot as plt

    for waves, labels in dataloader:

        print(waves.size(), labels.size())
        nk.signal_plot([w for w in waves], labels=['fast' if l==0 else 'glucose' for l in labels], subplots=True)
        plt.show()

        break   

'''    
    # Establish training set, validation set, test set by date randomly.
    date_list = os.listdir(RAW_DIR)
    random.shuffle(date_list)

    # split ratio 6:2:2 
    # 8-2-3
    training_list = date_list[:8]
    val_list = date_list[8:10]
    test_list = date_list[10:]
    print(training_list, val_list, test_list)
    # ['1006', '1005', '0924', '0926', '0928', '0930', '1002', '1008'] ['0927', '1007'] ['0923', '0925', '0921']

    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno("Confirm", "Proper splitting?")
    if result == True:
        print("Confirm and build dataset")
    
        # csv to csv, 
        # save .pt file
        SAVE_DIR = NEW_DIR+"/training_data/NEW_CSV_DIR"
        data_utils.csv_to_csv(RAW_DIR, training_list, SAVE_DIR)
        data_utils.csv_to_pt(root=SAVE_DIR, meal=False, one_wave=True, mode='not_combine', 
                             file_path=NEW_DIR+'/training_data/PT_DIR/one_wave.pt')    

        SAVE_DIR = NEW_DIR+"/val_data/NEW_CSV_DIR"
        data_utils.csv_to_csv(RAW_DIR, val_list, SAVE_DIR)
        data_utils.csv_to_pt(root=SAVE_DIR, meal=False, one_wave=True, mode='not_combine', 
                             file_path=NEW_DIR+'/val_data/PT_DIR/one_wave.pt')

        SAVE_DIR = NEW_DIR+"/test_data/NEW_CSV_DIR"
        data_utils.csv_to_csv(RAW_DIR, test_list, SAVE_DIR)
        data_utils.csv_to_pt(root=SAVE_DIR, meal=False, one_wave=True, mode='not_combine', 
                             file_path=NEW_DIR+'/test_data/PT_DIR/one_wave.pt')

    else:
        print("Cancel.")
'''


    
''' 
# additional dataset
if __name__ == '__main__':

    UNMASKED_DIR = '../dataset/additional_data/data_unmasked'
    RAW_DIR = '../dataset/additional_data/data_raw'
    NEW_DIR = '../dataset/additional_data'

    #data_utils.raw_data_masking(UNMASKED_DIR, RAW_DIR)

    # csv to csv, 
    # save .pt file
    SAVE_DIR = NEW_DIR+"/NEW_CSV_DIR"
    ONE_WAVE_PT = NEW_DIR+'/PT_DIR/one_wave.pt'

    date_list = os.listdir(RAW_DIR)

    data_utils.csv_to_csv(RAW_DIR, date_list, SAVE_DIR)
    data_utils.csv_to_pt(root=SAVE_DIR, meal=False, one_wave=True, mode='not_combine', 
                         file_path=ONE_WAVE_PT) 


    # Test dataloader and visualize samples    
    dataloader = data_utils.ecg_loader(ONE_WAVE_PT, load_from_pt=True, batch_size=8, shuffle=True)

    import numpy as np
    import neurokit2 as nk
    import matplotlib.pyplot as plt

    for waves, labels in dataloader:

        print(waves.size(), labels.size())
        nk.signal_plot([w for w in waves], labels=[l for l in labels], subplots=True)
        plt.show()

        break   
'''    
