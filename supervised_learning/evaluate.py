# -*- coding: utf-8 -*-

import numpy as np
import torch 
import torch.nn.functional as F
import time
import argparse
from tqdm import tqdm

import os, sys
sys.path.append("..") 

from utils.data_utils import *
from utils.model_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# classes = ('fast:0', 'glucose:1')
parser = argparse.ArgumentParser(description='Pytorch ECG Training')
parser.add_argument('--resume', '-r', action='store_true', 
                    help='resume from checkpoint')
args = parser.parse_args()

if __name__ == '__main__':

    BATCH_SIZE = 128

    # Define the path to the .pth file
    model_name = 'effnetv2_ecg_l_xxxs_20240105_epoch100_1e-4'
    model_path = '../checkpoint/'+model_name+'/'+model_name+'_simple.pth'
    from models.EfficientNetV2 import *
    net = effnetv2_ecg_l_xxxs()

    ONE_WAVE_PT_TRAINING = '../dataset/training_data/PT_DIR/one_wave.pt' 
    ONE_WAVE_PT_VAL =      '../dataset/val_data' 
    ONE_WAVE_PT_TEST =     '../dataset/test_data' 
    #ONE_WAVE_PT_ADD =      '../dataset/additional_data'     

    ONE_WAVE_PT_EVALUATE = ONE_WAVE_PT_ADD
    print(ONE_WAVE_PT_EVALUATE)

    load_model = True

    if load_model:
        print('==> Load model...')    

        # Load the dictionary
        checkpoint = torch.load(model_path) 

        for k in list(checkpoint.keys()):
            k_new = k.replace("module.","")
            checkpoint[k_new] = checkpoint[k]
            del checkpoint[k]   

        net.load_state_dict(checkpoint)

    # ---------------------- Resume training ------------------------------------------------
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint...')
        checkpoint_path = '../checkpoint/' + model_name
        assert os.path.isdir(checkpoint_path), \
                'Error: found no checkpoint/model directory!'
        checkpoint = torch.load(checkpoint_path + '/' + model_name + '.pth')
        
        for k in list(checkpoint['model']):
            k_new = k.replace("module.","")
            checkpoint['model'][k_new] = checkpoint['model'][k]
            del checkpoint['model'][k]   

        net.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        print('The last model accuracy is {:.3f}%'.format(best_acc))
        last_epoch = checkpoint['epoch']
        print('Last training epoch: {}'.format(last_epoch))        
        start_epoch = checkpoint['epoch'] + 1
        seed = checkpoint['seed']


    net = net.to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)

    # Set the model to evaluation mode
    net.eval()

    evaloader = ecg_loader(ONE_WAVE_PT_EVALUATE+'/PT_DIR/one_wave.pt', load_from_pt=True,
                batch_size=BATCH_SIZE, shuffle=True, NUM_WORKERS=1)

    criterion = torch.nn.BCELoss()  


    # ---------------------- Evaluate individually ------------------------------------------------------
    test_loss = 0 
    correct = 0
    total = 0
    # True Positive and negtive
    TP, TN = 0, 0 
    # Condition Positive and negtive
    CP, CN = 0, 0

    with torch.no_grad():
        for i, (waves, labels) in enumerate(tqdm(evaloader)):

            # Add one dimension to waves and labels        
            waves = waves.unsqueeze(1).float()
            labels = labels.unsqueeze(1).float()
            waves, labels = waves.to(device), labels.to(device)

            outputs = net(waves)
            loss = criterion(outputs, labels)       

            test_loss += loss.item()    

            predicted_labels = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item() 
            TP += ((predicted_labels + labels) == 2).sum().item()
            TN += ((predicted_labels + labels) == 0).sum().item()
            CP += labels.sum().item()
            CN += (labels.size(0) - labels.sum().item())    

        print('Evaluation --> Loss: {:.4f} \tAcc: {:.3f}% \tCorrect/Total: ({}/{})\n\
            Specificity: {:.3f}% \tSensitivity: {:.3f}% \tTP/TN: {}/{}'
                        .format(test_loss/total, 100.*correct/total, correct, total, \
                                100.*TP/CP, 100.*TN/CN, TP, TN))


    # ---------------------- Evaluate wave-wisely ------------------------------------------------------
    test_loss = 0 
    correct = 0
    total = 0
    # True Positive and negtive
    TP, TN, FP, FN = 0, 0, 0, 0
    # Condition Positive and negtive
    CP, CN = 0, 0
    waves_num_total = 0
    wrong_pred = []

    # Build wave-wise input
    # fast1:0, glucose1:1 
    print('Fast waves:')
    fast_dir = ONE_WAVE_PT_EVALUATE+'/NEW_CSV_DIR/fast1'
    fast_ecg_list = [os.path.join(fast_dir, ecg) for ecg in os.listdir(fast_dir)] 
    i = len(fast_ecg_list)
    for file_path in tqdm(fast_ecg_list):
        waves = one_wave_process(file_path)
        waves_num = len(waves)
        waves_num_total = waves_num_total + waves_num
        labels = np.zeros(waves_num)
        waves = waves.copy()
        waves, labels = torch.tensor(waves), torch.tensor(labels)
        
        with torch.no_grad():
            condition = labels[0]
            # Add one dimension to waves and labels        
            waves = waves.unsqueeze(1).float()
            labels = labels.unsqueeze(1).float()
            waves, labels = waves.to(device), labels.to(device)

            outputs = net(waves)
            loss = criterion(outputs, labels)       

            test_loss += loss.item()    

            predicted_labels = torch.round(outputs)

            # calculate whole wave's score 
            prediction_score = predicted_labels.sum().item() 

            if prediction_score < waves_num/2:  # Prediction is Negtive/fast;fast1:0, glucose1:1

                if condition.item() == 0:       # Condition Negtive
                    TN += 1
                    CN += 1
                else:                           # Condition Positive 
                    FN += 1
                    CP += 1 
            else:                               # Prediction is Positive/glucose
                wrong_pred.append(file_path)

                if condition.item() == 0:       # Condition Negtive
                    CN += 1
                    FP += 1
                else:                           # Condition Positive
                    TP += 1
                    CP += 1

    print('Glucose waves:')
    glucose_dir = ONE_WAVE_PT_EVALUATE+'/NEW_CSV_DIR/glucose1'
    glucose_ecg_list = [os.path.join(glucose_dir, ecg) for ecg in os.listdir(glucose_dir)] 
    i = i + len(glucose_ecg_list)
    
    for file_path in tqdm(glucose_ecg_list):
        waves = one_wave_process(file_path)
        waves_num = len(waves)
        waves_num_total = waves_num_total + waves_num
        labels = np.ones(waves_num)
        waves = waves.copy()
        waves, labels = torch.tensor(waves), torch.tensor(labels)

        with torch.no_grad():
            condition = labels[0]
            # Add one dimension to waves and labels        
            waves = waves.unsqueeze(1).float()
            labels = labels.unsqueeze(1).float()
            waves, labels = waves.to(device), labels.to(device)

            outputs = net(waves)
            loss = criterion(outputs, labels)       

            test_loss += loss.item()    

            predicted_labels = torch.round(outputs)

            # calculate whole wave's score 
            prediction_score = predicted_labels.sum().item() 

            if prediction_score < waves_num/2:  # Prediction is Negtive/fast;fast1:0, glucose1:1

                wrong_pred.append(file_path)

                if condition.item() == 0:       # Condition Negtive
                    TN += 1
                    CN += 1
                else:                           # Condition Positive 
                    FN += 1
                    CP += 1 
            else:                               # Prediction is Positive/glucose

                if condition.item() == 0:       # Condition Negtive
                    CN += 1
                    FP += 1
                else:                           # Condition Positive
                    TP += 1
                    CP += 1
                
    print("Total evaluated waves:", waves_num_total)
    print('Evaluation wave-wise--> Loss: {:.4f} \tAcc: {:.3f}% \tCorrect/Total: ({}/{})\n\
         Specificity: {:.3f}% \tSensitivity: {:.3f}% \tTP/TN: {}/{} \tLabel 0/1: {}/{}'
                        .format(test_loss/waves_num_total, 100.*(TN+TP)/i, (TN+TP), i, \
                                100.*TP/CP, 100.*TN/CN, TP, TN, (TN+FP), (FN+TP)))
    if i != []:
        print('Wrong prediction list:')
        for wave in wrong_pred:
            print('\t',wave) 
