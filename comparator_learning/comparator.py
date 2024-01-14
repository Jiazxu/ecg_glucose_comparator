# -*- coding: utf-8 -*-
# Version ONE:
# Version TWO:
# VErsion THREE:
# Version FOUR:
# Version FIVE: During training, an ref_train value from waves_1 is obtained, then added it to the validation process.
#               --> out, ref_train = net(waves_1, waves_2)
#               --> out, _ = net(ref_train, waves_val, evaluate=True)


import torch 
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

import os, sys
sys.path.append("..") 
import argparse
import time

from tqdm import tqdm

from utils.data_utils import *
from utils.model_utils import *

# ---------------------- Choose model -------------------------------------------- 

model_name = 'effnetv2_ecg_comparator_v5_l_xxxs_20240111_epoch200_4e-3'
from models.EfficientNetV2_comparator_v5 import *


# ---------------------- Configuration and parameters -------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

ONE_WAVE_PT_TRAINING = '../dataset/training_data/PT_DIR/one_wave.pt' 
ONE_WAVE_PT_VAL =      '../dataset/val_data/PT_DIR/one_wave.pt' 
ONE_WAVE_PT_TEST =     '../dataset/test_data/PT_DIR/one_wave.pt' 

checkpoint_path = '../checkpoint/' + model_name
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

best_acc = 0    # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epoch = 200
BATCH_SIZE = 128 # train and val batch size    
NUM_WORKERS = 1  # CPU cores
#LR = 1e-3
LR = 4e-3
MOMENTUM = 0.9
LR_DECAY = 0.0005 # weight decay
LR_INIT = 0.01

#classes = ('fast', 'glucose')


# ---------------------- Parser on the terminal -------------------------------------------------
parser = argparse.ArgumentParser(description='Pytorch ECG Training')
parser.add_argument('--lr', default=LR, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', 
                    help='resume from checkpoint')
args = parser.parse_args()


# ---------------------- Train ------------------------------------------------------------------
if __name__ == '__main__':
    
    try:
        seed
    except NameError:
        seed = torch.initial_seed()
        print('Used seed: {}'.format(seed))
    else:
        torch.manual_seed(seed)
   
    # ---------------------- Data and dataloader -----------------------------------------  
    trainloader = ecg_loader(ONE_WAVE_PT_TRAINING, load_from_pt=True, 
                batch_size=BATCH_SIZE, shuffle=True, NUM_WORKERS=NUM_WORKERS)
    valloader = ecg_loader(ONE_WAVE_PT_VAL, load_from_pt=True,
                batch_size=int(BATCH_SIZE/2), shuffle=True, NUM_WORKERS=NUM_WORKERS)

    # ---------------------- Model -------------------------------------------------------
    print('==> Building model...')  

    net = effnetv2_ecg_comparator_v5_l_xxxs()

    # print the model's number of parameters
    cal_params(net)

    # torch 2.0
    # net = torch.compile(net)
    
    net = net.to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)
        cudnn.benchmark = True  

    # ---------------------- Loss function -----------------------------------------------
    print('==> Building criterion...') 
    # Binary classification
    criterion = nn.BCELoss()  

    # ---------------------- Optimizer ---------------------------------------------------
    print('==> Building optimizer...')
    optimizer = optim.Adam(net.parameters(), lr=args.lr)    
    #optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                      momentum=MOMENTUM, weight_decay=LR_DECAY)
    
    
    # ---------------------- Lr_scheduler ---------------------------------------------------
    print('==> Building lr_scheduler...')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # ---------------------- Resume training ------------------------------------------------
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint...')
        checkpoint_path = '../checkpoint/' + model_name
        assert os.path.isdir(checkpoint_path), \
                'Error: found no checkpoint/model directory!'
        checkpoint = torch.load(checkpoint_path + '/' + model_name + '.pth')
        net.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        print('The last model accuracy is {:.3f}%'.format(best_acc))
        start_epoch = checkpoint['epoch'] + 1
        seed = checkpoint['seed']
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=MOMENTUM, weight_decay=LR_DECAY)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Traverse all epoch
    total_time = time.time()

    counter_train = []
    counter_val = []
    loss_history_train = []
    loss_history_val = []
    acc_train = []
    acc_val = [] 
    iteration_number = 0

    for epoch in range(start_epoch, start_epoch+total_epoch):
        
        # ---------------------- Start training ---------------------------------------------
        print('\n==> Epoch <%d> starting training...' % epoch)
        start_time = time.time()
        net.train()
        train_loss = 0 
        correct = 0 
        total = 0 
        # True Positive and negtive; Condition Positive and negtive
        TP, TN, CP, CN = 0, 0, 0, 0

        for i, (waves_train, labels_train) in enumerate(tqdm(trainloader)):

            # waves: [batch size, 384 points]
            # Split into two sequences on batch_size
            # labels: [batch size] --> [batch size/2, 1]
            labels = remake_labels(labels_train)
            waves_split = torch.split(waves_train, int(len(waves_train)/2), dim=0)    
            waves_1, waves_2 = waves_split[0], waves_split[1]
            
            # Add one dimension to waves
            waves_1 = waves_1.unsqueeze(1).float()
            waves_2 = waves_2.unsqueeze(1).float()            
            waves_1, waves_2, labels = waves_1.to(device), waves_2.to(device), labels.to(device)

            optimizer.zero_grad()
            # get wave_1's reference, which will be useful in the validation part.
            out, ref_train_1 = net(waves_1, waves_2)

            # make sure ref_train and labels_train_1's batch size = BATCH_SIZE/2
            if len(labels_train)==BATCH_SIZE:
                ref_train = ref_train_1
                lable_split = torch.split(labels_train, int(len(labels_train)/2), dim=0)
                labels_train = lable_split[0]

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()    

            train_loss += loss.item()  

            predicted_labels = torch.round(out)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()   
            TP += ((predicted_labels + labels) == 2).sum().item()
            TN += ((predicted_labels + labels) == 0).sum().item()
            CP += labels.sum().item()
            CN += (labels.size(0) - labels.sum().item())    

            monitor_interval = 5
            if (i+1) % monitor_interval == 0:
                print('Training --> Step: {} \tLoss: {:.4f} \tAcc: {:.3f}% \tCorrect/Total: ({}/{})\n\
         Specificity: {:.3f}% \tSensitivity: {:.3f}% \tTP/TN: {}/{}'
                        .format(i+1, train_loss/(i+1), 100.*correct/total, correct, total, \
                                100.*TP/CP, 100.*TN/CN, TP, TN))
                
                iteration_number += monitor_interval
                counter_train.append(iteration_number)
                loss_history_train.append(train_loss/(i+1))
                acc_train.append(100.*correct/total)

        train_time = time.time()    
        
        # ---------------------- Validation -------------------------------------------------------
        val_interval = 1
        if (epoch+1) % val_interval == 0:
            print('\n==> Epoch <%d> starting validating...' % epoch)
            net.eval()
            val_loss = 0 
            correct = 0
            total = 0
            # True Positive and negtive; Condition Positive and negtive
            TP, TN, CP, CN = 0, 0, 0, 0

            with torch.no_grad():
                for i, (waves_val, labels_val) in enumerate(tqdm(valloader)): 

                    new_label = []
                    for i in range(len(labels_val)):
                        if labels_val[i]==labels_train[i]:
                            new_label.append([0])
                        else:
                            new_label.append([1])               

                    labels = torch.tensor(new_label).float() 

                    # Add one dimension to waves
                    waves_val = waves_val.unsqueeze(1).float()            
                    waves_val, labels = waves_val.to(device), labels.to(device)        

                    optimizer.zero_grad()
                    out, _ = net(ref_train, waves_val, evaluate=True)
                    loss = criterion(out, labels)  

                    val_loss += loss.item()    

                    predicted_labels = torch.round(out)
                    total += labels.size(0)
                    correct += (predicted_labels == labels).sum().item() 
                    TP += ((predicted_labels + labels) == 2).sum().item()
                    TN += ((predicted_labels + labels) == 0).sum().item()
                    CP += labels.sum().item()
                    CN += (labels.size(0) - labels.sum().item())    

                print('Validating --> Step: {} \tLoss: {:.4f} \tAcc: {:.3f}% \tCorrect/Total: ({}/{})\n\
             Specificity: {:.3f}% \tSensitivity: {:.3f}% \tTP/TN: {}/{}'
                            .format(i+1, val_loss/(i+1), 100.*correct/total, correct, total, \
                                    100.*TP/CP, 100.*TN/CN, TP, TN))

                counter_val.append(epoch)
                loss_history_val.append((val_loss/(i+1)))
                acc_val.append(100.*correct/total)

            # ---------------------- save checkpoint --------------------------------------------
            acc = 100.*correct/total
            if acc > best_acc:
                print('\n==> Saving checkpoint...')
                state = {
                    'epoch': epoch,
                    'model': net.state_dict(),
                    'acc': acc,
                    'seed': seed,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                torch.save(state, checkpoint_path + '/' + model_name + '.pth')
                best_acc = acc
            
        scheduler.step()
        val_time = time.time()
        
        # ---------------------- simple save ------------------------------------------------------------------
        torch.save(net.state_dict(), checkpoint_path + '/' + model_name + '_simple.pth')

        print(' _______________________\n\
|                      \t|\n\
| Total time: {:.2f} \t|\n\
| Epoch time: {:.2f} \t|\n\
| Training time: {:.2f} \t|\n\
| Val time: {:.2f} \t|\n\
|_______________________|'. \
               format(time.time() - total_time,
                      time.time() - start_time,
                      train_time - start_time,
                      val_time - train_time)
             )

    # plot Loss history
    plt.subplot(2, 1, 1)

    total_i = counter_train[-1]
    counter_train = [(i*total_epoch/total_i) for i in counter_train]
    plt.plot(counter_train, loss_history_train, color='red', label='train loss')
   
    plt.plot(counter_val, loss_history_val, color='green', label='val loss')
    
    #plt.title('Loss history')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # plot Acc history
    plt.subplot(2, 1, 2)   
    plt.plot(counter_train, acc_train, color='red', label='train acc') 
    plt.plot(counter_val, acc_val, color='green', label='val acc')
    
    #plt.title('Acc history')
    plt.xlabel('Epochs')
    plt.ylabel('Acc(%)')
    plt.legend()

    plt.show()

