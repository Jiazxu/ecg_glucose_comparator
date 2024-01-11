# -*- coding: utf-8 -*-
# Traditional supervised learning
# Dataset is small.

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import argparse
import time
from tqdm import tqdm

import os, sys
sys.path.append("..") 

from utils.data_utils import *
from utils.model_utils import *

# ---------------------- Choose model -------------------------------------------- 

#model_name = 'MobileNetV2'
#from models.MobileNetV2 import *
model_name = 'effnetv2_ecg_l_xxxs_20240105_epoch30_1e-4'
from models.EfficientNetV2 import *

# ---------------------- Configuration and parameters -------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

ONE_WAVE_PT_TRAINING = '../dataset/training_data/PT_DIR/one_wave.pt' 
ONE_WAVE_PT_VAL =      '../dataset/val_data/PT_DIR/one_wave.pt' 
ONE_WAVE_PT_TEST =     '../dataset/test_data/PT_DIR/one_wave.pt' 

checkpoint_path = '../checkpoint/' + model_name
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

best_acc = 0    # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epoch = 30
BATCH_SIZE = 128 # train and test batch size    
NUM_WORKERS = 1  # CPU cores
LR = 1e-4
MOMENTUM = 0.9
LR_DECAY = 0.0005 # weight decay
LR_INIT = 0.01

#classes = ('fast', 'glucose')

# ---------------------- Parser on the terminal -------------------------------------------------
parser = argparse.ArgumentParser(description='Pytorch CIFAR10 Training')
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
                batch_size=BATCH_SIZE, shuffle=True, NUM_WORKERS=NUM_WORKERS)

    # ---------------------- Model -------------------------------------------------------
    print('==> Building model...')  

    net = effnetv2_ecg_l_xxxs()

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

    counter = []
    loss_history = [] 
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

        for i, (waves, labels) in enumerate(tqdm(trainloader)):

            # Add one dimension to waves and labels
            waves = waves.unsqueeze(1).float()
            labels = labels.unsqueeze(1).float()
            waves, labels = waves.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(waves)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    

            train_loss += loss.item()

            predicted_labels = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()   
            TP += ((predicted_labels + labels) == 2).sum().item()
            TN += ((predicted_labels + labels) == 0).sum().item()
            CP += labels.sum().item()
            CN += (labels.size(0) - labels.sum().item())    

            if (i+1) % 5 == 0:

                print('Train --> Step: {} \tLoss: {:.4f} \tAcc: {:.3f}% \tCorrect/Total: ({}/{})\n\
         Specificity: {:.3f}% \tSensitivity: {:.3f}% \tTP/TN: {}/{}'
                        .format(i+1, train_loss/(i+1), 100.*correct/total, correct, total, \
                                100.*TP/CP, 100.*TN/CN, TP, TN))
                
                iteration_number += 5
                counter.append(iteration_number)
                loss_history.append(loss.item())

        train_time = time.time()    
        
        # ---------------------- Validation -------------------------------------------------------
        val_interval = 3
        if (epoch+1) % val_interval == 0:
            print('\n==> Epoch <%d> testing...' % epoch)
            net.eval()
            test_loss = 0 
            correct = 0
            total = 0
            # True Positive and negtive
            TP, TN = 0, 0 
            # Condition Positive and negtive
            CP, CN = 0, 0   

            with torch.no_grad():
                for i, (waves, labels) in enumerate(tqdm(valloader)):
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

                print('Test --> Step: {} \tLoss: {:.4f} \tAcc: {:.3f}% \tCorrect/Total: ({}/{})\n\
             Specificity: {:.3f}% \tSensitivity: {:.3f}% \tTP/TN: {}/{}'
                            .format(i+1, test_loss/(i+1), 100.*correct/total, correct, total, \
                                    100.*TP/CP, 100.*TN/CN, TP, TN))
            
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
        
        test_time = time.time()
        
        # ---------- simple save -----------
        torch.save(net.state_dict(), checkpoint_path + '/' + model_name + '_simple.pth')
        
        print(' _______________________\n\
|                      \t|\n\
| Total time: {:.2f} \t|\n\
| Epoch time: {:.2f} \t|\n\
| Training time: {:.2f} \t|\n\
| Testing time: {:.2f} \t|\n\
|_______________________|'. \
               format(time.time() - total_time,
                      time.time() - start_time,
                      train_time - start_time,
                      test_time - train_time)
             )
    # plot loss history
    plt.plot(counter, loss_history)
    plt.show()