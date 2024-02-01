---
# A Personalized blood glucose level binary classification deep neural network, Acc=100%
---

## I. OUTLINE - Model and Result

**1.Model Architecture**

<div align="center">

<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/description/comparator_learning.png">
</div>

**2.Training Loss&Acc**

<div align="center">

<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/checkpoint/effnetv2_ecg_comparator_v5_l_xxxs_20240111_epoch100_4e-3/effnetv2_ecg_comparator_v5_l_xxxs_20240111_epoch100_4e-3.png">
</div>

**3.Evaluation Acc**

test dataset:

```terminal
# Load model with pretrained weights
cd ecg_glucose_comparator
python evaluate.py --resume
```

Testing result:

**beat-wise:** Acc: 98.910% Correct/Total: (998/1009)

**wave-wise (30s sample):** Acc: 100.000% Correct/Total: (32/32)

```terminal
../dataset/test_data
==> Load model...
==> Resuming from checkpoint...
The last model accuracy is 97.585%
Last training epoch: 89
==> Loading data...
==> DataLoader completed!
==> Data total samples: 5363
==> Loading data...
==> DataLoader completed!
==> Data total samples: 1009
checkpoint 0: 50 50
checkpoint 1: torch.Size([50, 1]) torch.Size([50, 8, 96])
checkpoint 2: 49.0 49.0
100%|█████████████████████████████████████████████████| 1009/1009 [00:18<00:00, 54.96it/s]
Evaluation beat-wise--> Loss: 0.0168     Acc: 98.910%    Correct/Total: (998/1009)
         Specificity: 100.000%  Sensitivity: 98.588%    TP/TN: 230/768  Label 0/1: 779/230
Fast waves:
100%|█████████████████████████████████████████████████| 25/25 [00:08<00:00,  2.89it/s]
Glucose waves:
100%|█████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.78it/s]
Total evaluated beats: 1009
Evaluation wave-wise--> Loss: 0.0252    Acc: 100.000%   Correct/Total: (32/32)
         Specificity: 100.000%  Sensitivity: 100.000%   TP/TN: 7/25     Label 0/1: 25/7
Wrong prediction list:
```

## II. Dataset

**All ECG waves were recorded from a specific individual (myself)** and **the model can only be applied to this specific individual (myself) !**

**Total 264 one-lead ECG wave samples (30s for each wave, divided into 8,543 beats totally) recorded in 13 days.**

Recording equipment: iWatch Ultra (Software: Version 1.9; Sample Rate: 512 hertz)

I splitted the data randomly **by date** into three dataset: training dataset, validation dataset and test dataset. The splitting ratio is 60:20:20.

🤔△If all wave samples were mixed and splitted randomly, it could deteriorate the model performance in the real world. Because the waves recorded in the same date are similar.

These ECG waves were classified into two labels:

Label **[fast/f]**: low blood glucose (BG) level, recorded in fasting condition.

Label **[glocose/g]**: high blood glucose (BG) level, recorded after consuming a large amount of sugar.

<div align="center">

<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/dataset/example.png">

</div>

## III. Model Architecture

This model's backbone is [EfficientNetV2(Fused-MBConv)](https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py).

**1. Classic supervised learning**

As a benchmark, I feeded the data into a classic EfficientNetV2 model.
Here is the model architecture:

<div align="center">

<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/description/supervised_learning.png">
</div>

**2. Comparator learning**

However, due to the scale of the dataset, the training process was overfitted really easily, even though the model was so small (no more than 9,000 parameters!).

🤔Thus, I reconstructed the architecture and training workflow. From a mathematical perspective, by **comparing the similarity of two beats randomly**, the training data increases exponentially and could alleviate the overfitting.

In the EffNetV2_comparator, I used two MBConv block: the **feature_encoder** extracts the features from the similarities of two beats; and the **feature_comparator** works as a non-linear hyper-dimension distance function, instead of a euclidean metric function or a linear classifier.

After finishing this project, I found this training style looks like Contrastive Learning lol🤣.

Here is my model:

<div align="center">

<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/description/comparator_learning.png">

</div>

## IV. Training and validating process

1. Classic supervised learning

<div align="center">

<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/checkpoint/effnetv2_ecg_l_xxxs_20240111_epoch150_1e-4/effnetv2_ecg_l_xxxs_20240111_epoch150_1e-4.png">

</div>

2. Comparator learning

<div align="center">

<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/checkpoint/effnetv2_ecg_comparator_v5_l_xxxs_20240111_epoch100_4e-3/effnetv2_ecg_comparator_v5_l_xxxs_20240111_epoch100_4e-3.png">
</div>

## V. About this project

Since I was a medicine student a few years ago, I have been curious about the application of big data on healthcare.

🤔Human body is a complex system, producing high dimensional data all the time. For decades, Scientists are trying to analysis the manifold from these complicated data and its projections on different dimensions and levels, and related them with all kinds of phenomena, such as diseases, mental conditions, pathological changes and results from labs and medical equipments. On the other hand, it is also a really exciting road to discover new data source and new methdology to manage human beings heathcare.

From a statistical aspect, however, we should exploit individual data to find more useful and personally unique information or patterns which are hidden by group statistics and 95% confidence interval -- which we called **personalized medicine** or **precision medicine**.

My vision is fucusing individual, to broaden the possibilities and perspectives of big data, combining them with wearables.

Using iWatch Ultra, I collected all the ECG data from myself in September, 2023.

(1) According to my recent physical examination, I have a normal health condition, which means that I am able to raise my BG level significantly after consuming a large amount of sugar.

(2) BG level can affect the autonomic nervous system (ANS), which is involved in the regulation of heart rhythm.

Based on these facts, I assume that I could simulate two different blood level and record the ECG waves simultaneously, and try to find some relationships between BG level and ECG waves.

During a series of [OGTT](https://www.mayoclinic.org/tests-procedures/glucose-tolerance-test/about/pac-20394296)-like tests on myself. In the morning, I recorded 10-15 waves at fasting state. Then I drunk the glucose solution(200 milliliters of water containing 60 grams of glucose powder). One hour later, another 10-15 ECG samples were recorded. I typically recorded my ECG everyday or two, and eventually got a 13-day, 264 samples dataset. (Unfortunately, I gained 5kg weight. 🤣🤣🤣)

△In order to mitigate the affects of unrelevant factors, I drunk no coffee or tea, ate nothing, and made sure my mood and body position stayed in a similar state.
