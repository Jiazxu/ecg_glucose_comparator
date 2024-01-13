---
# A blood glucose level binary classification deep neural network
---

## I. Dataset

Total 264 ECG samples (divided into 8,543 QRS waves) recorded in 13 days.

I splitted the data randomly **by date** into three dataset: training dataset, validation dataset and test dataset. The splitting ratio is 60:20:20.
△ If all samples were mixed and splitted randomly, it could deteriorate the model performance in the real world. Because the waves recorded in the same date are similar.

These ECG waves were classified into two labels:

Label [fast/f]: low blood glucose (BG) level, recorded in fasting condition.

Label [glocose/g]: high blood glucose (BG) level, recorded after consuming a large amount of sugar.

<div align="center">
<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/dataset/example.png">

</div>

## II. Model Architecture

This model's backbone is [EfficientNetV2(Fused-MBConv)](https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py).

**1. Classic supervised learning**

As a benchmark, I feeded the data into a classic EfficientNetV2.
Here is the model architecture:

<div align="center">
<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/description/supervised_learning.png">

</div>

**2. Comparator learning**

However, due to the scale of the dataset, the training process was overfitted really easily, even though the model was so small (no more than 9,000 parameters!).

Thus, I reconstructed the architecture and training workflow. From a mathematical perspective, by **comparing the similarity of two QRS waves randomly**, the training data increases exponentially and could alleviate the overfitting.

After finishing this project, I found this training style looks like Contrastive Learning lol.

Here is my model:

<div align="center">
<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/description/comparator_learning.png">

</div>

## III. Training and validating process

1. Classic supervised learning

<div align="center">
<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/checkpoint/effnetv2_ecg_l_xxxs_20240111_epoch150_1e-4/effnetv2_ecg_l_xxxs_20240111_epoch150_1e-4.png">

</div>

2. Comparator learning

<div align="center">
<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/checkpoint/effnetv2_ecg_comparator_v5_l_xxxs_20240111_epoch100_4e-3/effnetv2_ecg_comparator_v5_l_xxxs_20240111_epoch100_4e-3.png">

</div>

## IV. About this project

I have been curious about the application of big data on healthcare for a long time, since I was a undergraduate medicine student.

Human body is a complex system, creating high dimensional data all the time. For decades, Scientists are trying to analysis the manifold from these complicated data and its projections on different dimensions and levels, and related them with all kinds of phenomena, such as diseases, mental conditions, pathological changes and results from labs and medical equipments. On the other hand, it is also a really exciting road to discover new data source and new methdology to manage human beings heathcare.

My vision is to broaden the possibilities and perspectives of big data, combining them with wearables.

I collected all the ECG data from my iWatch Ultra in September, 2023.

(1) According to my recent physical examination, I have a normal health condition, which means that I am able to raise my BG level significantly after consuming a large amount of sugar.
(2) BG level can affect the autonomic nervous system (ANS), which is involved in heart rhythm.

Based on these facts, I assume that I could simulate two different blood level and record the ECG waves simultaneously, and try to find some relationships between BG level and ECG waves.

During a series of [OGTT](https://www.mayoclinic.org/tests-procedures/glucose-tolerance-test/about/pac-20394296)-like tests on myself. I recorded dozens of waves before the test in the morning, at fasting state. One hour later after drinking the syrupy glucose solution(200 milliliters of water containing 60 grams of glucose powder), I recorded another dozens of ECG samples.

In order to mitigate the affects of unrelevant factors, I drunk no coffee or tea, ate nothing, and made sure my mood and body position stayed in a similar state.
