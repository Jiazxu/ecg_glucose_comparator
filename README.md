---
## A blood glucose level binary classification deep neural network
---

### I. Dataset

Total 264 ECG waves

Label [fast]: low blood glucose level

Label [glocose]: high blood glucose level

<div align="center">
<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/dataset/example.png">

</div>

### II. Architecture

This model's backbone is [EfficientNetV2(Fused-MBConv)](https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py).

1. Classic supervised learning
   As a benchmark, I feeded the data into a classic EfficientNetV2.
   Here is the model architecture:

<div align="center">
<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/description/supervised_learning.png">

</div>

2. Comparator learning
   However, due to the scale of the dataset, the model could be very easy to be overfitted. Thus, I reconstructed the architecture and training workflow, which is more like Contrastive Learning.
   Here is the model architecture:

<div align="center">
<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/description/comparator_learning.png">

</div>

### III. Training and validating process

1. Classic supervised learning

2. Comparator learning

<div align="center">
<img src="https://github.com/Jiazxu/ecg_glucose_comparator/blob/master/checkpoint/effnetv2_ecg_comparator_v5_l_xxxs_20240111_epoch100_4e-3/effnetv2_ecg_comparator_v5_l_xxxs_20240111_epoch100_4e-3.png">

</div>

### IV. Some stories about this project

I collected all the ECG data from my iWatch Ultra in September, 2023.
<<<<<<< HEAD

During a series of [OGTT](https://www.mayoclinic.org/tests-procedures/glucose-tolerance-test/about/pac-20394296)-like tests on myself. I recorded dozens of waves before the test, and another dozens of waves after drinking the glucose solution(200 milliliters of a syrupy glucose solution containing 60 grams of glucose powder).

These ECG waves were classified into two labels:

Label [fast]: low blood glucose level

Label [glocose]: high blood glucose level
=======
I did a series of [OGTT](https://www.mayoclinic.org/tests-procedures/glucose-tolerance-test/about/pac-20394296)-like tests on myself. I recorded dozens of waves before the test, and another dozens of waves after drinking the glucose solution(200 milliliters of a syrupy glucose solution containing 60 grams of glucose powder).
These ECG waves were classified into two labels: fast and glucose
>>>>>>> c1ab0439e0ac16cabdee4c2c576c14baf37540c9
