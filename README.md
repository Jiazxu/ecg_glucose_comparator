---
## A blood glucose level binary classification deep neural network
---

### Dataset

Total 264 ECG waves
Label [fast]: low blood glucose level
Label [glocose]: high blood glucose level

<div align="center">
<img src="https://github.com/Jiazxu/ecg_glucose_comparator/dataset/example.png" height="600px">

</div>

### Architecture

waiting...

### Training and test process

<div align="center">
<img src="https://github.com/Jiazxu/ecg_glucose_comparator/checkpoint/effnetv2_ecg_comparator_v5_l_xxxs_20240111_epoch100_4e-3/effnetv2_ecg_comparator_v5_l_xxxs_20240111_epoch100_4e-3.png" height="600px">

</div>

### Some stories about this project

I collected all the ECG data from my iWatch Ultra in September, 2023.
I did a series of [OGTT](https://www.mayoclinic.org/tests-procedures/glucose-tolerance-test/about/pac-20394296)-like tests on myself. I recorded dozens of waves before the test, and another dozens of waves after drinking the glucose solution(200 milliliters of a syrupy glucose solution containing 60 grams of glucose powder).
These ECG waves were classified into two labels: fast
