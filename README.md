# Eye segmentation project

## A project collaboration with Med school to segment the eye nerve from eye CT scans

## Dependencies
| Package|
|:--:|
|Pytorch|
|Torchvision|
|pandas|
|scipy|
|scikit-learn|

## Data Pre-processing procedure
1. Get the binary mask from the .csv file (image size [496 x 1536]
```
python utils/get_mask.py
```
2. Split the training and testing sets manually by patient id (top 5480 images)
```
python utils/train_test_split.py
```
3. Pre-process the each image and binary mask into 512 x 512 by cutting into 3 pieces and mirror symmetric padding
```
python utils/pre_processing.py
```


## To-do list: 
- [x] make the README file explaining the project 
- [x] understand the type of data that I need in this project
- [x] make the helper function to read and load pictures into Pytorch dataset
- [x] make the helper function to read labels into Pytorch dataset
- [x] start finish the Unet model
- [x] Debug and run the training module
- [x] Work on GPU version of minimal training product
- [x] Add loss monitoring
- [x] Solve the environment issue
- [x] Add evaluation module
- [x] Split the dataset
- [x] Change the epoch based training into mini-batch based training
- [x] Debug the train test split and pre-processing
- [x] Try full datasets training
- [x] Change the time recorder from epoch base to batch-base
- [x] Add the plotting module during training and output plotting to the tensorboard
- [x] Debug the problem of output segmentation map is all 0 or 1
- [x] Debug confusion matrix plot generating 
- [x] Change the target mask from [1,0] structure to [0, 1] structure so the confusion plot better reflects the IoU values
- [x] Time profiling to find the training bottleneck issue
- [x] Check RAM issue if that is causing the program to slow down
- [x] Debug the full dataset training bottleneck
- [x] Full set IoU evaluation instead of single image ones
- [x] Evaluation master function
- [x] Evaluation plot function
- [x] Evaluation module which loads from the trained model
- [x] Debug the evaluation function
- [ ] Add post processing to the output result

## Time profiling
**Update on 2020.07.02: The time performance issue is resolved by re-installing gpu version of pytorch** 
### Time profile results
To address the training time consumption issue (takes about 30s to train on each image), a time profiling job is done both using the Cprofiler and the manual checkpoints added to the program. The Cprofiler outputs around 19M function calls, which is not very readable by human. Therefore the table of manual checkpoints are recorded below.

#### Time performance on training function level
| Operation | Approximate Time taken|
|:---------:|:---------------------:|
|enter epoch, set up metric holder| 0.012 s|
|set model to train state| 0.03s|
|take the grouped data point from train loader| 0.85s|
|get the image and binary mask from the grouped data point | 0.01s|
|put the img and mask on gpu| 0.01s|
|zero the gradient| 0.01 s|
|logit=model(img)| 12.2s|
|make loss| 0.1s|
|loss.backward()|20s|
|optm.step()|0.2s|
|adding training samples|0.02s|
|calculate IoU and adding losses to tensorboard|0.06s|
|change model to evaluation mode|0.03s|

#### Time performance on forward model level
| Operation(in channel, out channel, kernelsize, padding) | Approximate Time taken| Accumulate time|
|:---------:|:---------------------:|:------------------------:|
|model preparation| 0.47s| 0.47s|
|conv(3,64,3,1)+relu| 0.2s| 0.68s|
|conv(64,64,3,1)+relu| 1.5s|2.12s|
|First 3 layer of ResNet 18| 0.16s|2.28s|
|4-5 layer of ResNet 18| 0.4s|2.68s|
|6th layer of ResNet 18| 0.4s|3.02s|
|7th layer of ResNet 18| 0.16s|3.18s|
|8th layer of ResNet 18| 0.15s|3.33s|
|conv(512,512,1,0)+relu| 0.01s|3.34s|
|Upsample 2 times| 0.01s|3.35s|
|conv(256,256,1,0)+relu| 0.01s|3.36s|
|concatenate | 0.01s|3.37s|
|conv(256+512,512,3,1)+relu| 0.2s|3.57s|
|Upsample 2 times| 0.05s|3.62s|
|conv(128,128,1,0)+relu| 0.01s|3.63s|
|concatenate | 0.01s|3.64s|
|conv(128+512,256,3,1)+relu| 0.4s|4.02s|
|Upsample 2 times| 0.08s|4.10s|
|conv(64,64,1,0)+relu| 0.01s|4.11s|
|concatenate | 0.02s|4.13s|
|conv(64+256,256,3,1)+relu| 0.76s|4.89s|
|Upsample 2 times| 0.65s|5.54s|
|conv(64,64,1,0)+relu| 0.05s|5.59s|
|concatenate | 0.05s|5.56s|
|conv(64+256,128,3,1)+relu| 1.8s|7.53s|
|Upsample 2 times| 0.7s| 8.21s|
|concatenate|0.12s|8.33s|
|conv(64+128, 64, 3, 1) +relu|4.4s|12.76s|
|conv(64,2,1,0)|0.06s|12.82s|




|



