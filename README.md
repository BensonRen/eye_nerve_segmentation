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

## Time profile results
To address the training time consumption issue (takes about 30s to train on each image), a time profiling job is done both using the Cprofiler and the manual checkpoints added to the program. The Cprofiler outputs around 19M function calls, which is not very readable by human. Therefore the table of manual checkpoints are recorded below.

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
- [ ] Check RAM issue if that is causing the program to slow down
- [ ] Debug the full dataset training bottleneck



