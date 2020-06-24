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
- [ ] Debug confusion matrix plot generating 
- [ ] Debug the full dataset training bottleneck



