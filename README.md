# HandWriting Eraser

## Introduction

This repo aims to use deeplabv3+ to remove the handwriting on the papers.
I define the task of removing handwriting as a segmentation task. So we use a segmentation model named Deeplabv3+ to solve this task.

## Environment

```
Python 3.7
Pytorch 1.7+
```

```
pip install requirements.txt
```

## Datasets

We use one dataset supported by Baidu Aistdio at https://aistudio.baidu.com/aistudio/datasetdetail/121039 

## Optimized result

To optimize the segmentation result we use the tracks as follows:

1. Use the focal loss to replace the cross-entropy loss. This will help us solve the class imbalance problem in this task.
2. Use overlapping cropping to enhance the datasets.
3. Don't directly resize the image when predicting the result. Instead, I cut down the input image into many small images than predicting them separately.

## The Result of the model

Our model receives the 0.81 MAP in the test dataset.

### Some results

![image-20220425203938163](images/image-20220425203938163.png)

![image-20220425204005046](images/image-20220425204005046.png)


## References

[1] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[2] [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

[3] [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)

