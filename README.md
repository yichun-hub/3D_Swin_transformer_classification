# 3D_Swin_transformer_classification
## Introduction
The model is modified by [Swin_UNTER](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV). The model has been augmented with a classification head, which can classify the type you want.
Noticed that the model is for 3D medical image.

## Installation and dependencies
1. Clone the repository
2. Install required packages

## Training
Depending on the classification task — binary or two-class(multiple classes) — different loss functions and evaluation metrics are used:

For two-class(multiple classes) classification model is defined as below:

```
model = SwinTransformerForClassification(
    img_size=(64,64,64),
    num_classes = 2,
    in_channels=1,
    out_channels=786, 
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
)
```
The above model is used for CT images (1-channel input) with input image size `(64, 64, 64)` and for `2` classes for the classification output and feature size of `48`. `out_channels` means the number of output channels from the feature map.
* **Loss function:**  `CrossEntropyLoss()`.
This loss is used for multi-class classification problems (in this case, two classes). It is combined with a softmax activation function, which transforms the output into a probability distribution.
* **Evaluation Metric:**
The predicted class is the one with the highest probability, determined by finding the maximum value in the predictions.

For binary classification model is defined as below:
```
model = SwinTransformerForClassification(
    img_size=(64,64,64),
    num_classes = 1,
    in_channels=1,
    out_channels=786, 
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
)
```
The above model is used for CT images (1-channel input) with input image size `(64, 64, 64)`. Noticed that the class (`num_classes`) will be set as `1` because it predicts the positive or not.
* **Loss function:**  `BCEWithLogitsLoss()`.
This loss function combines a sigmoid activation with binary cross-entropy loss, outputting a single probability score.
* **Evaluation Metric:**
A prediction value greater than 0.5 is classified as positive (commonly labeled as 1), and less than 0.5 as negative (commonly labeled as 0).

## Usage
```
python main.py --batch_size=<batch_size> --logdir=<model_name> --optim_lr=1e-4 --lrschedule=warmup_cosine --roi_x=64 --roi_y=64 --roi_z=64 --val_every 1 --save_checkpoint
```
