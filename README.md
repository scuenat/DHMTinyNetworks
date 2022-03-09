# DHM Tiny Networks
This repository contains the training scripts to train a tiny model as proposed in this paper, ARXIV LINK.

Three neural networks are proposed:

- TVGG 
- TViT
- Or TSwinT

# How to run a training script
The training script takes several parameters in consideration.
- **dataset**: path to the dataset like the one proposed at this address (experimental holograms).
- **batch_size**: number of holograms considered at once
- **num_roi**: number of ROIs randomly extracted of the size specified in the settings file. 
- **model_type**: either "tswint", "tvit" or "tvgg". 

**Please note**: that the batch size is multiplied by the "num_roi". This gives the number of inputs a neural network will process at once. 

As an example:
`python3 train.py --dataset="c:/temp/test_dataset" --batch_size=2 --num_roi=4 --model_type="tswint"`


# Outputs 
The training script is copying the generated model into the folder "workspace" (which is automatically created if it does not exist).
The name of the model is constructed as follows: {model_type}.h5. (e.g.: tvit.h5).
A test file (CSV) is also generated based on the corresponding settings (TEST_SPLIT). This file contains a list of images not seen during training. This set must be used to evaluate the performance of the trained model (e.g: test_images_128_tvgg.csv).

# General information

All tiny networks are based on original versions. We essentially modified the size of the layers or the number of layers.
TVGG is based on a VGG16 (all the kernel filter have been divided by 2). TViT is based on the code proposed by vit-keras (https://github.com/faustomorales/vit-keras). 
TSwinT is based on the following code: https://github.com/rishigami/Swin-Transformer-TF. We have extended the code only by adding a new configuration for the TSwinT architecture. 

| Model | Number of parameters |
| --- | --- |
| TVGG| 3M |
| TViT| 4M |
| TSwinT| 2.7M |