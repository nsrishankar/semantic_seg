# Semantic Segmentation

Semantic segmentation using the KITTI dataset, and can be done using three methods:

## Methods
- **Full-Convolutional Network [2014]** Uses a pretrained network, upsample using deconvolution, and have skip connections to improve coarseness of upsampling.
- **SegNet [2015]** Encoder-Decoder architecture
- **ResNet-DUC [2017]** Dense Upsampling and Hybrid Dilated convolution

To improve the quality of segmentation can use something resembling FlowNet with optical flow to improve the quality of segmentation wrt. ground truth.
