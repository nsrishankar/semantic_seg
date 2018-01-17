# Semantic Segmentation

![alt_text](seg_out/semantic_simplified.gif)

Semantic segmentation using the KITTI dataset, and can be done using three methods:

## Methods
- **Full-Convolutional Network [2014]** Uses a pretrained network, upsample using deconvolution, and have skip connections to improve coarseness of upsampling.
- **SegNet [2015]** Encoder-Decoder architecture
- **ResNet-DUC [2017]** Dense Upsampling and Hybrid Dilated convolution

This was done using a pretrained VGG-16 model with skip-layers and deconvolutions to obtain a FCN and then trained for 50 epochs with an Adam Optimizer to minimize cross-entropy loss. As can be seen, the semantic annotation is very jagged (and in some cases with much different lighting conditions, the annotation overlaps with other classes). This can be slightly mitigated by image augmentation before feeding it into the network.

## Improvements
- To improve the quality of segmentation can use something resembling FlowNet with optical flow to improve the quality of segmentation wrt. ground truth. 
- Implement [in-progress] multi-class semantic segmentation using Citiscapes or Mapillary Vistas datasets using state-of-art.
