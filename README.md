# remote_sensing_solar_pv
A repository for sharing progress on the automated detection of solar PV arrays in sentinel-2 remote sensing imagery.

## Problem
Environmental risk, both transition and physical risk, threatens a smooth and orderly transition to a sustainable economy. Insufficient asset-level data exists to adequately measure environmental risk exposure in the real economy. The diffusion of solar PV is a fundamental transition risk to many large incumbent utilities and electricity systems. Detecting and localising solar PV generating stations in remote sensing imagery might allow both public and private sector organisations to better understand the ongoing energy transition and prepare appropriately resilient business models and physical infrastructures.

## Training Data
An initial dataset of solar PV installations is obtained from the World Resources Institute [Resource Watch](https://resourcewatch.org/data/explore/a86d906d-9862-4783-9e30-cdb68cd808b8). Polygons are hand-labelled using [Google Earth Engine](https://earthengine.google.com/) and rasterised into class labels using a variety of geospatial libraries. Remote sensign data is obtained from the Sentinel-2 mission - the highest-resolution publicly available imagery with global coverage. The full band spectrum from the Sentinel-2 mission is obtained, see band details below. Data are accessed using [DescartesLabs](http://www.descarteslabs.com/) API. 

**Table 1: Sentinel-2 Bands**

| Band        | Central Wavelength [nm] | Spatial Resolution [m] |
| ------------- |:-------------:| -----:|
| Band 1 - Aerosol      | 443.9 | 60 |
| Band 2 - Blue      | 496.6      |   10 |
| Band 3 - Green      | 560.0      |   10 |
| Band 4 - Red      | 664.5      |   10 |
| Band 5 - Red Edge 1      | 703.9      |   20 |
| Band 6 - Red Edge 2      | 740.2      |   20 |
| Band 7 - Red Edge 3      | 782.5      |   20 |
| Band 8 - NIR      | 835.1      |   10 |
| Band 8A - Narror NIR      | 864.8      |   20 |
| Band 9 - Water Vapour      | 945.0      |   60 |
| Band 10 - SWIR-Cirrus      | 1373.5      |   60 |
| Band 11 - SWIR      | 1613.7      |   20 |
| Band 12 - SWIR      | 2202.4      |   20 |


An initial training set of approximately 500 sample 2kmx2km rasters are developed for testing computer vision models. With 14 multispectral bands and a minimum resolution 10m, Input datacubes thus have dimension (bs,200,200,14) where bs is the *training batch size*.

## Semantic Segmentation Models
In the past several years, research into fully convolutional neural networks has yielded some specialist architectures for the classification of images where the number of classes is small, the properties of the image are known (such as fixed subject distance, or single-band sensor), and the available training data is small. These architectures are suited to problems like medical imagery and remote sensing analysis, rather than general object recognition among thousands of object classes trained on corpuses of millions of images. Four architectures are tested for their ability to classify solar PV arrays in sentinel-2 imagery. [SegNet]((https://arxiv.org/pdf/1511.00561.pdf) (Badrinarayanan, Kendall, & Cipolla 2016) is one of the original encoder-decoder architectures used for semantic segmentation. [UNet](https://arxiv.org/abs/1505.04597) (Ronneberger, Fischer, & Brox 2015) is a popular encoder-decoder architecture which forwards low-orders features to deconvolutional layers. [ResUNet](https://arxiv.org/pdf/1711.10684.pdf) (Zhang & Liu 2017) adds residual blocks to the UNet architecture to improve provide information bypass. A deeper and narrower [ResUNet](https://arxiv.org/abs/1709.00201) (Li et al. 2017) adds further encoding and decoding blocks at the expense of intermediate features. Table 2 shows some parameters for the trained networks.

## Results

### Training Efficiency
All four architectures were trained for 400 epochs on the training set, with 18% of the data retained for validation.

**Model Training - Segnet**

![alt text](https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/model_segnet.png "Segnet Training")


**Model Training - UNet**

![alt text](https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/model_unet.png "UNet Training")


**Model Training - ResUNet**

![alt text](https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/model_resunet.png "ResUNet Training")


**Model Training - Deep ResUNet**

![alt text](https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/model_deep_resunet.png "Deep ResUNet Training")


### Jaccard Index (IoU)
A common metric for semantic segmentation problems which captures both precision and recall performance of a trained model is the Jaccard Index - commonly the Intersection-over-Union. Jaccard Indices for the selected architectures are also shown in Table 2.

**Table 2: Network Architectures**

| Architecture         |    SegNet     |  U-Net | ResUNet | DeepResUNet |
| -------------------- |:-------------:| :-----:| :-----: | :----------:|
| Training Epochs      | 400           | 400    | 400     | 400         |
| Trainable Parameters | 32M           | 32M    | 32M     | 5M          |
| Batch Size           | 32            | 16     | 12      | 16          |
| IoU                  | 86%           | 94%    |  98%    | 80%         |

### Selected Samples
A number of illustrative samples are shown below. 

| RGB  | DeePresUNet | SegNet | U-Net | ResUNet | Ground Truth | 
| ---- | ----   | ----  | ----    | ------------| -------------|
| ![at][22_rgb] | ![at][22_dru] | ![at][22_seg] | ![at][22_unet] | ![at][22_runet] | ![at][22_truth] |
| ![at][27_rgb] | ![at][27_dru] | ![at][27_seg] | ![at][27_unet] | ![at][27_runet] | ![at][27_truth] |
| ![at][21_rgb] | ![at][21_dru] | ![at][21_seg] | ![at][21_unet] | ![at][21_runet] | ![at][21_truth] |
| ![at][34_rgb] | ![at][34_dru] | ![at][34_seg] | ![at][34_unet] | ![at][34_runet] | ![at][34_truth] |


[22_truth]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_resunet_22_TRUE.png "Ground Truth"

[27_truth]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_resunet_27_TRUE.png "Ground Truth"

[21_truth]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_resunet_21_TRUE.png "Ground Truth"

[34_truth]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_resunet_34_TRUE.png "Ground Truth"

[22_runet]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_resunet_22_PRED.png "Ground Truth"
[27_runet]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_resunet_27_PRED.png "Ground Truth"

[21_runet]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_resunet_21_PRED.png "Ground Truth"

[34_runet]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_resunet_34_PRED.png "Ground Truth"

[22_dru]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_deep_resunet_22_PRED.png "Ground Truth"

[27_dru]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_deep_resunet_27_PRED.png "Ground Truth"

[21_dru]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_deep_resunet_21_PRED.png "Ground Truth"

[34_dru]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_deep_resunet_34_PRED.png "Ground Truth"

[22_rgb]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_resunet_22_INPUT.png "Ground Truth"

[27_rgb]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_resunet_27_INPUT.png "Ground Truth"

[21_rgb]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_resunet_21_INPUT.png "Ground Truth"

[34_rgb]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_resunet_34_INPUT.png "Ground Truth"

[22_seg]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_segnet_22_PRED.png "Ground Truth"

[27_seg]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_segnet_27_PRED.png "Ground Truth"

[21_seg]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_segnet_21_PRED.png "Ground Truth"

[34_seg]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_segnet_34_PRED.png "Ground Truth"

[22_unet]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_unet_22_PRED.png "Ground Truth"

[27_unet]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_unet_27_PRED.png "Ground Truth"

[21_unet]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_unet_21_PRED.png "Ground Truth"

[34_unet]: https://github.com/Lkruitwagen/remote_sensing_solar_pv/blob/master/sample_out/model_unet_34_PRED.png "Ground Truth"
