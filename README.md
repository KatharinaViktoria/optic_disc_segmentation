# optic_disc_segmentation
Segmentation uncertainty under domain-shift

Optic disc segmentation models trained on non-glaucoma cases loss: binary cross entropy/Dice models: vanilla 2D UNet, 2D UNet ensemble (N=10), 2D MC-UNet
Dataset: REFUGE-2 Environment and packages:
- Docker image: deepo jupyter image
- MONAI version 0.2.0 (old but it's working) - the code will surely break in another MONAI version (most likely due to the data loader and/or transformations0
preprocessing:
- images resized to 1634x1634x3
- ground truth: instead of the original two label optic disc/optic cup segmentation, the models are trained for a binary segmentation task: optic disc+optic cup vs background
More details: https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12032/1203211/Do-I-know-this-segmentation-uncertainty-under-domain-shift/10.1117/12.2611867.short?SSO=1
