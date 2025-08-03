<div align="center">
<h1>Moving Light Adaptive Colonoscopy Reconstruction via Improved Structure-compliant 3D Gaussian </h1>
</div>

## Overview

To address the challenge of dynamic illumination in 3D Gaussian Splatting (3DGS) for colonoscopy, where moving light sources degrade reconstruction quality, this work implements ***an improved structure-compliant 3DGS method (ColStrGS)***. We propose tethering Gaussians around colon surfaces via voxelized anchors and enforcing gravity constraints to stabilize their positions, eliminating structure-violating artifacts. Furthermore, ColStrGS adapts Gaussian appearances dynamically based on camera distance to mimic photometric variations. Experiments demonstrate the superiority of ColStrGS over other state-of-the-arts, particularly in reducing depth errors while maintaining high rendering quality.

<p align="center">
    <img src="assets/method.png"/ width=800> <br />
</p>

## Experimental Results

### 📊 Evaluation on C3VD
<p align="center">
    <img src="assets/experiment.png"/ width=800> <br />
</p>

### 📷 Visualization Comparisons
<p align="center">
    <img src="assets/comparision.png"/ width=800> <br />
</p>

### 👨‍⚕️ Validation on Clinical Data
To further evaluate the 3D reconstruction performance in clinical applications, we provide more results on a real-world benchmark called [**Colon10k**](https://ieeexplore.ieee.org/abstract/document/9433780). We compare both the rendering fidelity and geometric accuracy with another clinically-oriented method, [Gaussian Pancakes](https://github.com/smbonilla/GaussianPancakes).

As the initialization(RNNSLAM) used in Gaussian Pancakes is not accessible, we use the following strategy instead:

First, we generate temporal-consistent pseudo depth maps via [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) and run [OneSLAM](https://github.com/arcadelab/OneSLAM) with the depth to generate the pose and point cloud that match the depth scale. With the prepared pose and point cloud, we follow the paradigm in Gaussian Pancake and initialize the Gaussians for both GS methods.

<div style="text-align: center;">
    <figure>
        <img src="assets/merged.gif" width=500 alt="Visualization image"/>
        <figcaption>Our PSNR=25.36（left）  Gaussian Pancake PSNR=24.15(right)</figcaption>
    </figure>
</div>

<div style="text-align: center;">
    <figure>
        <img src="assets/merged_image.png" width=600 alt="Visualization image"/>
        <figcaption>Our Depth MSE=0.23（top）  Gaussian Pancake Depth MSE=1.14(down)</figcaption>
    </figure>
</div>


Qualitative comparisons show that the novel views rendered by Gaussian Pancakes exhibit some blurry artifacts, whereas our method achieves higher rendering fidelity. Additionally, the depth maps reconstructed by our approach are smoother and contain less noise.

## Implementations

Currently, we have released the core code, and the full version will be made available after the paper is accepted.

## Acknowledgement
Thanks the authors for their works:

[3DGS](https://github.com/graphdeco-inria/gaussian-splatting)

[EndoGaussian](https://github.com/CUHK-AIM-Group/EndoGaussian)

[Scaffold-GS](https://github.com/city-super/Scaffold-GS)

[OneSLAM](https://github.com/arcadelab/OneSLAM)

[Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
