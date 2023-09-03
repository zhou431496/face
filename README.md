##FaceCLIP: CLIP-driven Accurate and Detailed 3D Face Reconstruction from a
Single Image  —— PyTorch implementation ##




## Installation
1. Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/zhou431496/faceclip.git
cd faceclip
conda env create -f environment.yml
source activate face
```

2. Install Nvdiffrast library:
```
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast    # ./Deep3DFaceRecon_pytorch/nvdiffrast
pip install .
apt-get install freeglut3-dev
apt-get install binutils-gold g++ cmake libglew-dev mesa-common-dev build-essential libglew1.5-dev libglm-dev
apt-get install mesa-utils
apt-get install libegl1-mesa-dev 
apt-get install libgles2-mesa-dev
apt-get install libnvidia-gl-525
If there is a "[F glutil.cpp:338] eglInitialize() failed" error, you can try to change all the "dr.RasterizeGLContext" in util/nv_diffrast.py into "dr.RasterizeCudaContext".
```

3. Install Arcface Pytorch:
```
cd ..    # ./faceclip
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/
```
4. Install CLIP
```
git clone https://github.com/openai/CLIP.git
```
5. Install ffl_loss
```
   pip install focal-frequency-loss   ####https://github.com/EndlessSora/focal-frequency-loss
```






## Contact
If you have any questions, please contact the paper authors.



Part of the code in this implementation takes [CUT](https://github.com/taesungp/contrastive-unpaired-translation) as a reference.

