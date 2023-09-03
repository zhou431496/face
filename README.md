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
   pip install focal-frequency-loss //https://github.com/EndlessSora/focal-frequency-loss
   ```
## Inference with a pre-trained model

### Prepare prerequisite models
1. Our method uses [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) to represent 3d faces. Get access to BFM09 using this [link](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads). After getting the access, download "01_MorphableModel.mat". In addition, we use an Expression Basis provided by [Guo et al.](https://github.com/Juyong/3DFace). Download the Expression Basis (Exp_Pca.bin) using this [link (google drive)](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view?usp=sharing). Organize all files into the following structure:


2. We provide a model trained on a combination of [VGGFace2]and [FFHQ](https://github.com/NVlabs/ffhq-dataset) datasets.:






## Contact
If you have any questions, please contact the paper authors.



Part of the code in this implementation takes [CUT](https://github.com/taesungp/contrastive-unpaired-translation) as a reference.

