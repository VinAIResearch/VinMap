# Installation guide
GPU or CPU model runner:

``` python
## GPU
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
## CPU
pip install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
```

``` python
git clone https://github.com/VinAIResearch/VinMap
cd VinMap

# Essential env
pip install mmcv-full
pip install mmdet==2.25.0
pip install mmcv==1.6.0

# Local Setup
cd mmocr
pip install -v -e .
cd ../

# Local Setup
cd vietocr
pip install -v -e .
cd ../
cd PaddleOCR
pip install -v -e .

# Extra env
pip install paddlepaddle
pip install mmrotate
pip install mmcls==0.25.0
pip install unidecode
pip install Pillow==9.5.0
pip install opencv-python==4.5.5.64
```

Downloading the weights and putting them inside weight directory: [directory](https://drive.google.com/drive/folders/1I-b_2dDkKWBEdhxMGF5FwhGIawq94BF1?usp=sharing)

- Text Segmentation

- Text Classification

- Text Recognition
