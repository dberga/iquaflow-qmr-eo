# Regressor Metrics

The regressor metrics is a tool to measure the following metrics on image datasets:

rer - is a measure of the edge response (from the blurred edge) which also informs on image sharpness 
snr - Signal to noise ratio.
sigma - of a gaussian distribution. It measure blur by defining its kernel.
sharpness - Edge response
scale - resolution with respect to 1meter/pixel

## Instrucctions:

1. Build the python environment. This can be done with a docker or conda (see sections below)
2. Prepare your dataset with a the file "annotations.json" and a folder with the images. The annotations file can be empty.
3. Run:
```
usage: python iqf_use_case_metrics.py [-h] [--imf IMF]

optional arguments:
  -h, --help  show this help message and exit
  --imf IMF   Images folder of the dataset. Its parent directory contains the
              annotations.json file
```

## Environment with docker

make build
make dockershell
python iqf_use_case_metrics.py

## Environment with conda

```
# To run the example dataset, download it:
./download.sh

conda create -n satellogic python=3.6  -q -y
    
git clone https://gitlab+deploy-token-28:xkxRsx2anp-u3_V4aAK9@publicgitlab.satellogic.com/iqf/iq_tool_box- && \
cd iq_tool_box- && git checkout regressor-rebase-b && \
conda run -n satellogic pip install -e . && \
cd ..

# install acording to your GPU system
conda run -n satellogic pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

conda run -n satellogic pip install piq tqdm tensorboardX && \
conda run -n satellogic pip install imageio scikit-image && \
conda run -n satellogic pip install rasterio==1.2.6  && \
conda run -n satellogic pip install kornia --no-deps

conda activate satellogic

python iqf_use_case_metrics.py

```