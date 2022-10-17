# IQUAFLOW - QMRNet EO Dataset Evaluation

Note: Use our [jupyter notebook](IQF-UseCase-EO.ipynb) to run the use case.

- The rest of code is distributed in distinct repos [IQUAFLOW framework](https://github.com/satellogic/iquaflow), [QMRNet's Super-Resolution Use case](https://github.com/dberga/iquaflow-qmr-sisr) and [QMRNet's Loss for SR](https://github.com/dberga/iquaflow-qmr-loss).

# Regressor Metrics

Regressor metrics as a tool to measure quality on image datasets:

 - *rer* is a measure of the edge response ( mesures the degree of the transition ) which also informs on image sharpness.
 - *snr* - Signal to noise (gaussian) ratio.
 - *sigma* - of a gaussian distribution. It measures blur by defining its kernel.
 - *sharpness* - Edge response (lower is blurred, higher is oversharpened)
 - *scale* - resolution proportion scale (x2 from 0.30 is 0.15 m/px)

## Instrucctions:

1. Build the python environment. This can be done with a docker or conda (see sections below)
2. Prepare your dataset with a the file "annotations.json" and a folder with the images. The annotations file can be empty.
3. Run:
```
usage: python iqf_use_case_qmr.py [-h] [--imf IMF]

optional arguments:
  -h, --help  show this help message and exit
  --imf IMF   Images folder of the dataset. Its parent directory contains the
              annotations.json file
```

## Environment with docker
```
make build
make dockershell
python IQF-UseCase-EO.py
```
## Environment with conda

```
# To run the example dataset, download it:
./download.sh

conda create -n satellogic python=3.6  -q -y

# install acording to your GPU system
conda run -n satellogic pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

pip install git+https://YOUR_GIT_TOKEN@github.com/satellogic/iquaflow.git

conda run -n satellogic pip install piq tqdm tensorboardX && \
conda run -n satellogic pip install imageio scikit-image && \
conda run -n satellogic pip install rasterio==1.2.6

conda activate satellogic

python IQF-UseCase-EO.py

```
Note: make sure to replace "YOUR_GIT_TOKEN" to your github access token, also in [Dockerfile](Dockerfile).

# Cite

If you use content of this repo, please cite:

```
@article{berga2022,
  title={QMRNet: Quality Metric Regression for EO Image Quality Assessment and Super-Resolution},
  author={Berga, David and Gallés, Pau and Takáts, Katalin and Mohedano, Eva and Riordan-Chen, Laura and Garcia-Moll, Clara and Vilaseca, David and Marín, Javier},
  journal={arXiv preprint arXiv:2210.06618},
  year={2022}
}
@article{galles2022,
  title={IQUAFLOW: A NEW FRAMEWORK TO MEASURE IMAGE QUALITY},
  author={Gallés, Pau and Takáts, Katalin and Hernández-Cabronero, Miguel and Berga, David and Pega, Luciano and Riordan-Chen, Laura and Garcia-Moll, Clara and Becker, Guillermo and Garriga, Adán and Bukva, Anica and Serra-Sagristà, Joan and Vilaseca, David and Marín, Javier},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2022}
}
```
