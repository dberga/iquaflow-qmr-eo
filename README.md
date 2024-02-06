![](https://github.com/satellogic/iquaflow/blob/main/docs/source/iquaflow_logo_mini.png)
Check [QMRNet's Remote Sensing article](https://www.mdpi.com/2072-4292/15/9/2451) and [iquaflow's JSTARS article](https://ieeexplore.ieee.org/abstract/document/10356628) for further documentation. You also can [install iquaflow with pip](https://pypi.org/project/iquaflow/) and look at the [iquaflow's wiki](https://iquaflow.readthedocs.io/en/latest/). 


# IQUAFLOW - QMRNet EO Dataset Evaluation

- Note: Use our [jupyter notebook](IQF-UseCase-EO.ipynb) to run the use case.

- The rest of code is distributed in distinct repos [iquaflow framework](https://github.com/satellogic/iquaflow), [QMRNet's Super-Resolution Use case](https://github.com/dberga/iquaflow-qmr-sisr), [QMRNet's Loss for SR](https://github.com/dberga/iquaflow-qmr-loss) and [QMRNet standalone code](https://github.com/satellogic/iquaflow/tree/main/iquaflow/quality_metrics).

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
usage: python IQF-UseCase-EO.py [-h] [--imf IMF]

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

# Design and Train the QMRNet (regressor.py)

In [QMRNet standalone code](https://github.com/satellogic/iquaflow/tree/main/iquaflow/quality_metrics) you can find several scripts for training and testing the QMRNet, mainly integrated in `regressor.py`. Using `run_spec.sh` you can specify any of the `cfgs\` folder where the architecture design and hyperparameters are defined. You can create new versions by adding new `.cfg` files.

# Cite

If you use content of this repo, please cite:

```
@article{berga2023,
AUTHOR = {Berga, David and Gallés, Pau and Takáts, Katalin and Mohedano, Eva and Riordan-Chen, Laura and Garcia-Moll, Clara and Vilaseca, David and Marín, Javier},
TITLE = {QMRNet: Quality Metric Regression for EO Image Quality Assessment and Super-Resolution},
JOURNAL = {Remote Sensing},
VOLUME = {15},
YEAR = {2023},
NUMBER = {9},
ARTICLE-NUMBER = {2451},
URL = {https://www.mdpi.com/2072-4292/15/9/2451},
ISSN = {2072-4292},
ABSTRACT = {The latest advances in super-resolution have been tested with general-purpose images such as faces, landscapes and objects, but mainly unused for the task of super-resolving earth observation images. In this research paper, we benchmark state-of-the-art SR algorithms for distinct EO datasets using both full-reference and no-reference image quality assessment metrics. We also propose a novel Quality Metric Regression Network (QMRNet) that is able to predict the quality (as a no-reference metric) by training on any property of the image (e.g., its resolution, its distortions, etc.) and also able to optimize SR algorithms for a specific metric objective. This work is part of the implementation of the framework IQUAFLOW, which has been developed for the evaluation of image quality and the detection and classification of objects as well as image compression in EO use cases. We integrated our experimentation and tested our QMRNet algorithm on predicting features such as blur, sharpness, snr, rer and ground sampling distance and obtained validation medRs below 1.0 (out of N = 50) and recall rates above 95%. The overall benchmark shows promising results for LIIF, CAR and MSRN and also the potential use of QMRNet as a loss for optimizing SR predictions. Due to its simplicity, QMRNet could also be used for other use cases and image domains, as its architecture and data processing is fully scalable.},
DOI = {10.3390/rs15092451}
}
@article{galles2024,
  title = {A New Framework for Evaluating Image Quality Including Deep Learning Task Performances as a Proxy},
  volume = {17},
  ISSN = {2151-1535},
  url = {https://ieeexplore.ieee.org/document/10356628},
  DOI = {10.1109/jstars.2023.3342475},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  publisher = {Institute of Electrical and Electronics Engineers (IEEE)},
  author = {Gallés,  Pau and Takáts,  Katalin and Hernández-Cabronero,  Miguel and Berga,  David and Pega,  Luciano and Riordan-Chen,  Laura and Garcia,  Clara and Becker,  Guillermo and Garriga,  Adan and Bukva,  Anica and Serra-Sagristà,  Joan and Vilaseca,  David and Marín,  Javier},
  year = {2024},
  pages = {3285–3296}
}
```
