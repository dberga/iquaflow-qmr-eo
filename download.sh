#!/bin/bash
if [ -z $1 ] then
TO_PATH=./Data
else
TO_PATH=$1
fi

# inria-aid_short (Inria AILD, 10 images) [30 cm/px][5000x5000]
wget https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/datasets/inria-aid_short.zip -O $TO_PATH/file.zip
unzip -o $TO_PATH/file.zip -d $TO_PATH
mv $TO_PATH/test_datasets/inria-aid_short $TO_PATH
rm -rf $TO_PATH/test_datasets
rm -rf $TO_PATH/file.zip

# test-ds (UCMerced Test, 380 images) [30 cm/px][232x232]
TO_PATH=./Data
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-sisr-use-case/datasets/ucmerced-test-ds-pre-executed.tar.gz -O $TO_PATH
tar xvzf $TO_PATH/ucmerced-test-ds-pre-executed.tar.gz -C $TO_PATH
rm $TO_PATH/ucmerced-test-ds-pre-executed.tar.gz

# DeepGlobe (DeepGlobe, 469 images) [50 cm/px][2448x2448]
wget https://public-remote-sensing-datasets.s3-eu-west-1.amazonaws.com/compressed/DeepGlobe-50cm.tar.gz $TO_PATH
cd $TO_PATH
tar -xvf $TO_PATH/DeepGlobe-50cm.tar.gz -C $TO_PATH
mkdir $TO_PATH/DeepGlobe
mkdir $TO_PATH/images
mv $TO_PATH/scratch/pau/SATE00_SUPE00/DATA_SOURCES/DeepGlobe/* $TO_PATH/DeepGlobe/images
rm -rf $TO_PATH/scratch
rm -rf $TO_PATH/DeepGlobe-50cm.tar.gz

# AerialImageDataset (Intria AILD, train 180+ val 180 images) [30 cm/px][5000x5000]
wget https://public-remote-sensing-datasets.s3-eu-west-1.amazonaws.com/compressed/Inria-NEW2-AerialImageDataset.zip $TO_PATH
unzip -d $TO_PATH $TO_PATH/Inria-NEW2-AerialImageDataset.zip
rm -rf $TO_PATH/Inria-NEW2-AerialImageDataset.zip

# shipsnet (kaggle ships, 4000 ship images [80x80], 7 scenes [~2800x1600]) [3 m/px]
wget https://public-remote-sensing-datasets.s3-eu-west-1.amazonaws.com/compressed/kaggle-ships-in-satellite-imagery-3meter-pixel-size.zip $TO_PATH
unzip -d $TO_PATH $TO_PATH/kaggle-ships-in-satellite-imagery-3meter-pixel-size.zip
mv $TO_PATH/shipsnet.json $TO_PATH/shipsnet
mv $TO_PATH/scenes $TO_PATH/shipsnet
rm -rf $TO_PATH/kaggle-ships-in-satellite-imagery-3meter-pixel-size.zip

# UCMerced_LandUse (UCMerced, 2100 images) [30 cm/px; 1 foot][256x256]
wget https://public-remote-sensing-datasets.s3-eu-west-1.amazonaws.com/compressed/UCMerced_LandUse_30cm.zip $TO_PATH
unzip -d $TO_PATH $TO_PATH/UCMerced_LandUse_30cm.zip
cd $TO_PATH/UCMerced_LandUse/Images
mkdir ALL_CATEGORIES
cd ALL_CATEGORIES
for file in ../*/*; do
    ln -s $file .
done
rm -rf $TO_PATH/UCMerced_LandUse_30cm.zip
cd ../../../..

# USGS (USGS, 279 images) [ ? 30 cm/px][~5000x5000]
wget https://public-remote-sensing-datasets.s3-eu-west-1.amazonaws.com/compressed/USGS.tgz $TO_PATH
tar -xvf $TO_PATH/USGS.tgz -C $TO_PATH
mkdir $TO_PATH/USGS
mv $TO_PATH/scratch/pau/SATE00_SUPE00/DATA_SOURCES/hr_images $TO_PATH/USGS
rm -rf $TO_PATH/USGS.tgz
rm -rf $TO_PATH/scratch

# XView [30 cm/px][]
wget https://public-remote-sensing-datasets.s3-eu-west-1.amazonaws.com/compressed/xview_train_images.tgz $TO_PATH
tar -xvf $TO_PATH/xview_train_images.tgz -C $TO_PATH
mkdir $TO_PATH/XView
mv $TO_PATH/train_images $TO_PATH/XView
rm -rf $TO_PATH/xview_train_images.tgz
wget https://public-remote-sensing-datasets.s3-eu-west-1.amazonaws.com/compressed/xview_val_images.tgz $TO_PATH
tar -xvf $TO_PATH/xview_val_images.tgz -C $TO_PATH
mv $TO_PATH/val_images $TO_PATH/XView
rm -rf $TO_PATH/xview_val_images.tgz

# ECODSE [Hyperspectral]
# wget https://public-remote-sensing-datasets.s3-eu-west-1.amazonaws.com/compressed/ECODSEdataset.zip $TO_PATH


