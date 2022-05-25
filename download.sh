#!/bin/bash

# inria-aid_short (Inria 10 images)
TO_PATH=./Data
wget https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/datasets/inria-aid_short.zip -O $TO_PATH/file.zip
unzip -o $TO_PATH/file.zip -d $TO_PATH
rm $TO_PATH/file.zip

# test-ds (UCMerced)
TO_PATH=./Data
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-sisr-use-case/datasets/ucmerced-test-ds-pre-executed.tar.gz -O $TO_PATH/ucmerced-test-ds-pre-executed.tar.gz
tar xvzf $TO_PATH/ucmerced-test-ds-pre-executed.tar.gz -C $TO_PATH
rm $TO_PATH/ucmerced-test-ds-pre-executed.tar.gz

