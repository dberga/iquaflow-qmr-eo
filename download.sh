#!/bin/bash

TO_PATH=./tests
wget https://image-quality-framework.s3.eu-west-1.amazonaws.com/iq-tool-box/datasets/inria-aid_short.zip -O $TO_PATH/file.zip
unzip -o $TO_PATH/file.zip -d $TO_PATH
rm $TO_PATH/file.zip