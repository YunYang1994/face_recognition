#! /bin/bash

#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#   
#   Editor      : VIM 
#   File name   : convert_to_ir.sh
#   Author      : YunYang1994
#   Created date: 2020-02-28 16:06:54
#   Description : 
#
#================================================================

python3 -m mmdnn.conversion._script.convertToIR -f mxnet -n model-symbol.json -w model-0000.params --inputShape 3,112,112 -o mobilefacenet
python3 -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath mobilefacenet.pb --IRWeightPath mobilefacenet.npy --dstModelPath tf_mobilefacenet.py 
