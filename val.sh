#!/bin/bash

#source activate STR

GPU=2
    
CUDA_VISIBLE_DEVICES=${GPU} \
python val.py \
	--test_1 ~/workspace/Dataset/DataDB/IIIT5K_testLmdb \
	--LR True \
	--cuda \
	--n_bm 5 \
	--MODEL output/latest.pth
    