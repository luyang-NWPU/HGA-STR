#!/bin/bash

#source activate STR

GPU=2
    
CUDA_VISIBLE_DEVICES=${GPU} \
python main.py \
	--train_1 ~/workspace/Dataset/DataDB/SynthText \
	--train_2 ~/workspace/Dataset/DataDB/Synth90K \
	--test_1 ~/workspace/Dataset/DataDB/IIIT5K_testLmdb \
	--LR True \
	--batchSize 136 \
	--niter 4 \
	--lr 1 \
	--cuda \
	--displayInterval 2000 \
	--valInterval 10000 \
	--n_bm 5 \
	--val_start_epoch 2.0
    