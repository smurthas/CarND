#!/usr/bin/env bash

# trains a model on a remote host and downloads results

set -e

host_dir=/home/carnd/carnd

host=$1
model_py=model.py

nb_epoch=$2
out_file=$3
train_set=raw_data_sets/$4

scp $model_py $host:$host_dir/$model_py
ssh $host "export PATH=\"/home/carnd/anaconda3/bin:\$PATH\" && source activate carnd-term1 && cd $host_dir && python $model_py $nb_epoch $out_file $train_set"
scp $host:$host_dir/$out_file.h5 $out_file.h5

