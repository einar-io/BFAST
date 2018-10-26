#!/usr/bin/env bash
set -e -x

DATA_DIR=../../data
futhark-opencl bfast-kernels.fut

function gen_dataset() {
  gunzip -c $DATA_DIR/sahara.in.gz \
    | ./bfast-kernels --binary-output -e bfast_$1_out \
    | gzip > $DATA_DIR/sahara-$1.out.gz
}

gunzip -c $DATA_DIR/sahara.in.gz \
  | ./bfast-kernels --binary-output -e bfast_inputs \
  | gzip > $DATA_DIR/sahara-all.in.gz

gen_dataset 1
gen_dataset 2
gen_dataset 3
gen_dataset 4a
gen_dataset 4b
gen_dataset 4c
gen_dataset 5
gen_dataset 6
gen_dataset 7a
gen_dataset 7b
gen_dataset 8


