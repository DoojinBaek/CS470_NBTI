#!/bin/bash
set -e

EXPERIMENT="embedding_loss"
ABSTRACT="False"

LIST=("DOG D LuckiestGuy-Regular")

for wo in "${LIST[@]}"
do
  set -- $wo
  ARGS="--experiment $EXPERIMENT --optimized_letter $2 --seed 0 --font $3 --use_wandb 0 --wandb_user none --abstract ${ABSTRACT}"
  CUDA_VISIBLE_DEVICES=0 python code/main.py $ARGS --semantic_concept $1 --word $1 --memo "6.8" || continue
done