#!/bin/bash
conda activate word
set -e

USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="none"
EXPERIMENT="embedding_loss"

WORDLIST=("HARD")
FONTLIST=("DeliusUnicase-Regular")
ABSTRACT="True"

for wo in "${WORDLIST[@]}"
do
  letter=("H" "D")
  for l in "${letter[@]}"
  do
    for f in "${FONTLIST[@]}"
    do
      ARGS="--experiment $EXPERIMENT --optimized_letter ${l} --seed 0 --font ${f} --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER} --abstract ${ABSTRACT}"
      CUDA_VISIBLE_DEVICES=0 python code/main.py $ARGS --semantic_concept "${wo}" --word "${wo}" --memo "" || continue
    done
  done
done
