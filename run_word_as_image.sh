#!/bin/bash

set -e

USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="none"
EXPERIMENT="embedding_loss"

WORDLIST=("NBTI")
FONTLIST=("DeliusUnicase-Regular" "HobeauxRococeaux-Sherman" "IndieFlower-Regular"
                    "JosefinSans-Light" "KaushanScript-Regular" "LuckiestGuy-Regular" "Noteworthy-Bold" "Quicksand" "Saira-Regular")
CONCRETE="True"

for wo in "${WORDLIST[@]}"
do
  letter=("N" "B" "T" "I")
  # letter=R
  SEED=0
  for l in "${letter[@]}"
  do
    for f in "${FONTLIST[@]}"
    do
      echo $f
      echo $l
      echo $wo
      CUDA_VISIBLE_DEVICES=0 python code/main.py --experiment "embedding_loss" --semantic_concept study --word $wo --optimized_letter $l --font $f --abstract "True"
    done
  done
done
