#!/bin/bash

set -e

USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="none"
EXPERIMENT="embedding_loss"

WORDLIST=("Romantic")
FONTLIST=("DeliusUnicase-Regular" "HobeauxRococeaux-Sherman" "IndieFlower-Regular"
                    "JosefinSans-Light" "KaushanScript-Regular" "LuckiestGuy-Regular" "Noteworthy-Bold" "Quicksand" "Saira-Regular")
CONCRETE="True"

for wo in "${WORDLIST[@]}"
do
  # letter_=("O")
  letter=R
  SEED=0
  for f in "${FONTLIST[@]}"
  do
    echo $f
    echo $letter
    echo $wo
    CUDA_VISIBLE_DEVICES=0 python code/main.py --semantic_concept $wo --word $wo --optimized_letter $letter --abstract $CONCRETE
  done
done
