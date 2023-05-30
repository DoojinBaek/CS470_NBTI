#!/bin/bash

set -e

USE_WANDB=0 # CHANGE IF YOU WANT WANDB
WANDB_USER="none"
EXPERIMENT=conformal_0.5_dist_pixel_100_kernel201

WORDLIST=("SAD" "LIFE" "MATH" "STOP" "WISH" "LAZY" "AGE" "STUDY" "SWEET" "NAME" "SING" "HOLY" "COLD" "FANCY" "CLEAN" "SLEEPY" "CREATIVE" "BRAVE" "ADORABLE" "GENTLE" "DANGEROUS" "FANTASTIC" "CONVERSATION" "EXERCISE" "GLOOMY" "FRESH" "HUGE" "EMPTY" "DAY" "QUESTION" "ROMANTIC" "WORLD" "ENERGETIC" "POTENTIAL" "ATTITUDE" "PROPERTY" "WEIRD" "REWARD" "MINUTE" "VIRTUAL" "TEMPERATURE" "ILLUSION" "RISING" "SPICY" "LOVELY" "AIM" "ATTRACTION" "LONELY" "REVOLUTION" "POETIC")
FONTLIST=("HobeauxRococeaux-Sherman" "DeliusUnicase-Regular")
CONCRETE="True"

for wo in "${WORDLIST[@]}"
do
  # letter_=("O")
  # letter_=("D" "A" "N" "G" "E" "R" "O" "U" "S")
  SEED=0
  for i in $(echo $wo | sed -e 's/\(.\)/\1\n/g');
  do
    for f in "${FONTLIST[@]}"
    do
      echo "$i"
      ARGS="--experiment $EXPERIMENT --optimized_letter ${i} --seed $SEED --font ${f} --use_wandb ${USE_WANDB} --wandb_user ${WANDB_USER} --abstract ${CONCRETE}"
      CUDA_VISIBLE_DEVICES=0 python code/main.py $ARGS --semantic_concept "${wo}" --word "${wo}" --memo "with-new-font"
    done
  done
done
