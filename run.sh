#!/bin/bash
set -e

EXPERIMENT="embedding_loss"
CONCRETIZER="TRUE"

WORD="SOFT"
LETTER="O"
FONT="JellyChoco"

CUDA_VISIBLE_DEVICES=0 python code/main.py --semantic_concept $WORD --word $WORD --optimized_letter $LETTER --font $FONT --experiment $EXPERIMENT --abstract ${CONCRETIZER} --seed 0 --use_wandb 0 --wandb_user none
