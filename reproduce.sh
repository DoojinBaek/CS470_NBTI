#!/bin/bash
## setting conda env
conda env create -f test_env.yaml

## download dataset

## learn embedding loss
conda activate word
code/letter_classifier 0

## copy best weight checkpoint
cp code/logs/0/max_val_acc_checkpoint.pt ./code/max_val_acc_checkpoint.pt

## Fine tuning GPT-3.5

## Start