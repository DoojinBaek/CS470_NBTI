#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
## setting conda env
conda env create -f test_env.yaml

## download dataset
wget http://143.248.235.11:5000/fontsdataset/dataset.zip
unzip dataset.zip -d ./code/data/

## learn embedding loss
conda init bash
conda activate word
python ./code/letter_classifier.py 0 100

## copy best weight checkpoint
cp ./logs/0/max_val_acc_checkpoint.pt .code/max_val_acc_checkpoint.pt

## Fine tuning GPT-3.5
bash ./code/finetuning/finetunemodel.sh

## Start
experiment="embedding_loss"
word="SOFT"
letter="O"
font="JellyChoco.ttf"
python ./code/main.py $ARGS --semantic_concept "${word}" --word "${word}" --font "${font}"  --optimized_letter "${letter}" --experiment "${experiment}" --abstract "True"
