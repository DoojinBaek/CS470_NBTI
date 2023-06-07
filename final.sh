#!/bin/bash

# Specify the directory path
directory="/root/CS470_Final/CS470_Word_As_Image/code/data/fonts"

words=("IDEA" "VICTORY" "SAD" "LIFE" "OPTIMAL" "IDEAL" "SOFT" "SOLID" "CRUCIAL" "INFINITE" "ETERNAL" "FORGIVE")

letters=("A" "R" "A" "I" "M" "A" "O" "L" "A" "N" "R" "G")

experiment="embedding_loss"

# Check if the arrays have the same length
if [ ${#array1[@]} -ne ${#array2[@]} ]; then
  echo "Arrays must have the same length."
  exit 1
fi

# Check if the directory exists
if [ -d "$directory" ]; then
  # Loop through each file in the directory
  for file in "$directory"/*; do
    # Check if the file is a regular file (not a directory)
    if [ -f "$file" ]; then
      # Remove the ".ttf" extension from the file name
      filename=$(basename "$file")
      font="${filename%.ttf}"
      for ((i=0; i<${#words[@]}; i++)); do
        word=${words[i]}
        letter=${letters[i]}
        CUDA_VISIBLE_DEVICES=0 python code/main.py $ARGS --semantic_concept "${word}" --word "${word}" --font "${font}"  --optimized_letter "${letter}" --experiment "${experiment}" --abstract "True"
      done
    fi
  done
else
  echo "Directory not found."
fi