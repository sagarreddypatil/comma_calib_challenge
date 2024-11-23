#!/bin/bash

# for i in {0..4}; do
#     # run the prediction script
#     echo "Predicting $i.hevc"
#     python3 predict.py labeled/$i.hevc > predictions/$i.txt
# done

for i in {5..9}; do
    # run the prediction script
    echo "Predicting $i.hevc"
    python3 predict.py unlabeled/$i.hevc > submission/$i.txt
done