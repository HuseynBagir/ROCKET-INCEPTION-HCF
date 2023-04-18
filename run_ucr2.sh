#!/bin/bash

file_names=('ACSF1' 'Adiac' 'ArrowHead' 'Beef' 'BirdChicken' 'Coffee' 'FordA' 'Haptics' 'HouseTwenty' 'MedicalImages' 'MiddlePhalanxTW')

for file_name in "${file_names[@]}"; do
    # source ~/anaconda3/etc/profile.d/conda.sh
    # conda activate tfgpu
       
    python3 get_results.py --dataset $file_name
done
