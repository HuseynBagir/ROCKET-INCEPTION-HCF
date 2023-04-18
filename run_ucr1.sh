#!/bin/bash

file_names=('ACSF1' 'Adiac' 'ArrowHead' 'Beef' 'BirdChicken' 'Coffee' 'FordA' 'Haptics' 'HouseTwenty' 'MedicalImages' 'MiddlePhalanxTW')

for file_name in "${file_names[@]}"; do
    # source ~/anaconda3/etc/profile.d/conda.sh
    # conda activate tfgpu

    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 517 --custom-filters 6 --inception-pm $file_name --pooling ppv+max+GAP --dataset $file_name
    
    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 500 --custom-filters 6 --inception-pm $file_name --pooling ppv+max+GAP --dataset $file_name
    
    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 500 --custom-filters 6 --inception-pm $file_name --pooling ppv+GAP --dataset $file_name
    
    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 500 --custom-filters 6 --inception-pm $file_name --pooling ppv+max --dataset $file_name
    
    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 500 --custom-filters 6 --inception-pm $file_name --pooling max+GAP --dataset $file_name
    
    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 500 --custom-filters 6 --inception-pm $file_name --pooling ppv --dataset $file_name
    
    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 500 --custom-filters 6 --inception-pm $file_name --pooling max --dataset $file_name
    
    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 500 --custom-filters 6 --inception-pm $file_name --pooling GAP --dataset $file_name
   
    python3 get_results.py --dataset $file_name
done
