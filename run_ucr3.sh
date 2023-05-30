file_names=('ACSF1' 'Adiac' 'ArrowHead' 'Beef' 'BirdChicken' 'CinCECGTorso' 'Coffee' 'FordA' 'Haptics'  'HouseTwenty' 'MedicalImages' 'MiddlePhalanxTW')

for file_name in "${file_names[@]}"; do
    # source ~/anaconda3/etc/profile.d/conda.sh
    # conda activate tfgpu

    python3 -u main.py --transformation Inception --inception-pm $file_name --pooling ppv+lspv+mpv+mipv --dataset $file_name

    python3 get_results.py --dataset $file_name

done
