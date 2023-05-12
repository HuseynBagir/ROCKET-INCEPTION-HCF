file_names=('ACSF1' 'Adiac' 'ArrowHead' 'Beef' 'BirdChicken' 'Coffee' 'Haptics'  'HouseTwenty' 'MedicalImages' 'MiddlePhalanxTW')

for file_name in "${file_names[@]}"; do
    # source ~/anaconda3/etc/profile.d/conda.sh
    # conda activate tfgpu

    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 10000 --custom-filters 6 --inception-pm $file_name --pooling ppv+mpv+mipv+GAP+max --dataset $file_name

    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 10000 --custom-filters 6 --inception-pm $file_name --pooling ppv+mpv+mipv+GAP --dataset $file_name

    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 10000 --custom-filters 6 --inception-pm $file_name --pooling ppv+mpv+GAP+max --dataset $file_name
done
