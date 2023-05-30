file_names=('ACSF1' 'Adiac' 'ArrowHead' 'BME' 'Beef' 'BeetleFly' 'BirdChicken' 'Car' 'Chinatown' 'Coffee' 'Computers' 'Crop' 'DiatomSizeReduction' 'DistalPhalanxOutlineAgeGroup' 'DistalPhalanxOutlineCorrect' 'DistalPhalanxTW' 'DodgerLoopDay' 'DodgerLoopGame' 'DodgerLoopWeekend' 'ECG200' 'ECGFiveDays' 'Earthquakes' 'FaceFour' 'FiftyWords' 'Fish' 'FordA' 'GunPoint' 'Ham' 'Haptics' 'Herring' 'HouseTwenty' 'InlineSkate' 'LargeKitchenAppliances' 'Lightning2' 'Lightning7' 'Meat' 'MedicalImages' 'MiddlePhalanxTW' 'OSULeaf' 'OliveOil')

for file_name in "${file_names[@]}"; do
    # source ~/anaconda3/etc/profile.d/conda.sh
    # conda activate tfgpu

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling ppv+lspv+mpv+mipv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling ppv --dataset $file_name

    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 10000 --custom-filters 6 --inception-pm $file_name --pooling ppv+lspv+mpv+mipv+GAP+max --dataset $file_name

    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 10000 --custom-filters 6 --inception-pm $file_name --pooling ppv+lspv+mpv+mipv --dataset $file_name


    python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 10000 --custom-filters 6 --inception-pm $file_name --pooling ppv --dataset $file_name

    python3 get_results.py --dataset $file_name

done
