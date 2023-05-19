file_names=('ACSF1' 'Adiac' 'ArrowHead' 'Beef' 'BirdChicken' 'CinCECGTorso' 'Coffee' 'FordA' 'Haptics'  'HouseTwenty' 'MedicalImages' 'MiddlePhalanxTW')

for file_name in "${file_names[@]}"; do
    # source ~/anaconda3/etc/profile.d/conda.sh
    # conda activate tfgpu

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling ppv+lspv+mpv+mipv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling ppv+lspv+mpv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling ppv+lspv+mipv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling ppv+mpv+mipv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling lspv+mpv+mipv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling ppv+lspv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling ppv+mpv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling ppv+mipv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling lspv+mpv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling lspv+mipv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling mpv+mipv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling ppv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling lspv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling mpv --dataset $file_name

    python3 -u main.py --transformation MultiROCKET+HCF+Inception --multirocket-filters 40000 --custom-filters 6 --inception-pm $file_name --pooling mipv --dataset $file_name

done
