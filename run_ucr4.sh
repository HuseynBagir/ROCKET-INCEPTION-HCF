file_names=('ACSF1' 'Adiac' 'ArrowHead' 'Beef' 'BirdChicken' 'CinCECGTorso' 'Coffee' 'Haptics'  'HouseTwenty' 'MedicalImages' 'MiddlePhalanxTW')

for file_name in "${file_names[@]}"; do
    python3 get_results.py --dataset $file_name
done


