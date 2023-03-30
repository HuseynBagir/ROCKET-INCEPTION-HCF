#!/bin/bash

file_names=('ACSF1' 'Adiac' 'AllGestureWiimoteX' 'AllGestureWiimoteY' 'AllGestureWiimoteZ' 'ArrowHead' 'BME' 'Beef' 'BeetleFly' 'BirdChicken' 'CBF' 'Car' 'Chinatown' 'ChlorineConcentration' 'CinCECGTorso' 'Coffee' 'Computers' 'CricketX' 'CricketY' 'CricketZ' 'Crop' 'DiatomSizeReduction' 'DistalPhalanxOutlineAgeGroup' 'DistalPhalanxOutlineCorrect' 'DistalPhalanxTW' 'DodgerLoopDay' 'DodgerLoopGame' 'DodgerLoopWeekend' 'ECG200' 'ECG5000' 'ECGFiveDays' 'EOGHorizontalSignal' 'EOGVerticalSignal' 'Earthquakes' 'ElectricDevices' 'EthanolLevel' 'FaceAll' 'FaceFour' 'FacesUCR' 'FiftyWords' 'Fish' 'FordA' 'FordB' 'FreezerRegularTrain' 'FreezerSmallTrain' 'Fungi' 'GestureMidAirD1' 'GestureMidAirD2' 'GestureMidAirD3' 'GesturePebbleZ1' 'GesturePebbleZ2' 'GunPoint' 'GunPointAgeSpan' 'GunPointMaleVersusFemale' 'GunPointOldVersusYoung' 'Ham' 'HandOutlines' 'Haptics' 'Herring' 'HouseTwenty' 'InlineSkate' 'InsectEPGRegularTrain' 'InsectEPGSmallTrain' 'InsectWingbeatSound' 'ItalyPowerDemand' 'LargeKitchenAppliances' 'Lightning2' 'Lightning7' 'Mallat' 'Meat' 'MedicalImages' 'MelbournePedestrian' 'MiddlePhalanxOutlineAgeGroup' 'MiddlePhalanxOutlineCorrect' 'MiddlePhalanxTW' 'MixedShapesRegularTrain' 'MixedShapesSmallTrain' 'MoteStrain' 'NonInvasiveFetalECGThorax1' 'NonInvasiveFetalECGThorax2' 'OSULeaf' 'OliveOil' 'PLAID' 'PhalangesOutlinesCorrect' 'Phoneme' 'PickupGestureWiimoteZ' 'PigAirwayPressure' 'PigArtPressure' 'PigCVP' 'Plane' 'PowerCons' 'ProximalPhalanxOutlineAgeGroup' 'ProximalPhalanxOutlineCorrect' 'ProximalPhalanxTW' 'RefrigerationDevices' 'Rock' 'ScreenType' 'SemgHandGenderCh2' 'SemgHandMovementCh2' 'SemgHandSubjectCh2' 'ShakeGestureWiimoteZ' 'ShapeletSim' 'ShapesAll' 'SmallKitchenAppliances' 'SmoothSubspace' 'SonyAIBORobotSurface1' 'SonyAIBORobotSurface2' 'StarLightCurves' 'Strawberry' 'SwedishLeaf' 'Symbols' 'SyntheticControl' 'ToeSegmentation1' 'ToeSegmentation2' 'Trace' 'TwoLeadECG' 'TwoPatterns' 'UMD' 'UWaveGestureLibraryAll' 'UWaveGestureLibraryX' 'UWaveGestureLibraryY' 'UWaveGestureLibraryZ' 'Wafer' 'Wine' 'WordSynonyms' 'Worms' 'WormsTwoClass' 'Yoga')

for file_name in "${file_names[@]}"; do
    # source ~/anaconda3/etc/profile.d/conda.sh
    # conda activate tfgpu
     python3 -u main.py --transformation ROCKET+HCF --rocket-filters 500 --custom-filters 0 --dataset $file_name
     python3 -u main.py --transformation ROCKET+HCF --rocket-filters 500 --custom-filters 6 --dataset $file_name
     python3 -u main.py --transformation ROCKET+HCF --rocket-filters 517 --custom-filters 0 --dataset $file_name
     python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 500 --custom-filters 0 --pooling ppv+max --dataset $file_name
     python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 500 --custom-filters 0 --pooling GAP --inception-pm $file_name --dataset $file_name
     python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 500 --custom-filters 6 --pooling ppv+max --inception-pm $file_name --dataset $file_name
     python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 500 --custom-filters 6 --pooling GAP --inception-pm $file_name --dataset $file_name
     python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 517 --custom-filters 0 --pooling ppv+max --inception-pm $file_name --dataset $file_name
     python3 -u main.py --transformation ROCKET+HCF+Inception --rocket-filters 517 --custom-filters 0 --pooling GAP --inception-pm $file_name --dataset $file_name
     python3 get_results.py --dataset $file_name
done
