import argparse
import os
import pandas as pd
import numpy as np
from sys import exit

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        help="which dataset to run the experiment on.",
        type=str,
        default='Coffee'
    )

    parser.add_argument(
        '--output-directory',
        help="output directory parent",
        type=str,
        default='results/'
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = get_args()

    output_dir_results = args.output_directory
    
    for root, transformations, files in os.walk(output_dir_results):

        if not os.path.exists(output_dir_results + 'results_ucr.csv'):
            df = pd.DataFrame(columns=['dataset']+[transformation for transformation in transformations])

        else:

            df = pd.read_csv(output_dir_results + 'results_ucr.csv')
            datasets = list(df['dataset'])

            if args.dataset in datasets:
                
                print("Already done")
                exit()
        
        dict_to_add = {'dataset' : args.dataset}

        for transformation in transformations:
            
            output_dir_transformation = output_dir_results + transformation + '/'

            for root, runs, files in os.walk(output_dir_transformation):

                scores = []

                for run in runs:

                    output_dir_run = output_dir_transformation + run + '/'
                    output_dir_dataset = output_dir_run + args.dataset + '/'

                    acc = np.asarray(pd.read_csv(output_dir_dataset + 'metric.csv'))[0,0]
                    scores.append(acc)

                dict_to_add[transformation] = np.mean(scores)

                break
        break

    df = df.append(dict_to_add, ignore_index=True)
    df.to_csv(output_dir_results + 'results_ucr.csv', index=False)