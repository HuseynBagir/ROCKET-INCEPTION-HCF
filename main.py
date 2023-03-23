def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from transformations.transformation import Transformation
from classifiers.RIDGE import RIDGE

from utils.utils import load_data, znormalisation, create_directory, encode_labels

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        help="which dataset to run the experiment on.",
        type=str,
        default='Coffee'
    )

    parser.add_argument(
        '--transformation',
        help='which classifier to use',
        type=str,
        choices=['ROCKET', 'HCF', 'ROCKET+HCF', 'HCF+ROCKET'],
        default='ROCKET+HCF'
    )

    parser.add_argument(
        '--rocket-filters',
        type=int,
        default=10000
    )

    parser.add_argument(
        '--custom-filters',
        type=int,
        default=6
    )

    parser.add_argument(
        '--runs',
        help="number of runs to do",
        type=int,
        default=5
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

    transformations = (args.transformation).split('+')

    output_dir_parent = args.output_directory
    create_directory(output_dir_parent)

    # output_dir_transformation = output_dir_parent + args.transformation + '/'
    # create_directory(output_dir_transformation)

    output_dir_transformation = output_dir_parent
    
    for i, transformation_name in enumerate(transformations):

        output_dir_transformation = output_dir_transformation + transformation_name
    
        if transformation_name == 'ROCKET':
            output_dir_transformation = output_dir_transformation + '-' + str(args.rocket_filters)
        
        elif transformation_name == 'HCF':
            output_dir_transformation = output_dir_transformation + '-' + str(args.custom_filters)
        
        if i < len(transformations) - 1:
            output_dir_transformation = output_dir_transformation + '+'
    
    output_dir_transformation = output_dir_transformation + '/'
    create_directory(output_dir_transformation)

    xtrain, ytrain, xtest, ytest = load_data(file_name=args.dataset)

    xtrain = znormalisation(xtrain)
    xtest = znormalisation(xtest)

    ytrain = encode_labels(ytrain)
    ytest = encode_labels(ytest)

    length_TS = int(xtrain.shape[1])
    n_classes = len(np.unique(ytrain))

    for _run in range(args.runs):

        print('dataset ',args.dataset)
        print('run ',_run)

        output_dir = output_dir_transformation + 'run_' + str(_run) + '/'
        create_directory(output_dir)

        output_dir = output_dir + args.dataset + '/'
        create_directory(output_dir)

        df = pd.DataFrame(columns=['accuracy'])

        _Transformation = Transformation(transformations=transformations, length_TS=length_TS,
                                         n_filters_rocket=args.rocket_filters, n_filters_hcf=args.custom_filters)
        
        transformed_xtrain, transformed_xtest = _Transformation.transform(xtrain=xtrain, xtest=xtest)

        clf = RIDGE()
        clf.fit(transformed_xtrain, ytrain)
        acc = clf.predict(transformed_xtest, ytest)

        df = df.append({'accuracy' : acc}, ignore_index=True)

        df.to_csv(output_dir + 'metric.csv', index=False)