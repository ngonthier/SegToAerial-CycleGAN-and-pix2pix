import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import glob
import numpy as np
import cv2
import sys
import argparse
import json
import torch_fidelity
import pathlib
import shutil
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html


def create_folder_per_modality(path_results='\results\flair_pix2pix\test_latest\images'):
    fakes = glob.glob(os.path.join(path_results, '*_fake_B.png'))
    reals = glob.glob(os.path.join(path_results, '*_real_B.png'))
    pathlib.Path(os.path.join(path_results,'reals')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(path_results,'fakes')).mkdir(parents=True, exist_ok=True)
    for fake in fakes:
        src = fake
        dst = os.path.join(path_results,'fakes',os.path.basename(fake))
        shutil.copyfile(src, dst)
    for real in reals:
        src = real
        head, tail = os.path.split(os.path.split(real)[0])
        dst = os.path.join(path_results,'reals',os.path.basename(real))
        shutil.copyfile(src, dst)


def eval_pix2pix(method, path):

    create_folder_per_modality(path_results=path)
    real_path = os.path.join(path,'reals')
    fake_path = os.path.join(path,'fakes')

    eval_dict = { 'method': method}

    # 'kid_subset_size': 50, 'kid_subsets': 10 pour le vrai cas 
    eval_args = {'isc': True, 'fid': True, 'kid': True, 'kid_subset_size': 1, 'kid_subsets': 1, 'verbose': False, 'cuda': True}
    metric_dict_AB = torch_fidelity.calculate_metrics(input1=real_path, input2=fake_path, **eval_args)
    print('metric_dict_AB',metric_dict_AB)
    eval_dict['ISC'] = metric_dict_AB['inception_score_mean']
    eval_dict['FID'] = metric_dict_AB['frechet_inception_distance']
    eval_dict['KID'] = metric_dict_AB['kernel_inception_distance_mean']*100.
    print('[*] evaluation finished!')
    
    #print('rmse: {0}, acc5: {1}, acc10: {2}, pFPR: {3}, iFPR: {4}'.format(RMSE, acc5, acc10, pFPR100, iFPR))
    return eval_dict


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='path to the generated and gt images', default='output_images')
    parser.add_argument('--output_path', type=str, help='path to save the evaluation results', default='results')
    parser.add_argument('--dataset', type=str, default='FLAIR')
    parser.add_argument('--method', type=str, default='pix2pix')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)
    return args

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    eval_metrics = eval_pix2pix(args.method, args.path)
    pathlib.Path(os.path.join(args.output_path)).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_path,args.method + '_' + args.dataset + '_result.json'), 'w') as fp:
        json.dump(eval_metrics, fp)
    

if __name__ == '__main__':
    main()