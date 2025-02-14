import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import random
import argparse
import os

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing


import utils
from models import * 
from Experiments import *


def parse_args():    
    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('-d', '--dataset', type=str, help='Choice of "traffic", "tourism", "labor"')
    
    args = parser.parse_args()
    return args.dataset

def load_config(dataset, config_file='config.yaml'):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config.get(dataset, {})

def plot(results, metrics, get, model_name, color="black"):
    mean_res = get_mean(results, metrics)    
    plt.plot(range(mean_res.shape[0]), mean_res[get], label=model_name, color=color)

def plot_results(GET):

    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('{} by hierarchy level'.format(GET), fontsize=16)
    plt.xlabel('Hierarchy level', fontsize=14)
    plt.ylabel(GET, fontsize=14)

    plot(base_results, metrics, GET, "base model", color='red')
    plot(projection_results, metrics, GET, "projection model", color='blue')
    plot(coherency_results, metrics, GET, "coherency loss model", color='green')
    plot(profhit_results, metrics, GET, "profhit model", color='orange')

    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/{}_{}.png'.format(dataset_name, GET))

def plot_all_results(): 
    for get in metrics:
        plot_results(get)
    

if __name__ == "__main__":
    
    
    dataset_name = parse_args()
    data, base_agg_mat = utils.load_data(dataset_name)
    params = load_config(dataset_name)
    
    base_results, metrics, base_losses = repeat_exp(BaseModel, base_agg_mat, data, params)
    coherency_results, _, coherency_losses = repeat_exp(BaseModel, base_agg_mat, data, params, coherency_loss=True)
    projection_results, _, projection_losses = repeat_exp(BaseModel, base_agg_mat, data, params, project=True)
    profhit_results, _, profhit_losses = repeat_exp(BaseModel, base_agg_mat, data, params, profhit_loss=True)
    
    
    plot_all_results()