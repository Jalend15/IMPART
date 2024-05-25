from tabulate import tabulate
import pandas as pd
import pickle
import os
import clip
import torch

os.makedirs('../.cache', exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
_, clip_preprocess = clip.load('ViT-B/32', device)

def save_result(model, dataset, accuracy, file_name='../.cache/result.pkl'):
    try:
        with open(file_name, 'rb') as f:
            result_dict = pickle.load(f)
    except:
        result_dict = {}
    
    if model not in result_dict:
        result_dict[model] = {}
    result_dict[model][dataset] = accuracy
    
    with open(file_name, 'wb') as f:
        pickle.dump(result_dict, f)

def display_table(result_dict):
    datasets = set()
    for model in result_dict:
        for dataset in result_dict[model]:
            datasets.add(dataset)
    datasets = sorted(list(datasets))

    table_dict = {'': datasets}
    for model in result_dict:
        table_dict[model] = []
        for dataset in datasets:
            if dataset in result_dict[model]:
                table_dict[model].append(result_dict[model][dataset])
            else:
                table_dict[model].append('-')

    print(tabulate(table_dict, headers="keys", tablefmt="fancy_grid"))

def clear_results(file_name='../.cache/result.pkl'):
    os.remove(file_name)

def display_results(file_name='../.cache/result.pkl'):
    try:
        with open(file_name, 'rb') as f:
            result_dict = pickle.load(f)
        display_table(result_dict)
    except:
        print('No results found!')