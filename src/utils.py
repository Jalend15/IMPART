from tabulate import tabulate
import pickle
import os
import clip
import torch
from torchvision.datasets import CIFAR100, CIFAR10, Caltech101

os.makedirs('../.cache', exist_ok=True)

# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
_, clip_preprocess = clip.load('ViT-B/32', device)

from src.zero_shot_classifier import ZeroShotClassifier

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

def get_result_dict(file_name='../.cache/result.pkl'):
    with open(file_name, 'rb') as f:
        result_dict = pickle.load(f)
    return result_dict

def clear_results(file_name='../.cache/result.pkl'):
    os.remove(file_name)

def display_results(file_name='../.cache/result.pkl'):
    try:
        with open(file_name, 'rb') as f:
            result_dict = pickle.load(f)
        display_table(result_dict)
    except:
        print('No results found!')

def run_benchmark(model, result_file):
    clf = ZeroShotClassifier()

    cifar10_testset = CIFAR10(root='../.cache/datasets', download=True, train=False, transform=clip_preprocess)
    model_name, dataset_name, accuracy = clf.evaluate_testset(model, cifar10_testset, cifar10_testset.classes)
    save_result(model_name, dataset_name, accuracy, file_name=result_file)

    cifar100_testset = CIFAR100(root='../.cache/datasets', download=True, train=False, transform=clip_preprocess)
    model_name, dataset_name, accuracy = clf.evaluate_testset(model, cifar100_testset, cifar100_testset.classes)
    save_result(model_name, dataset_name, accuracy, file_name=result_file)

    caltech101_testset = Caltech101(root='../.cache/datasets', download=True, transform=clip_preprocess)
    model_name, dataset_name, accuracy = clf.evaluate_testset(model, caltech101_testset, caltech101_testset.categories)
    save_result(model_name, dataset_name, accuracy, file_name=result_file)