import torch
from torchvision.datasets import CIFAR100
from classification import utils

class BaseClassifier:

    def __init__(self, name):
        self.name = name

    def train(self, dataset):
        pass

    def get_predictions(self, dataset, classes):
        pass

    def evaluate_testset(self, dataset, classes, dataset_name=None):
        predictions, class_ids = self.get_predictions(dataset, classes)
        num_correct = torch.sum(predictions == class_ids).item()
        accuracy = num_correct/len(class_ids) * 100

        if dataset_name is None:
            dataset_name = dataset.__class__.__name__
        utils.save_result(self.name, dataset_name, f'{accuracy:.2f}')
        print(f"Accuracy: {accuracy:.2f}")
