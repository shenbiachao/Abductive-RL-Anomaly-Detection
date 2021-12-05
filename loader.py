import pandas as pd
import config
import torch
from sklearn.utils import shuffle
import numpy as np


def load_har():
    source = pd.read_csv("./data/har.csv")
    source = shuffle(source)
    width = source.shape[1]
    class_one = pd.DataFrame(columns=source.columns)
    class_two = pd.DataFrame(columns=source.columns)
    temp = pd.DataFrame(columns=source.columns)
    for i in range(len(source)):
        label = source.iloc[i, width - 1]
        if label != 1 and label != 2:
            temp = temp.append(source.iloc[i, :])
    for i in range(len(source)):
        label = source.iloc[i, width - 1]
        if label == 1 and len(class_one) < len(temp) * config.contamination_rate / 2:
            class_one = class_one.append(source.iloc[i, :])
        elif label == 2 and len(class_two) < len(temp) * config.contamination_rate / 2:
            class_two = class_two.append(source.iloc[i, :])
    source = pd.concat([temp, class_one, class_two])
    source = source.sample(frac=1).reset_index(drop=True)

    length = len(source)
    dataset_train = source.iloc[:int(length*config.train_percentage), :]
    dataset_test = source.iloc[int(length*config.train_percentage):, :]
    dataset_l = dataset_train.iloc[:int(len(dataset_train)*config.labeled_percentage), :]
    dataset_u = dataset_train.iloc[int(len(dataset_train)*config.labeled_percentage):, :]

    l_label = torch.tensor(dataset_l.values)[:, -1].float().to(config.device)
    l_label = torch.tensor([1 if i == 1 or i == 2 else 0 for i in l_label])
    dataset_l = torch.tensor(dataset_l.values)[:, :-1].float().to(config.device)
    dataset_u = torch.tensor(dataset_u.values)[:, :-1].float().to(config.device)
    test_label = torch.tensor(dataset_test.values)[:, -1].float().to(config.device)
    test_label = torch.tensor([1 if i == 1 or i == 2 else 0 for i in test_label])
    dataset_test = torch.tensor(dataset_test.values)[:, :-1].float().to(config.device)

    return dataset_l, l_label, dataset_u, dataset_test, test_label
