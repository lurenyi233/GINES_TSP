import argparse
import random
import numpy as np
import pandas as pd
import torch
import datetime
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import os

from models.GINES import GINES
from data.dataset import TSPDataset
from utils.utils import train_model

def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()  # Clear gradients.
        logits = model(data.pos, data.edge_index, data.edge_attr, data.batch)  # Forward pass.
        loss = criterion(logits, data.y)  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0

    for data in loader:
        logits = model(data.pos, data.edge_index, data.edge_attr, data.batch)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    return total_correct / len(loader.dataset)


if __name__ == '__main__':



    seed = 41
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



    starttime = datetime.datetime.now()

    parser = argparse.ArgumentParser(description='Set Folds')
    # parser.add_argument('--folds', default=1, type=int, help='Number of Fold to use')
    parser.add_argument('--k_values', default=10, type=int, help='Number of K in KNN graph')
    parser.add_argument('--h_dim', default=32, type=int, help='Hidden layer dimension')
    parser.add_argument('--aggr', default='max', type=str, help='Aggregation method for GNN layer')
    args = parser.parse_args()

    # fold = args.folds
    k_value = args.k_values
    hidden_channels = args.h_dim
    aggr = args.aggr

    dataset_path = "./data/TSPDataset/"


    for fold in range(1, 11):
        print("==============", fold, "================")

        model = GINES(hidden_channels=hidden_channels, aggr_method=aggr)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.01,
                                     weight_decay=0.01)
        criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

        dataset_train = TSPDataset(root=dataset_path,
                                               split="train", folds=fold)
        dataset_test = TSPDataset(root=dataset_path,
                                              split="val", folds=fold)

        dataset_train.transform = T.Compose([T.KNNGraph(k=k_value), T.NormalizeScale(), T.Distance()])
        dataset_test.transform = T.Compose([T.KNNGraph(k=k_value), T.NormalizeScale(), T.Distance()])

        dataset_train, dataset_valid = torch.utils.data.random_split(
            dataset=dataset_train,
            lengths=[810, 90],
            generator=torch.Generator().manual_seed(0)
        )

        train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
        valid_loader = DataLoader(dataset_valid, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=100, shuffle=False)

        model, train_loss, valid_loss, test_acc_list = train_model(model, optimizer, criterion, train_loader, valid_loader, test_loader, 20,
                                                                   100)

        optimizer.zero_grad()
        test_acc = test(model, test_loader)
        test_acc_list.append(test_acc)
        print(f'Test Accuracy: {test_acc:.4f}')

        experiment_name = "GINEConv_" + str(hidden_channels) + "_" + str(k_value) + "_" + str(fold) + "_" + aggr
        csv_name = experiment_name + ".csv"
        dir_name = str(os.path.split(os.path.realpath(__file__))[0])
        pd.DataFrame(np.array(test_acc_list, ndmin=2)).to_csv(dir_name + '/' + csv_name, mode='a',
                                                              index=True, header=True)
        model_name = experiment_name + ".pth"
        torch.save(model.state_dict(), dir_name + '/' + model_name)

    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)