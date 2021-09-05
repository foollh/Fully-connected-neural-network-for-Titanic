import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Preprocessing import preprocesssing as pre


# prepare data
class TitanicDataSet(Dataset):
    def __init__(self, filepath, train):
        self.train = train
        self.trainData, self.testData = pre(filepath)
        if self.train == True:
            self.len = self.trainData.shape[0]
        else:
            self.len = self.testData.shape[0]

    def __getitem__(self, item):
        if self.train == True:
            inputs = self.trainData[item][1:8]
            labels = self.trainData[item][0]
        else:
            inputs = self.testData[item][1:8]
            labels = self.testData[item][0]
        return inputs, labels

    def __len__(self):
        return self.len


# model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(7, 200)
        self.linear2 = nn.Linear(200, 100)
        self.linear3 = nn.Linear(100, 10)
        self.linear4 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x


# train
def train(model, epochs, criterion, optimizer, train_loader, test_loader):
    for epoch in range(epochs):
        temp = []
        for i, (inputs, labels) in enumerate(train_loader):
            # forward
            labels = labels.unsqueeze(-1)
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            temp.append(loss.item())
            # print(epoch, i, loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:", epoch, "loss:", sum(temp) / len(temp))
        if epoch % 10 == 0:
            test(model, test_loader)


# test
def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            y_pred = model(inputs)
            y_pred = torch.where(y_pred > 0.5, 1, 0)
            y_pred=y_pred.squeeze(-1)
            total += labels.size(0)
            correct += (y_pred == labels).sum().item()
    print("accuracy on test dataset: ", (100 * correct / total))


if __name__ == "__main__":
    train_path = "../DataSet/titanic/train.csv"
    # test_path = "../DataSet/titanic/test.csv"

    titanic_train = TitanicDataSet(train_path, train=True)
    titanic_test = TitanicDataSet(train_path, train=False)

    train_loader = DataLoader(dataset=titanic_train, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=titanic_test, batch_size=32, shuffle=False, num_workers=4)
    # loss and optimizer
    model = Model()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train(model, 120, criterion, optimizer, train_loader, test_loader)
