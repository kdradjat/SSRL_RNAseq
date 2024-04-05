import torch.nn as nn
import torch.nn.functional as F
import torch
#import wandb
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
from cancerclassification.early_stopping import (
    EarlyStopping,
)



class Net(nn.Module):
    """Basic neural network for cancer classification task.

    Args:
        input_dim (int): Inputs dimension.
        output_dim (int): Outputs dimension.
        l1 (int, optional): Dimension of the first hidden layer. Defaults to 512.
        l2 (int, optional): Dimension of the second hidden layer. Defaults to 256.
        l3 (int, optional): Dimension of the third hidden layer. Defaults to 128.
    """

    def __init__(self, input_dim, output_dim, l1=512, l2=256, l3=128, l4=64, bn=True, dropout_rate=0):
        super().__init__()
        
        self.bn = bn
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_dim, l1)
        self.fc1_bn = nn.BatchNorm1d(l1)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(l1, l2)
        self.fc2_bn = nn.BatchNorm1d(l2)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(l2, l3)
        self.fc3_bn = nn.BatchNorm1d(l3)
        #self.fc4 = nn.Linear(l3, output_dim)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(l3, l4)
        self.fc4_bn = nn.BatchNorm1d(l4)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(l4, output_dim)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.fc1(x)
        if self.bn : x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        if self.bn : x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        if self.bn : x = self.fc3_bn(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc4(x)
        if self.bn : x = self.fc4_bn(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.fc5(x)
        
        """x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.relu(self.fc4_bn(self.fc4(x)))
        x = self.fc5(x)"""
        return x


def train_epoch(net, dataloader, optimizer, loss_function, binary):
    """Trains the network for one epoch.

    Args:
        net (nn.Module): Neural network (model) of the training.
        dataloader (Dataloader): Custom Pytorch Dataloader of the training dataset.
        optimizer (optim): Optimizer of the neural network.
        loss_function (function): Loss function of the neural network.
        binary (bool): Boolean referring to the type of the classification task
            (multiclass / binary)

    Returns:
        (float, float): Cumulated sum of the losses over the mini-batches and number of
        mini-batches.
    """

    epoch_loss = 0
    train_steps = 0

    # Training
    net.train()
    for batch in dataloader:

        # get the inputs; batch is a list of [inputs, labels]
        X, y = batch
        if binary:
            y = y.unsqueeze(1)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = net(X.view(-1, X.shape[1])) 
        loss = loss_function(output, y)
        epoch_loss += loss.item()
        train_steps += 1
        loss.backward()
        optimizer.step()

    return epoch_loss, train_steps

def eval_epoch(net, dataloader, loss_function, binary):
    """Evaluates the network (inference, loss, performance) on the validation or the
    test set.

    Args:
        net (nn.Module): Neural network (model) of the training.
        dataloader (Dataloader): Custom Pytorch Dataloader of the validation or the test
            dataset.
        loss_function (function): Loss function of the neural network.
        binary (bool): Boolean referring to the type of the classification task
            (multiclass / binary)

    Returns:
        (float, float, float, float): Cumulated sum of the losses over the mini-batches,
        number of mini-batches, predictions and ground truth.
    """

    epoch_loss = 0
    eval_steps = 0

    y_true = []
    y_pred = []

    activation = nn.Sigmoid() if binary else nn.Softmax(dim=0)
    net.eval()
    with torch.no_grad():

        for batch in dataloader:
            X, y = batch
            if binary:
                y = y.unsqueeze(1)
            output = net(X.view(-1, X.shape[1]))
            loss = loss_function(output, y)
            epoch_loss += loss.item()
            eval_steps += 1

            if binary:
                prediction = activation(output).detach().cpu().numpy().round()
            else:
                prediction = activation(output).detach().cpu().numpy().argmax(1)

            y_pred += prediction.tolist()
            y_true += y.detach().cpu().numpy().tolist()

    return epoch_loss, eval_steps, y_pred, y_true


def train_nn(
    config,
    dataloaders,
    net,
    binary=False,
    optimizer=optim.Adam,
    weights=None,
    early_stop=0,
    val=True,
    test=True,
    log=True,
    logger=None,
):
    """"

    Args:
        config (dict): Dictionnary containing the hyperparameters.
        dataloaders (tuple of Dataloader): Custom Pytorch Dataloaders of the train,
            validation and the test dataset.
        net (nn.Module): Neural network (model) of the training.
        binary (bool): Boolean referring to the type of the classification task
            (multiclass / binary)
        weights (torch.Tensor, optional): Weights to balance the loss function. Defaults
            to None.
        early_stop (int, optional): Patience of the early stopping of the training. To
            be set to 0 to disable early stopping. Defaults to 0.
        val (bool, optional): If True, performs an evaluation on the validation dataset
            at each epoch. Defaults to True.
        test (bool, optional): If True, performs an evaluation on the test dataset at
            each epoch. Defaults to True. Defaults to True.
        log (bool, optional): If True, the metrics related to the training are logged to
            the console. Defaults to True.
        logger (LogResults, optional): Instance of the class LogResults. If not
            specified, logging is disabled. Defaults to None.
    """

    trainset, valset, testset = dataloaders

    if binary:
        loss_function = nn.BCEWithLogitsLoss(pos_weight=weights)
    else:
        loss_function = nn.CrossEntropyLoss(weight=weights)

    optimizer = optimizer(net.parameters(), lr=config["lr_init"])

    if early_stop:
        es = EarlyStopping(patience=early_stop, min_delta=0.001, log=log)

    for epoch in range(config["epochs"]):
        train_loss, train_steps = train_epoch(
            net, trainset, optimizer, loss_function, binary
        )
        if val:
            val_loss, val_steps, val_y_pred, val_y_true = eval_epoch(
                net, valset, loss_function, binary
            )
        if test:
            test_loss, test_steps, test_y_pred, test_y_true = eval_epoch(
                net, testset, loss_function, binary
            )

        if logger and test:
            logger.log_epoch(
                epoch=epoch,
                valacc=accuracy_score(val_y_true, val_y_pred),
                valloss=val_loss / val_steps,
                testacc=accuracy_score(test_y_true, test_y_pred),
                testloss=test_loss / test_steps,
                optim=config["optim"], 
                bn=config["bn"],
                dropout_rate=config["dropout_rate"]
            )
        elif logger and not test :
            logger.log_epoch(
                epoch=epoch, 
                valacc=accuracy_score(val_y_true, val_y_pred), 
                valloss=val_loss/val_steps, 
                optim=config["optim"], 
                bn=config["bn"],
                dropout_rate=config["dropout_rate"]
            )
        if log:
            print(
                "| Epoch: {:>3}/{} | TrainLoss={:.4f} |".format(
                    epoch + 1,
                    config["epochs"],
                    train_loss / train_steps,
                ),
                end="",
            )
            if val:
                print(
                    " ValLoss={:.4f} | ValAcc={:.4f} |".format(
                        val_loss / val_steps,
                        accuracy_score(val_y_true, val_y_pred),
                    ),
                    end="",
                )
            if test:
                print(
                    " TestLoss={:.4f} | TestAcc={:.4f} |".format(
                        test_loss / test_steps,
                        accuracy_score(test_y_true, test_y_pred),
                    )
                )
            else:
                print("")

        if early_stop and epoch > 5:
            # es(val_loss)
            es(-accuracy_score(val_y_true, val_y_pred))
            if es.early_stop:
                break


def transfer_weights(net, weights_path, strict=False):
    net.load_state_dict(
        torch.load(weights_path),
        strict=strict,
    )