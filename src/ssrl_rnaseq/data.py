import random
import numpy as np  # manipulate N-dimensional arrays
import pandas as pd  # data frame
from sklearn import preprocessing  # basic ML models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from math import ceil
from torch.utils.data import RandomSampler, BatchSampler, SequentialSampler
import os
import torch
from sklearn.model_selection import KFold


class FastDataLoader:
    """Data loader. Combines a dataset and a sampler, and provides an iterable over the
    given dataset.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load. Defaults to 64.
        shuffle (bool, optional): set to True to have the data reshuffled at every
            epoch. Defaults to False.
        sampler (Sampler or Iterable, optional): defines the strategy to draw samples
            from the dataset. Can be any Iterable with __len__ implemented. If
            specified, shuffle must not be specified.
        drop_last (bool, optional): set to True to drop the last incomplete batch, if
            the dataset size is not divisible by the batch size. If False and the size
            of dataset is not divisible by the batch size, then the last batch will be
            smaller. Defaults to True.
    """

    def __init__(
        self, dataset, batch_size=64, shuffle=False, sampler=None, drop_last=True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with " "shuffle")

        if sampler is None:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        self.sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self):
        self.idx_iterator = iter(self.sampler)
        return self

    def __next__(self):
        idx = next(self.idx_iterator)
        return self.dataset[idx]

    def __len__(self):
        length = len(self.dataset)
        if self.drop_last:
            return length // self.batch_size
        else:
            return ceil(length / self.batch_size)


def read_process_data_TCGA(
    data_path,
    label_path,
    coding_genes=False,
    coding_genes_file='../data/protein-coding_gene.txt',
    preprocess_type='standard'
):
    class_df = pd.read_parquet(label_path)
    data_df = pd.read_parquet(data_path)

    # merging the dataframes based on "caseID"
    class_df["caseID"] = class_df.apply(lambda row: row.cases.split("|")[1], axis=1)
    df = class_df.merge(data_df, on="caseID", how="inner")
    #df = class_df.iloc[:100].merge(data_df.iloc[:100], on="caseID", how="inner")
    
    df = df.drop(columns=list(df.columns[:7]) + [df.columns[8]] + [df.columns[9]])  # columns management
    
    # encoding cancer names to integers
    le = preprocessing.LabelEncoder()
    df["cancer_type"] = le.fit_transform(df["cancer_type"])
    print(df.columns)
    np_dataset = df.to_numpy(dtype=np.float32)

    # normal standardardization
    if preprocess_type == 'standard': scaler = preprocessing.StandardScaler()
    elif preprocess_type == 'minmax': scaler = preprocessing.MinMaxScaler()
    np_dataset[:, 1:] = scaler.fit_transform(np_dataset[:, 1:])

    return np_dataset


def read_process_data_TCGA_unlabel(
    data_path,
    coding_genes=False,
    coding_genes_file='../data/protein-coding_gene.txt',
    preprocess_type='standard'
):
    data_df = pd.read_parquet(data_path)
    data_df = data_df.drop(columns="caseID")

    np_dataset = data_df.to_numpy(dtype=np.float32)

    # normal standardardization
    scaler = preprocessing.StandardScaler()
    np_dataset = scaler.fit_transform(np_dataset)

    return np_dataset


def read_process_data_ARCHS4(
    data_path,
    label_path,
    coding_genes=False,
    coding_genes_file='../data/protein-coding_gene.txt',
    preprocess_type='standard'
):
    class_df = pd.read_parquet(label_path)
    data_df = pd.read_parquet(data_path)
    labels = class_df["labels"]
    
    # encoding cancer names to integers
    le = preprocessing.LabelEncoder()
    data_df.insert(0, 'labels', le.fit_transform(class_df["labels"]))
    print(data_df.columns)
    np_dataset = data_df.to_numpy(dtype=np.float32)

    # normal standardardization
    if preprocess_type == 'standard': scaler = preprocessing.StandardScaler()
    elif preprocess_type == 'minmax': scaler = preprocessing.MinMaxScaler()
    np_dataset[:, 1:] = scaler.fit_transform(np_dataset[:, 1:])

    return np_dataset


def read_process_data_ARCHS4_unlabel(
    data_path,
    binary=False,
    coding_genes=False,
    coding_genes_file='../data/protein-coding_gene.txt',
    preprocess_type='standard'
):
    data_df = pd.read_parquet(data_path)
    
    if coding_genes:
        protein_coding_file = pd.read_csv(coding_genes_file, '\t')
        selected_columns = np.unique(protein_coding_file['ensembl_gene_id'].tolist()).tolist()
        selected_columns.pop()
        genes = data_df.columns
        intersection = list(set(selected_columns) & set(genes))
        data_df = data_df[intersection]
        
    np_dataset = data_df.to_numpy(dtype=np.float32)

    # normal standardardization
    if preprocess_type == 'standard': scaler = preprocessing.StandardScaler()
    elif preprocess_type == 'minmax': scaler = preprocessing.MinMaxScaler()
    np_dataset = scaler.fit_transform(np_dataset)

    return np_dataset


def read_data_TCGA_preprocessed():
    """Reads preprocessed TCGA data.

    Returns:
        numpy.ndarray: Numpy array of the preprocessed data.
    """
    return np.load("data/prepared_datasets/tcga_scaled.npy", allow_pickle=True)


def read_data_TCGA_survival_preprocessed():
    """Reads preprocessed TCGA data for the survival task.

    Returns:
        numpy.ndarray: Numpy array of the preprocessed data.
    """
    return np.load("data/prepared_datasets/tcga_survival_scaled.npy", allow_pickle=True)


def read_data_MA_preprocessed():
    """Reads preprocessed MicroArray data.

    Returns:
        numpy.ndarray: Numpy array of the preprocessed data.
    """
    return np.load("data/prepared_datasets/microarray_scaled.npy", allow_pickle=True)


class CancerDatasetTCGA(Dataset):
    """Custom Pytorch dataset to work with the TCGA data.

    Args:
        inputs (numpy.ndarray): 2-dimensional array of the inputs of the neural network.
        labels (numpy.ndarray): Array of the labels associated with the inputs.
        device (str, optional): Identifier of the cuda device. Defaults to None.
    """

    def __init__(self, inputs, labels, device=None):
        self.inputs = torch.from_numpy(inputs).to(dtype=torch.float32)
        self.labels = torch.from_numpy(labels).to(dtype=torch.long)

        if device:
            self.inputs = self.inputs.to(device)
            self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        label = self.labels[idx]
        values = self.inputs[idx]
        return values, label


class CancerDatasetMA(Dataset):
    """Custom Pytorch dataset to work with the MicroArray data.

    Args:
        inputs (numpy.ndarray): 2-dimensional array of the inputs of the neural network.
        labels (numpy.ndarray): Array of the labels associated with the inputs.
        device (str, optional): Identifier of the cuda device. Defaults to None.
    """

    def __init__(self, inputs, labels, device=None):
        self.inputs = torch.from_numpy(inputs).to(dtype=torch.float32)
        self.labels = torch.from_numpy(labels).to(dtype=torch.float32)

        if device:
            self.inputs = self.inputs.to(device)
            self.labels = self.labels.to(device)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        label = self.labels[idx]
        values = self.inputs[idx]
        return values, label


def generate_indices(data, prop=1, val_prop=0.15, test_prop=0.15, rs=0):
    """Generated train, validation and test indices that will be used in the
    dataloaders.

    Args:
        data (numpy.ndarray): 2-dimensional array of the dataset. The first column has
            to contain the class (ex: cancer / no cancer) information.
        prop (int, optional): Proportion of the dataset that is used to generate the
            indices. Defaults to 1.
        val_prop (float, optional): Proportion of data dedicated to the validation set.
            Defaults to 0.15.
        test_prop (float, optional): Proportion of data dedicated to the test set.
            Defaults to 0.15.
        rs (int, optional): Random state. Defaults to 0.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): Train, validation and test
        indices.
    """
    indices = list(range(len(data)))
    
    if test_prop != 0 :
        train_idx, test_idx = train_test_split(
            indices, test_size=test_prop, stratify=data[:,0], train_size=None, random_state=rs
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_prop / (1 - test_prop),
            train_size=None,
            #stratify=data[train_idx,0],
            random_state=rs,
        )
    else :
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_prop,
            train_size=None,
            stratify=data[:,0],
            random_state=rs,
        )
        test_idx=[]
    if prop != 1:
        modes = data[train_idx, 0]
        subtrain_idx = []
        for mode in np.unique(modes):
            candidates = np.array(train_idx)[np.argwhere(modes == mode).flatten()]
            # adding progressively 
            selected_idx = candidates[: round(len(candidates) * prop)]
            # random
            #selected_idx = np.random.choice(candidates, int(round(len(candidates)*prop)), replace=False)
            subtrain_idx += selected_idx.tolist()
        train_idx = subtrain_idx
        print(len(train_idx))

    return (train_idx, val_idx, test_idx)

def generate_indices_pretraining(data, prop=1, rs=0):
    indices = list(range(len(data)))
    if prop != 1:
        modes = data[:, 0]
        subtrain_idx = []
        for mode in np.unique(modes):
            candidates = np.array(indices)[np.argwhere(modes == mode).flatten()]
            # adding progressively 
            selected_idx = candidates[: round(len(candidates) * prop)]
            # random
            #selected_idx = np.random.choice(candidates, int(round(len(candidates)*prop)), replace=False)
            subtrain_idx += selected_idx.tolist()
        train_idx = subtrain_idx

    return train_idx


def compute_loss_weights(data, device=None):
    """Computes weights based on class proportions to balance the loss function. This
    function is specialized for the use of the cross entropy loss function.

    Args:
        data (numpy.ndarray): 2-dimensional array of the dataset. The first column has
            to contain the class (ex: cancer / no cancer) information.
        device (str, optional): Identifier of the cuda device. Defaults to None.

    Returns:
        torch.Tensor: Tensor containing the weights.
    """
    classes = np.unique(data[:, 0])
    w = torch.FloatTensor(
        compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=data[:, 0],
        )
    )
    if device:
        w = w.to(device)
    return w


def compute_loss_pos_weights(data, device=None):
    """Computes postive class weight based on class proportions to balance the loss
    function. This function is specialized for the use of the binary cross entropy with
    logits loss function.

    Args:
        data (numpy.ndarray): 2-dimensional array of the dataset. The first column has
            to contain the class (ex: cancer / no cancer) information.
        device (str, optional): Identifier of the cuda device. Defaults to None.

    Returns:
        torch.Tensor: Tensor containing the positive weight.
    """
    n_pos = sum(data[:, 0] == 1)
    n_neg = sum(data[:, 0] == 0)
    w = n_neg / n_pos
    w = torch.FloatTensor([w])
    if device:
        w = w.to(device)
    return w


def get_dataloaders(dataset, idx, bs=None, fast=True, verbose=True, drop_last=True):
    """Initiates and returns the train, validation, test dataloaders.

    Args:
        dataset (Dataset): Pytorch format dataset instance.
        idx ((numpy.ndarray, numpy.ndarray, numpy.ndarray)): train, validation and test
            indices.
        bs ([int, int, int], optional): If specified, defines the mini-batches size. If
            not, no mini-batch. Defaults to None.
        fast (bool, optional): If True, the functions uses the custom "FastDataLoader"
            class. Defaults to True.
        verbose (bool, optional): Verboseness. Defaults to True.

    Returns:
        (DataLoader, DataLoader, DataLoader): Train, validation and test dataloaders.
    """

    if verbose:
        print(f"{len(dataset)} elements in the dataset")

    train_idx, val_idx, test_idx = idx

    if bs is None:
        bs = [max(1, len(train_idx)), max(1, len(val_idx)), max(1, len(test_idx))]

    train_sample = SubsetRandomSampler(train_idx)
    val_sample = SubsetRandomSampler(val_idx)
    test_sample = SubsetRandomSampler(test_idx)

    Dload = DataLoader if fast else FastDataLoader

    trainset = Dload(dataset, batch_size=bs[0], sampler=train_sample, drop_last=drop_last)
    valset = Dload(dataset, batch_size=bs[1], sampler=val_sample, drop_last=drop_last)
    testset = Dload(dataset, batch_size=bs[2], sampler=test_sample, drop_last=drop_last)

    if verbose:
        print(f"{len(train_idx)} elements in the trainset")
        print(f"{len(val_idx)} elements in the valset")
        print(f"{len(test_idx)} elements in the testset")

    return (trainset, valset, testset)


def train_val_test_fold(n, n_splits=8, rs=0):
    """Generates K splits of the dataset in order to perform a K-fold cross test. The
    aim is not to find validation parameters but it's to have a better approximation of
    the model's performances when the dataset become too small. Used for the survival
    tasks.

    Args:
        n (int): length of the dataset
        n_splits (int, optional): Number of folds desired. Defaults to 8.
        rs (int, optional): Random state. Defaults to 0.

    Returns:
        list of tuple: List of length K containing the generated folds, each fold being
        a tuple of train, validation and test indices.
    """

    folds = []
    np.random.seed(rs)

    indices = np.array(range(n))
    np.random.shuffle(indices)

    kf = KFold(n_splits=n_splits)
    kf.get_n_splits(indices)

    for train_index, test_index in kf.split(indices):
        idx_train, idx_test = indices[train_index], indices[test_index]
        prop_train = len(idx_train) / n
        prop_test = len(idx_test) / n
        prop_val = prop_test / prop_train
        idx_val = idx_train[: round(prop_val * len(idx_train))]
        idx_train = idx_train[round(prop_val * len(idx_train)) :]
        folds += [(idx_train, idx_val, idx_test)]

    return folds


class LogResults:
    """Class implementing an object for the monitoring of the training of a neural
    network. Hyperparameters of the training can be specified, abd metrics of each epoch
    of the training are logged. Logs can be saved at any time to a csv format file.

    Args:
        name (str): Name of the project.
        hyp_str (list of str): List of the names of hyperparameters.
        hyp_vals (list, optional): Values for the hyperparameters. Defaults to None.
    """

    def __init__(self, name, hyp_str, hyp_vals=None):
        self.name = name
        self.counter = 0
        self.df_results = pd.DataFrame(
            columns=["id", "epoch", "val_acc", "val_loss", "test_acc", "test_loss", "optim", "bn", "dropout_rate"]
            + hyp_str
        )
        self.hyp_vals = hyp_vals

    def log_epoch(
        self, epoch=None, valacc=None, valloss=None, testacc=None, testloss=None, optim=None, bn=None, dropout_rate=None
    ):
        """Logs last epoch metrics by appending it to a global Pandas dataframe.

        Args:
            epoch (int, optional): Epoch number. Defaults to None.
            valacc (float, optional): Validation accuracy. Defaults to None.
            valloss (float, optional): Validation loss. Defaults to None.
            testacc (float, optional): Test accuracy. Defaults to None.
            testloss (float, optional): Test loss. Defaults to None.
        """
        ###
        #print([f"run-{self.counter}", epoch, valacc, valloss, testacc, testloss, optim, bn, dropout_rate]+ self.hyp_vals)
        #print(self.df_results.columns)
        ###
        new_serie = [f"run-{self.counter}", epoch, valacc, valloss, testacc, testloss, optim, bn, dropout_rate] + self.hyp_vals
        """new_serie = pd.DataFrame(
            [f"run-{self.counter}", epoch, valacc, valloss, testacc, testloss, optim, bn, dropout_rate]
            + self.hyp_vals,
            index=self.df_results.columns,
        )"""
        
        self.df_results.loc[len(self.df_results)] = new_serie
        #self.df_results = self.df_results.append(new_serie, ignore_index=True)
        #self.df_results = pd.concat([self.df_results, new_serie])
        #print(self.df_results)

    def update_hyps(self, hyp_vals):
        """Changes the hyperparameters values.

        Args:
            hyp_vals (list): List of the new values for the hyperparameters.
        """
        self.hyp_vals = hyp_vals

    def next_run(self):
        """Stops monitoring the current run (training) and initiates a new one."""
        self.counter += 1

    def save_csv(self):
        """Saves the logs to a csv file. Saved to the current directory, the filed is named based on the project name."""
        self.df_results.to_csv(f"{self.name}.csv")

    def show_progression(self):
        """Prints last epochs of the training to the console."""
        print(f"\nIteration {self.counter}, below are the last 5 epochs :")
        print(self.df_results.tail(5))