import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def read_process_data(data_path, label_path) :
  data_df = pd.read_parquet(data_path)
  class_df = pd.read_parquet(label_path)
  
  # merging the dataframes based on "caseID"
  class_df["caseID"] = class_df.apply(lambda row: row.cases.split("|")[1], axis=1)
  df = class_df.merge(data_df, on="caseID", how="inner")
  
  df = df.drop(columns=list(df.columns[:7]) + [df.columns[8]] + [df.columns[9]])  # columns management
  
  # encoding cancer names to integers
  le = preprocessing.LabelEncoder()
  df["cancer_type"] = le.fit_transform(df["cancer_type"])
  print(df.columns)
  np_dataset = df.to_numpy(dtype=np.float32)
  
  # normal standardardization
  scaler = preprocessing.StandardScaler()
  np_dataset[:, 1:] = scaler.fit_transform(np_dataset[:, 1:])

  return np_dataset


def read_process_data_TCGA_unlabel(data_path):
  data_df = pd.read_parquet(data_path)
  data_df = data_df.drop(columns="caseID")

  np_dataset = data_df.to_numpy(dtype=np.float32)

  # normal standardardization
  scaler = preprocessing.StandardScaler()
  np_dataset = scaler.fit_transform(np_dataset)

  return np_dataset
  

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
            indices, test_size=test_prop, train_size=None, random_state=rs
        )
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=val_prop / (1 - test_prop),
            train_size=None,
            random_state=rs,
        )
    else :
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_prop,
            train_size=None,
            random_state=rs,
        )
        test_idx=[]
    if prop != 1:
        modes = data[train_idx, 0]
        subtrain_idx = []
        for mode in np.unique(modes):
            candidates = np.array(train_idx)[np.argwhere(modes == mode).flatten()]
            selected_idx = candidates[: round(len(candidates) * prop)]
            #selected_idx = np.random.choice(candidates, int(round(len(candidates)*prop)), replace=False)
            subtrain_idx += selected_idx.tolist()
        train_idx = subtrain_idx
        print(len(train_idx))

    return (train_idx, val_idx, test_idx)


def read_process_data_MA(
    data="E-MTAB-3732.data2.parquet", label="classes.parquet", selected_type=None
):
    """Reads and processes (including a normal standardardization) the MicroArray data.

    Args:
        path (str, optional): Path of the folder containing MicroArray's csv files.
            Defaults to "/home/commun/data/MicroArray/E-MTAB-3732/".
        selected_type (str, optional): Value can be specified as "patient" or
            "cell line" to select only a part of the dataset. Defaults to None.

    Returns:
        numpy.ndarray: Numpy array of the processed data.
    """
    class_df = pd.read_parquet(label)
    data_df = pd.read_parquet(data)

    data_df = data_df.drop(columns=[data_df.columns[0]])

    #  because the columns / rows are inverted in the csv file...
    data_np = data_df.to_numpy(dtype=np.float32).T

    class_np = class_df["Cancer"].to_numpy()
    le = preprocessing.LabelEncoder()
    le.classes_ = ["normal", "cancer"]
    class_np = le.transform(class_np)
    class_np = class_np.astype(np.float32)

    if selected_type is not None:
        selected = (class_df["cell"] == selected_type).to_numpy()
        class_np = class_np[selected]
        data_np = data_np[selected]

    scaler = preprocessing.StandardScaler()
    data_np = scaler.fit_transform(data_np)

    return np.concatenate((np.expand_dims(class_np, axis=0), data_np.T)).T