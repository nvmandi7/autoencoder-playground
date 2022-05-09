import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

class CeresDataset(Dataset):
    """Torch dataset for loading Ceres data."""

    def __init__(self, pickle_file, transform=None):
        """
        Args:
            pickle_file (string): Path to the pickled pandas dataframe. 
                Format: N rows and k columns produces N data vectors of length k
                One column should be "Delayed", which will be used as the label and removed from the data vector

            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        dataframe = pd.read_pickle(pickle_file)
        self.df = dataframe

        # -------------------------------------------------------
        # Experiments

        # Sort by Order Entry Date
        dataframe.sort_values(by=["Order Entry Date"])

        # -------------------------------------------------------

        labels = dataframe["Delayed"] # int 0 or 1
        dataframe.drop(columns=["Delayed"], axis=1, inplace=True)

        self.data = torch.tensor(dataframe.to_numpy())
        self.labels = torch.tensor(labels.to_numpy())
        self.pickle_file = pickle_file
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise RuntimeError("CeresDataset: getitem idx greater than length of dataset")
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'shipment': self.data[idx], 'is_delayed': self.labels[idx]}

        # TEMP FOR TESTING - random data
        # sample = {'shipment': torch.rand(len(self.data[0])), 'is_delayed': int(torch.rand(1) < 0.5)}

        if self.transform:
            sample = self.transform(sample)

        return sample["shipment"], sample["is_delayed"]
    
    def data_len(self):
        return self.data.shape[1]
    
    # Return data partitioned by class
    def get_by_true_class(self, subset=None):
        data = self.data
        labels = self.labels

        if subset:
            data = data[subset.indices]
            labels = labels[subset.indices]

        positives = data[torch.where(labels == 1)]
        negatives = data[torch.where(labels == 0)]
        
        return positives, negatives

    # Return two subsets of given proportions, in order
    def partition(self, train_proportion):
        partition_index = int(train_proportion * len(self))
        train_subset = Subset(self, range(partition_index))
        test_subset = Subset(self, range(partition_index, len(self)))
        return train_subset, test_subset
