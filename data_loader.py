import logging
import pickle
import sys
import types
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd


def _patch_old_pandas_pickle_modules():
  
    old_module_name = "pandas.core.indexes.numeric"
    if old_module_name in sys.modules:
        return

    mod = types.ModuleType(old_module_name)
    mod.Int64Index = pd.Index
    mod.UInt64Index = pd.Index
    mod.Float64Index = pd.Index
    # Also expose NumericIndex, which some older pickles reference.
    mod.NumericIndex = pd.Index
    sys.modules[old_module_name] = mod


# Dataset
class MMDataset(Dataset):
    def __init__(self, args):

        self.args = args
        # Dataset mapping based on the name, it currently supports only 'Empathy'
        DATASET_MAP = {  'Empathy': self.__init_evaluation,}
        DATASET_MAP[args['dataset_name']]()

    def __init_evaluation(self):
        # Load the dataset from the pickle file
        # Patch old pandas module paths so legacy pickles can unpickle cleanly.
        _patch_old_pandas_pickle_modules()
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)

        # Extract the relevant data (text, vision, audio) for the specified mode
        self.text = data['text'].astype(np.float32)
        self.vision = data['vision'].astype(np.float32)
        self.audio = data['audio'].astype(np.float32)
        self.info = data['info']

        # Labels are stored as 1-7 (original rating scale) in the pickle file.
        # We center them by subtracting 4, mapping to -3…+3 (midpoint = 0).
        self.labels =  (np.array(data['labels']) - 4).astype(np.float32)

        # Handle missing (NaN) values by setting them to 0 in each modality
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0
        self.text[self.text != self.text] = 0

        # Update feature dimensions in the args dictionary
        self.args['feature_dims'][0] = self.text.shape[2]
        self.args['feature_dims'][1] = self.audio.shape[2]
        self.args['feature_dims'][2] = self.vision.shape[2]

        # Normalize features if required by args
        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()

    def __normalize(self):

        # Transpose (num_examples, max_len, feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

        # Compute the mean over the sequence length (max_len)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # Handle any remaining NaN values by setting them to 0
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        # Transpose back to the original shape (num_examples, max_len, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return self.text.shape[0]

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def __getitem__(self, index):
      
        # Create the sample dictionary with text, audio, vision, and labels
        sample = {
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'labels': torch.Tensor(self.labels[index].reshape(-1)) ,
            "info": self.info[index]
        }

        # Add sequence lengths for audio and vision
        sample['audio_lengths'] = self.audio.shape[0]
        sample['vision_lengths'] = self.vision.shape[0]

        return sample


# DataLoader function for handling multiple dataset splits (train, valid, test)
def MMDataLoader(args, num_workers=0):
    # Create datasets for each split
    datasets =  MMDataset(args)

    # Get the sequence lengths for each modality if specified in the arguments
    if 'seq_lens' in args:
        args['seq_lens'] = datasets.get_seq_len()

    # Create DataLoader objects for dataset
    dataLoader =   DataLoader(datasets ,
                       batch_size=64,
                       num_workers=0,
                       shuffle=False)


    return dataLoader
