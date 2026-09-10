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
    mod.NumericIndex = pd.Index
    sys.modules[old_module_name] = mod


class MMDataset(Dataset):
    def __init__(self, args):
        self.args = args
        DATASET_MAP = {'Empathy': self.__init_evaluation}
        DATASET_MAP[args['dataset_name']]()

    def __init_evaluation(self):
        _patch_old_pandas_pickle_modules()
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)

        self.text = data['text'].astype(np.float32)
        self.vision = data['vision'].astype(np.float32)
        self.audio = data['audio'].astype(np.float32)
        self.info = data['info']

        # Labels: stored as 1-7 in pickle, centered to -3..+3 (midpoint=0) for training
        self.labels = (np.array(data['labels']) - 4).astype(np.float32)

        self.args['feature_dims'][0] = self.text.shape[2]
        self.args['feature_dims'][1] = self.audio.shape[2]
        self.args['feature_dims'][2] = self.vision.shape[2]

        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()

    def __normalize(self):
        # (num_examples, max_len, feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

        # Mean over examples, then transpose back
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)


        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return self.text.shape[0]

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def __getitem__(self, index):
        sample = {
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'labels': torch.Tensor(self.labels[index].reshape(-1)),
            "info": self.info[index]
        }
        sample['audio_lengths'] = self.audio.shape[0]
        sample['vision_lengths'] = self.vision.shape[0]
        return sample


def MMDataLoader(args, num_workers=0):
    datasets = MMDataset(args)
    if 'seq_lens' in args:
        args['seq_lens'] = datasets.get_seq_len()

    dataLoader = DataLoader(datasets, batch_size=64, num_workers=0, shuffle=False)
    return dataLoader
