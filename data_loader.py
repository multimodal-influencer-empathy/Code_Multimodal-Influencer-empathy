import logging
import pickle
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd


#  Dataset
class MMDataset(Dataset):
    def __init__(self, args):
        """
        Initialize the MMDataset.

        :param args: Dictionary containing various configuration settings,
                     including file paths, dataset name, feature dimensions, etc.
        :param mode: Indicates which dataset split to use ('train', 'valid', 'test').
        """

        self.args = args

        # Dataset mapping based on the name, it currently supports only 'Empathy'
        DATASET_MAP = {  'Empathy': self.__init_evaluation,}
        DATASET_MAP[args['dataset_name']]()

    def __init_evaluation(self):
        """
        Initializes the dataset for evaluation purposes by loading the necessary data
        (text, vision, and audio features) and performing some basic preprocessing.
        """
        # Load the dataset from the pickle file
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)

        # Extract the relevant data (text, vision, audio) for the specified mode
        self.text = data['text'].astype(np.float32)
        self.vision = data['vision'].astype(np.float32)
        self.audio = data['audio'].astype(np.float32)
        self.info = data['info']

        # Labels dictionary to store the labels for the dataset
        self.labels =  np.array(data['labels']).astype(np.float32)

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
        """
        Normalize the audio and vision features by taking the mean across examples.
        This reduces the variability and removes potential NaN values.
        """
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
        """
        Returns the total number of examples in the dataset.
        """
        return self.text.shape[0]

    def get_seq_len(self):
        """
        Get the sequence lengths for text, audio, and vision modalities.

        :return: A tuple containing sequence lengths for text, audio, and vision.
        """
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def __getitem__(self, index):
        """
        Get a single sample from the dataset at the specified index.

        :param index: Index of the sample to retrieve.
        :return: A dictionary containing text, audio, vision features, labels, and  information.
        """
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

    # Create DataLoader objects for each split
    dataLoader =   DataLoader(datasets ,
                       batch_size=256,
                       num_workers=0,
                       shuffle=False)


    return dataLoader
