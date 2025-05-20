import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

MAXLEN = 2000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class CNN1DDataset(Dataset):
    def __init__(self, df, bert_dir, esm2_dir, terms_dict):
        # Initialize data
        self.df = df
        self.bert_dir = bert_dir
        self.esm2_dir = esm2_dir
        self.terms_dict = terms_dict

    def __len__(self):
        # Get total number of samples
        return len(self.df)

    def __getitem__(self, index):

        batch_df = self.df.iloc[index]
        name = batch_df.proteins
        seq = batch_df.sequences
        prop_annotations = batch_df.annotations
        terms_dict = self.terms_dict

        # bert - cafa
        bert = pickle.load(open(self.bert_dir + '/' + name + '.pkl', 'rb'))['pt5']
        features_bert = bert.astype(np.float32).T[:,:bert.shape[0]-1]

        esm2 = pickle.load(open(self.esm2_dir + '/' + name + '.pkl', 'rb'))['esm2']
        features_esm2 = esm2.numpy().astype(np.float32).T


        seqlen = len(seq)
        if seqlen > MAXLEN:
            seqlen = MAXLEN

        # Get labels (N)
        labels = np.zeros(len(terms_dict), dtype=np.int32)
        for g_id in prop_annotations:
            if g_id in terms_dict:
                labels[terms_dict[g_id]] = 1
        return features_bert, seqlen, labels, features_esm2

def cnn1d_collate(batch):

    feats = [item[0] for item in batch]
    lengths = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    esm2 = [item[3] for item in batch]


    truncated_bert_feats = []
    for feat in feats:
        if feat.shape[1] > MAXLEN:
            truncated_bert = feat[:, :MAXLEN]
        else:
            truncated_bert = feat
        truncated_bert_feats.append(truncated_bert)
    feats = truncated_bert_feats

    truncated_esm2_feats = []
    for feat in esm2:
        if feat.shape[1] > MAXLEN:
            truncated_esm2 = feat[:, :MAXLEN]
        else:
            truncated_esm2 = feat
        truncated_esm2_feats.append(truncated_esm2)
    esm2 = truncated_esm2_feats

    # truncated_pca_feats = []
    # for feat in pca:
    #     if feat.shape[1] > MAXLEN:
    #         truncated_pca = feat[:, :MAXLEN]
    #     else:
    #         truncated_pca = feat
    #     truncated_pca_feats.append(truncated_pca)
    # pca = truncated_pca_feats

    max_len = max(lengths)

    # Pad data to max sequence length in batch 对特征进行填充，将每个序列的特征长度填充为批次中最长序列的长度。
    feats_pad = [np.pad(item, ((0,0),(0,max_len-lengths[i])), 'constant') for i, item in enumerate(feats)]

    # else
    x1 = torch.from_numpy(np.array(feats_pad))

    # Pad data to max sequence length in batch
    feats_pad_esm2 = [np.pad(item, ((0, 0), (0, max_len - lengths[i])), 'constant') for i, item in enumerate(esm2)]

    # else
    x2 = torch.from_numpy(np.array(feats_pad_esm2))


    return CustomData(x1=x1, x2=x2, y=torch.from_numpy(np.array(labels)))

class SeqDataset(Dataset):
    def __init__(self, names, feats_dir, terms_dict,feats_type='onehot'):
        # Initialize data
        self.names = names
        self.feats_dir = feats_dir
        self.terms_dict = terms_dict
        self.feats_type = feats_type

    def __len__(self):
        # Get total number of samples
        return len(self.names)

    def __getitem__(self, index):
        # Load sample
        name = self.names[index]
        # print(name)
        terms_dict = self.terms_dict
        d = pickle.load(open(self.feats_dir + '/' + name + '.pkl', 'rb'))
        seq = d['sequence']
        seqlen = len(seq)
        if seqlen > 2000:
            seq = seq[:2000]
            seqlen = 2000

        # Get labels (N)
        prop_annotations = d['Y']
        labels = np.zeros(len(terms_dict), dtype=np.int32)
        for g_id in prop_annotations:
            if g_id in terms_dict:
                labels[terms_dict[g_id]] = 1

        return seq, seqlen,labels

def seq_collate(batch):

    feats = [item[0] for item in batch]
    lengths = [item[1] for item in batch]
    labels = [item[2] for item in batch]

    return CustomData(x=feats, y=torch.from_numpy(np.array(labels)))

class MLPDataset(Dataset):
    def __init__(self, names_file, feats_dir, terms_dict,feats_type='embeddings'):
        # Initialize data
        self.names = list(np.loadtxt(names_file, dtype=str))
        self.feats_dir = feats_dir
        self.terms_dict = terms_dict
        self.feats_type = feats_type

    def __len__(self):
        # Get total number of samples
        return len(self.names)

    def __getitem__(self, index):
        # Load sample
        name = self.names[index]
        terms_dict = self.terms_dict

        # Get protein-level features
        d = pickle.load(open(self.feats_dir + '\\' + name + '.pkl', 'rb'))
        X = d["X"]

        # Select features type
        if self.feats_type == 'onehot':
            # onehot = d["onehot"].toarray().astype(np.float32).T
            features = d["onehot"].astype(np.float32).T
        elif self.feats_type == 'pssm':
            features = X[:, :20].astype(np.float32).T

        elif self.feats_type == 'embedding':
            features = X[:, 20:].astype(np.float32).T

        elif self.feats_type == 'X':
            features = X[:, :20].astype(np.float32).T
            x1 = X[:, 20:].astype(np.float32).T

        else:
            print('[!] Unknown features type, try "embeddings" or "onehot".')
            exit(0)

        features = np.mean(features, 1)

        # Get labels (N)
        prop_annotations = d['Y']
        labels = np.zeros(len(terms_dict), dtype=np.int32)
        for g_id in prop_annotations:
            if g_id in terms_dict:
                labels[terms_dict[g_id]] = 1

        return features, labels

def mlp_collate(batch):
    # Get data, label and length (from a list of arrays)
    feats = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    return CustomData(x=torch.from_numpy(np.array(feats)), y=torch.from_numpy(np.array(labels)))

class CustomData(Data):
    def __init__(self, x=None, x1 = None,x2 = None,x3=None,x4=None,x5=None,y=None, **kwargs):
        super(CustomData, self).__init__()

        self.x = x
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.y = y

        for key, item in kwargs.items():
            self[key] = item