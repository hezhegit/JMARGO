import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

MAXLEN = 2000


class CNN1DDataset(Dataset):
    def __init__(self, df, bert_dir, esm2_dir, terms_dict):
        # Initialize data
        self.df = df
        self.bert_dir = bert_dir
        self.esm2_dir = esm2_dir
        self.terms_dict = terms_dict
        # self.bert_pca = joblib.load('/home/415/hz_project/MMSMAPlus-master/data/cafa3/pca_bert_n0.95.m')

    def __len__(self):
        # Get total number of samples
        return len(self.df)

    def __getitem__(self, index):
        """
        :param index: 根据给定索引 index 加载对应的样本。
        :return:
        """
        # Load sample 从 df 中获取样本信息，包括蛋白质名称、序列、注释等
        batch_df = self.df.iloc[index]
        name = batch_df.proteins
        seq = batch_df.sequences
        prop_annotations = batch_df.annotations
        terms_dict = self.terms_dict


        # bert - homo
        bert = pickle.load(open(self.bert_dir + '/' + name + '.pkl', 'rb'))['pt5']
        features_bert = bert.astype(np.float32).T[:,:bert.shape[0]-1]

        # esm2 homo
        esm2 = pickle.load(open(self.esm2_dir + '/' + name + '.pkl', 'rb'))['esm2']
        features_esm2 = esm2.numpy().astype(np.float32).T

        seqlen = len(seq)
        if seqlen > MAXLEN:
            seqlen = MAXLEN

        # Get labels (N) 构建样本的标签向量 labels，将基因ID映射到对应的标签索引，存在标签的位置置为1，否则为0。
        labels = np.zeros(len(terms_dict), dtype=np.int32)
        for g_id in prop_annotations:
            if g_id in terms_dict:
                labels[terms_dict[g_id]] = 1
        return features_bert, seqlen, labels, features_esm2

"""
这段代码定义了一个自定义函数 cnn1d_collate，用作 pyDataLoader 的 collate_fn 参数，用于在加载数据时对数据进行批处理和填充。
- 接收一个批次（batch）的数据，包含特征、序列长度、标签以及可能的子特征。
- 对数据进行填充和组织，使其能够组成一个批次，并返回。
"""
def cnn1d_collate(batch):

    # 从批次数据中分别提取特征、序列长度、标签以及可能的子特征，存储在相应的列表中。
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


    max_len = max(lengths)

    # Pad data to max sequence length in batch 对特征进行填充，将每个序列的特征长度填充为批次中最长序列的长度。
    feats_pad = [np.pad(item, ((0,0),(0,max_len-lengths[i])), 'constant') for i, item in enumerate(feats)]

    # else
    x1 = torch.from_numpy(np.array(feats_pad))

    # Pad data to max sequence length in batch 对特征进行填充，将每个序列的特征长度填充为批次中最长序列的长度。
    feats_pad_esm2 = [np.pad(item, ((0, 0), (0, max_len - lengths[i])), 'constant') for i, item in enumerate(esm2)]

    # else
    x2 = torch.from_numpy(np.array(feats_pad_esm2))



    # 构建一个 CustomData 对象，其中包含填充后的特征、子特征和标签，并返回。
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
        # Load pickle file with dictionary containing embeddings (LxF), sequence (L) and labels (1xN)
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