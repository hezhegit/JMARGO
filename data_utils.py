import pandas as pd
import numpy as np


def get_data(path, ont, split=0.95):
    """
    :param path: 数据集路径
    :param ont: mf bp cc
    :param split: 训练集train=》split训练集+（1-split）验证集
    :return: 标签terms及字典，训练集，验证集，测试集，标签个数
    """
    namespace_dir = path + "/" + ont
    # terms = np.loadtxt(namespace_dir + "/terms.txt", dtype=str)  # mouse
    terms = np.loadtxt(namespace_dir + f"/{ont}.txt", dtype=str)
    num_classes = len(terms)
    terms_dict = {v: i for i, v in enumerate(terms)}
    terms_df = pd.DataFrame({'terms': terms})
    terms = terms_df['terms'].values.flatten()

    train_df = pd.read_pickle(namespace_dir + '/train_data.pkl')
    test_df = pd.read_pickle(namespace_dir + '/test_data.pkl')
    n = len(train_df)
    index = np.arange(n)
    np.random.shuffle(index)
    train_size = int(n * split)
    valid_df = train_df.iloc[index[train_size:]]
    train_df = train_df.iloc[index[:train_size]]

    print(
        f"[*] train data: {len(train_df)} \t valid data : {len(valid_df)} \t  test data: {len(test_df)} \t    num_classes : {num_classes}")

    return terms_dict, terms, train_df, valid_df, test_df, num_classes


def get_data_homo(path, ont):
    """
    :param path: 数据集路径
    :param ont: mf bp cc
    :param split: 训练集train=》split训练集+（1-split）验证集
    :return: 标签terms及字典，训练集，验证集，测试集，标签个数
    """
    namespace_dir = path + "/" + ont
    terms = np.loadtxt(namespace_dir + "/terms.txt", dtype=str)  # mouse
    # terms = np.loadtxt(namespace_dir + "/terms.txt", dtype=str)
    num_classes = len(terms)
    terms_dict = {v: i for i, v in enumerate(terms)}
    terms_df = pd.DataFrame({'terms': terms})
    terms = terms_df['terms'].values.flatten()

    train_df = pd.read_pickle(namespace_dir + '/train_data.pkl')


    return terms_dict, terms, train_df, num_classes


def get_data_sw(path, ont):
    """
    :param path: 数据集路径
    :param ont: mf bp cc
    :param split: 训练集train=》split训练集+（1-split）验证集
    :return: 标签terms及字典，训练集，验证集，测试集，标签个数
    """
    namespace_dir = path + "/" + ont
    # terms = np.loadtxt(namespace_dir + "/terms.txt", dtype=str)  # mouse
    terms = np.loadtxt(namespace_dir + f"/{ont}.txt", dtype=str)
    num_classes = len(terms)
    terms_dict = {v: i for i, v in enumerate(terms)}
    terms_df = pd.DataFrame({'terms': terms})
    terms = terms_df['terms'].values.flatten()

    train_df = pd.read_pickle(namespace_dir + '/train_data.pkl')
    test_df = pd.read_pickle(namespace_dir + '/test_data.pkl')
    valid_df = pd.read_pickle(namespace_dir + '/valid_data.pkl')

    print(
        f"[*] train data: {len(train_df)} \t valid data : {len(valid_df)} \t  test data: {len(test_df)} \t    num_classes : {num_classes}")

    return terms_dict, terms, train_df, valid_df, test_df, num_classes