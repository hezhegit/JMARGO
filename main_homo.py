import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from Ablation.models import MERGO_Abl
from homo.config_homo import get_config
from data_utils import get_data_homo
from homo.engine_homo import train, load_checkpoint, test
from homo.protein_datasets_homo import CNN1DDataset, cnn1d_collate
from networks import MERGO
from utils import Ontology, now, NAMESPACES, FUNC_DICT, evaluate_annotations

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# random seed
def fix_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 0. setting
    args = get_config()
    go = Ontology(f'{args.namespace_dir}/{args.datasets}/go.obo', with_rels=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.model_dir = "models_homo/" + args.datasets + "/" + args.namespace.upper() + "/" + args.net_type
    args.out_file = args.model_dir + "/prediction"

    os.makedirs(args.model_dir, exist_ok=True)

    logger_file = "models_homo/" + args.datasets + "/" + args.namespace.upper() + "/{}_{}_{}_result.txt".format(
        args.datasets, args.namespace.upper(), args.net_type)

    F_txt = open(logger_file, "a+")



    # 1. train set
    terms_dict, terms, data, num_classes = get_data_homo(args.namespace_dir + "/" + args.datasets + "/Training",
                                                                           args.namespace)

    # 3. loss
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    # 4. 5 kf
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=216)
    fold_results_fmax = []
    fold_results_smin = []
    fold_results_aupr = []

    for fold, (train_index, test_index) in enumerate(kf.split(data)):
        print(f"======Fold {fold + 1}/{k}======")
        # dataset
        train_df = data.iloc[train_index]
        test_df = data.iloc[test_index]
        # 20% => valid_df
        n = len(train_df)
        index = np.arange(n)
        np.random.shuffle(index)
        train_size = int(n * 0.8)
        valid_df = train_df.iloc[index[train_size:]]
        train_df = train_df.iloc[index[:train_size]]


        print('*'*20+'  [1]  '+'*'*20, file=F_txt)

        # init model
        if args.net_type == "MERGO":
            net = MERGO(num_classes=num_classes, top_k=4, model_type='all').to(device)
        elif args.net_type == "MERGO_DL":
            net = MERGO(num_classes=num_classes, top_k=4, model_type='dl').to(device)
        elif args.net_type == "MERGO_SL":
            net = MERGO(num_classes=num_classes, top_k=4, model_type='sl').to(device)
        # ablation
        elif args.net_type == "MERGO_ESM2_SL":
            net = MERGO_Abl(num_classes=num_classes, model_type='esm2_sl').to(device)
        elif args.net_type == "MERGO_PT5_SL":
            net = MERGO_Abl(num_classes=num_classes, model_type='pt5_sl').to(device)
        elif args.net_type == "MERGO_ESM2_DL":
            net = MERGO_Abl(num_classes=num_classes, model_type='esm2_dl').to(device)
        elif args.net_type == "MERGO_PT5_DL":
            net = MERGO_Abl(num_classes=num_classes, model_type='pt5_dl').to(device)

        print(f"{now()} init model finished")
        print("[*] Number of model parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad))
        print("[*] Number of model parameters: ", sum(p.numel() for p in net.parameters() if p.requires_grad),
              file=F_txt)


        # dataset
        print(
            f"[*] train data: {len(train_df)}\t valid data: {len(valid_df)}\t test data: {len(test_df)}\t num_classes: {num_classes}")

        print(
            f"[*] train data: {len(train_df)}\t valid data: {len(valid_df)}\t test data: {len(test_df)}\t num_classes: {num_classes}",
            file=F_txt)

        train_loader = DataLoader(
            CNN1DDataset(df=train_df, bert_dir=args.bert_dir, esm2_dir=args.esm2_dir, terms_dict=terms_dict),
            batch_size=args.batch_size, shuffle=True, collate_fn=cnn1d_collate, pin_memory=True, num_workers=2
        )

        valid_loader = DataLoader(
            CNN1DDataset(df=valid_df, bert_dir=args.bert_dir, esm2_dir=args.esm2_dir, terms_dict=terms_dict),
            batch_size=args.batch_size, shuffle=False, collate_fn=cnn1d_collate, pin_memory=True, num_workers=2
        )

        test_loader = DataLoader(
            CNN1DDataset(df=test_df, bert_dir=args.bert_dir, esm2_dir=args.esm2_dir, terms_dict=terms_dict),
            batch_size=args.batch_size, shuffle=False, collate_fn=cnn1d_collate, pin_memory=True, num_workers=2
        )


        # train
        ckpt_dir = args.model_dir + '/checkpoint'
        os.makedirs(ckpt_dir, exist_ok=True)

        if args.phase == "train":
            print(f'=============== Training in the train set \t{now()}===============')
            print(f'=============== Training in the train set \t{now()}===============', file=F_txt)
            # Training and validation
            train(args=args, net=net, criterion=criterion, train_loader=train_loader, valid_loader=valid_loader,
                  test_loader=test_loader, ckpt_dir=ckpt_dir,
                  F_txt=F_txt, device=device, kf=fold)

        # test
        model_list = [f'fold_{fold}']
        train_df = pd.concat([train_df, valid_df], ignore_index=True)
        print(f"Training Size:{len(train_df)}\t Valing Size:{len(test_df)}")
        num_epoch = 0
        for name in model_list:
            model_dir = ckpt_dir + f'/model_{name}.pth.tar'
            print(model_dir)
            epoch_num = load_checkpoint(net, filename=model_dir)
            if epoch_num != num_epoch:
                num_epoch = epoch_num
            else:
                break
            epoch_num, test_loss, test_rocauc, test_fmax, y_true, y_pred_sigm = test(device=device, net=net,
                                                                                     criterion=criterion,
                                                                                     model_file=model_dir,
                                                                                     test_loader=test_loader)
            print(f"[*] Loaded checkpoint at epoch {epoch_num} for {name}ing:", file=F_txt)
            print("Test ROC AUC:{:.4f}\tTest Fmax:{:.4f}".format(test_rocauc, test_fmax), file=F_txt)
            # save prediction.pkl
            test_df['labels'] = list(y_true)
            test_df['preds'] = list(y_pred_sigm)
            test_df.to_pickle(args.out_file+f'_{fold}.pkl')

            # evaluate performance
            annotations = test_df['annotations'].values
            annotations = list(map(lambda x: set(x), annotations))
            test_annotations = []
            for i, row in enumerate(test_df.itertuples()):
                annots = set()
                for go_id in row.annotations:
                    if go.has_term(go_id):
                        annots |= go.get_anchestors(go_id)  # 将当前注释及其所有祖先的注释信息合并到 annots 集合中。
                test_annotations.append(annots)

            go.calculate_ic(annotations + test_annotations)
            go_set = go.get_namespace_terms(NAMESPACES[args.namespace[:2].lower()])  # 获取指定命名空间中的所有术语。
            go_set.remove(FUNC_DICT[args.namespace[:2].lower()])  # 从命名空间术语集合中移除特定的术语

            labels = test_annotations
            labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)),
                              labels))  # 对测试数据集中的每个样本的注释信息进行处理，只保留在指定命名空间中的术语，将其转换为集合。

            fmax = 0.0
            tmax = 0.0  # 对应最大 F-score 的阈值
            smin = 1000.0
            precisions = []
            recalls = []
            # 在每个阈值下，对测试数据集中的每个样本进行预测，根据预测的分数 y_pred_sigm 和当前阈值，筛选出预测的注释。
            for t in range(101):
                threshold = t / 100.0
                preds = []
                for i, row in enumerate(test_df.itertuples()):
                    annots = set()
                    for j, score in enumerate(y_pred_sigm[i]):
                        if score >= threshold:
                            annots.add(terms[j])

                    new_annots = set()
                    for go_id in annots:
                        new_annots |= go.get_anchestors(go_id)  # 将预测的注释进行处理，包括将其转换为祖先注释，并且只保留在指定命名空间中的注释
                    preds.append(new_annots)

                # Filter classes
                preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))

                fscore, prec, rec, s = evaluate_annotations(go, labels, preds)
                precisions.append(prec)
                recalls.append(rec)
                print(f'Fscore: {fscore}, S: {s}, threshold: {threshold}')
                if fmax < fscore:
                    fmax = fscore
                    tmax = threshold

                if smin > s:
                    smin = s

            precisions = np.array(precisions)
            recalls = np.array(recalls)
            sorted_index = np.argsort(recalls)
            recalls = recalls[sorted_index]
            precisions = precisions[sorted_index]
            aupr = np.trapz(precisions, recalls)
            fold_results_fmax.append(fmax)
            fold_results_smin.append(smin)
            fold_results_aupr.append(aupr)
            print(f'===Fold {fold + 1}/{k}=== Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}')
            print(f'===Fold {fold + 1}/{k}=== Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, AUPR: {aupr:0.3f}, threshold: {tmax}', file=F_txt)

            F_txt.flush()

    fmax_avg = np.mean(fold_results_fmax)
    smin_avg = np.mean(fold_results_smin)
    supr_avg = np.mean(fold_results_aupr)

    print(f'===KF-AVG=== Fmax: {fmax_avg:0.3f}, Smin: {smin_avg:0.3f}, AUPR: {supr_avg:0.3f}')
    print(f'===KF-AVG=== Fmax: {fmax_avg:0.3f}, Smin: {smin_avg:0.3f}, AUPR: {supr_avg:0.3f}',
          file=F_txt)

    F_txt.close()


if __name__ == '__main__':
    fix_random_seed(216)
    main()
