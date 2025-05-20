import argparse
from pathlib import Path

# root data path
path = Path("/home/415/hz_project/MMSMAPlus-master/data")

parser = argparse.ArgumentParser()

parser.add_argument('--num_head', dest='num_head', type=int, default=4)

parser.add_argument('--phase', dest='phase', default='train')  # 'train' / 'test'
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16)
parser.add_argument('--num_epochs', dest='num_epochs', type=int, default=12)  # homo : 15 / cafa : 12
parser.add_argument('--namespace', dest='namespace', default="bp")
parser.add_argument('--datasets', dest='datasets', default="cafa3")  # cafa3 / swiss_port
parser.add_argument('--net_type', dest='net_type', default="MERGO")  # MERGO
parser.add_argument('--namespace_dir', dest='namespace_dir', default=f"{path}")
parser.add_argument('--bert_dir', dest='bert_dir', default='/home/415/hz_project/MMSMAPlus-master/data/pt5/cafa3')  # cafa_feats / homo_feats / mouse_bert
parser.add_argument('--esm2_dir', dest='esm2_dir', default='/home/415/hz_project/MMSMAPlus-master/data/esm2_3b/cafa3')  # cafa_feats / homo_feats / mouse_bert

parser.add_argument('--model_dir', dest='model_dir')
parser.add_argument('--out_file', dest='out_file')
parser.add_argument('--device', dest='device')



def get_config():
    args = parser.parse_args()
    return args
