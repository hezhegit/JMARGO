import os
import sys


sys.path.append('.')

from extract_esm import extract_esm

def main(device):
    ont = 'bp'
    save_dir = '/home/415/hz_project/data_processed/esm2_3b/'
    os.makedirs(save_dir, exist_ok=True)

    fasta_file = '/home/415/hz_project/HEAL/data/nrPDB-GO_2019.06.18_val_sequences.fasta'
    extract_esm(fasta_file, save_dir, device=device, out_file=None)

    # for identifier, esm2 in ems2_seq.items():
    #     esm2 = esm2.numpy().squeeze()
    #     with open(os.path.join(save_dir, '{}.pkl'.format(identifier)), 'wb') as f:
    #         pickle.dump({"esm2": esm2}, f)
        # out = pickle.load(open(os.path.join(save_dir, '{}.pkl'.format(identifier)),'rb'))['esm2']

if __name__ == '__main__':
    main('cuda:1')
