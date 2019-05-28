import math
import random
import argparse

import utils.data_utils as data_utils


def split_data(data, splits):
    """Split the data in smiles_list to train, dev and test"""
    assert len(splits) == 3

    # test_p is redundant, but included to make the splits explicit
    train_p, val_p, test_p = splits
    assert train_p + val_p + test_p == 1.0

    n = len(data)
    train_idx = math.floor(n * train_p)
    val_idx = math.floor(n * val_p) + train_idx

    train_split = data[:train_idx]
    dev_split = data[train_idx:val_idx]
    test_split = data[val_idx:]

    return (train_split, dev_split, test_split)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, default='')
    parser.add_argument('-splits', type=str, default='0.8,0.1,0.1')
    parser.add_argument('-n_splits', type=int, default=10)
    parser.add_argument('-output_dir', type=str, default='')
    parser.add_argument('-multi', action='store_true', default=False)
    parser.add_argument('-rings', action='store_true', default=False)
    args = parser.parse_args()

    assert args.data_path != '' and args.output_dir != ''

    if args.rings:
        mol_data = data_utils.read_smiles_ring_data(args.data_path)
    elif args.multi:
        mol_data = data_utils.read_smiles_multiclass(args.data_path)
    else:
        mol_data = data_utils.read_smiles_from_file(args.data_path)
    n_data = len(mol_data)
    data_indices = list(range(0, n_data))
    splits = [float(x) for x in args.splits.split(',')]

    for i in range(args.n_splits):
        random.shuffle(data_indices)
        data_splits = split_data(data_indices, splits)

        output_file = open('%s/split_%d.txt' % (args.output_dir, i), 'w+')
        for idx, data_type in enumerate(['train', 'valid', 'test']):
            indices = data_splits[idx]
            indices = [str(x) for x in indices]

            split_str = ','.join(indices)
            output_file.write(data_type + ',' + split_str + '\n')
        output_file.close()


if __name__ == '__main__':
    main()
