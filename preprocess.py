from multiprocessing import Pool
import rdkit.Chem as Chem
import math, random, sys
import pickle
import argparse
from functools import partial
import torch
import numpy
import os 
from hgraph import MolGraph, common_atom_vocab, PairVocab
import rdkit

def to_numpy(tensors):
    convert = lambda x : x.numpy() if type(x) is torch.Tensor else x
    a,b,c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c

def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)

def tensorize_pair(mol_batch, vocab):
    x, y = zip(*mol_batch)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) #no need of order for x

def tensorize_cond(mol_batch, vocab):
    x, y, cond = zip(*mol_batch)
    cond = [map(int, c.split(',')) for c in cond]
    cond = numpy.array(cond)
    x = MolGraph.tensorize(x, vocab, common_atom_vocab)
    y = MolGraph.tensorize(y, vocab, common_atom_vocab)
    return to_numpy(x)[:-1] + to_numpy(y) + (cond,) #no need of order for x


def compute_chunk(split_id, list_of_lines, batch_size, vocab, savedir):
    smiles  = [ Chem.MolToSmiles(Chem.MolFromSmiles(line.strip())) for line in list_of_lines ]
    batches = ( smiles[i : i + batch_size] for i in range(0, len(smiles), batch_size) )
    batches_tensors = [ tensorize(batch, vocab) for batch in batches ]
    with open( os.path.join(savedir, 'tensors-%d.pkl' % split_id), 'wb') as f:
        pickle.dump(batches_tensors, f, pickle.HIGHEST_PROTOCOL)

def read_line(x):
#    print(x)
    return Chem.MolToSmiles(Chem.MolFromSmiles(x.strip()))
    
if __name__ == "__main__":
    from tqdm import tqdm
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mode', type=str, default='pair')
    parser.add_argument('--savedir', type=str, default='./')
    parser.add_argument('--batches_per_file', type=int, default=1000)
    parser.add_argument('--ncpu', type=int, default=8)
    args = parser.parse_args()

    with open(args.vocab) as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    args.vocab = PairVocab(vocab, cuda=False)

    pool = Pool(args.ncpu) 
    random.seed(1)

    if args.mode == 'pair':
        #dataset contains molecule pairs
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:2] for line in f]

        random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize_pair, vocab = args.vocab)
        all_data = pool.map(func, batches)
        num_splits = max(len(all_data) // args.batches_per_file, 1)

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'cond_pair':
        #dataset contains molecule pairs with conditions
        with open(args.train) as f:
            data = [line.strip("\r\n ").split()[:3] for line in f]

        random.shuffle(data)

        batches = [data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)]
        func = partial(tensorize_cond, vocab = args.vocab)
        all_data = pool.map(func, batches)
        num_splits = max(len(all_data) // args.batches_per_file, 1)

        le = (len(all_data) + num_splits - 1) // num_splits

        for split_id in range(num_splits):
            st = split_id * le
            sub_data = all_data[st : st + le]

            with open('tensors-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

    elif args.mode == 'single':
        #dataset contains single molecules
        from joblib import Parallel, delayed
        with open(args.train) as f:
            data = pool.map(read_line, ( line for line in f))                     

        n_smiles = len(data)
        num_per_chunk = args.batch_size*args.batches_per_file 
        num_splits = max(len(data) // num_per_chunk, 1)
        
        random.shuffle(data)
        print("N smiles:%d, N_splits:%d"%(n_smiles, num_splits))
        Parallel(n_jobs=args.ncpu, batch_size=1)( delayed(compute_chunk)(*args) for args in 
                                      tqdm( (i, data[ i*num_per_chunk : (i+1)*num_per_chunk ], 
                                                        args.batch_size, args.vocab, args.savedir)
                                                 for i in range(num_splits) ) )
                                                 
                                               
