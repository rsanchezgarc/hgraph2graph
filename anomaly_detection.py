import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse
from tqdm import tqdm

from hgraph import *
from preprocess import tensorize
import rdkit
from rdkit import Chem, DataStructs

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', required=True)

parser.add_argument('--seed', type=int, default=7)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)


parser.add_argument('--query_smiles', type=argparse.FileType('r'), required=True)

args = parser.parse_args()

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)]
prepro_vocab = PairVocab(vocab, cuda=False)
args.vocab = PairVocab(vocab)

model = HierVAE(args).cuda()

model.load_state_dict(torch.load(args.model)[0])
model.eval()

torch.manual_seed(args.seed)
random.seed(args.seed)

smiles = [ x.strip() for x in args.query_smiles.readlines()]
n_smiles = len(smiles)
batch_size = args.batch_size
#print(smiles)

import requests
url = "https://api.postera.ai/api/v1/synthetic-accessibility/fast-score/batch/"
headers = {"X-API-KEY": "v1:mAOYT8fFItGktDWNvo35vw"}; 

import importlib

spec = importlib.util.spec_from_file_location( "sascorer", os.path.expanduser("~/oxford/tools/sascore/sascorer.py"))
sascorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sascorer)

from rdkit.Chem import AllChem
#get_fingerprint= Chem.RDKFingerprint
#fingerprint_sim = DataStructs.FingerprintSimilarity

get_fingerprint= lambda mol: AllChem.GetMorganFingerprint(mol, 3)
fingerprint_sim = DataStructs.DiceSimilarity


def selecte_best_matches(input_smiles, list_reconstruction_attempts):
    n_mols = len(input_smiles)
    results = np.zeros( (n_mols, len(list_reconstruction_attempts) ) )
    
    ori_fps = [ get_fingerprint(Chem.MolFromSmiles(ori_smi)) for ori_smi in input_smiles]
    for a, reconstructed_smiles in enumerate(list_reconstruction_attempts):
        for j, (ori_fp, smi) in enumerate(zip(ori_fps, reconstructed_smiles)):
            mol2= Chem.MolFromSmiles(smi)
            if mol2 is None:
              similarity = -1
            else:
              similarity= fingerprint_sim(ori_fp, get_fingerprint(mol2))
            results[j,a] = similarity
    avg_sim = np.mean(results, axis=1)
    best_matches = np.argmax(results, axis=1).tolist()
    print(results)
    print(best_matches)
    best_sim = results[range(n_mols), best_matches]
    best_matches = [list_reconstruction_attempts[j][i] for i,j in zip(range(n_mols), best_matches) ]
    
    return avg_sim, best_sim, best_matches
    
def score_one_batch(input_smiles, reconstructed_smiles):
    r = requests.post(url, data={"smilesList": input_smiles}, headers=headers)
 
    if r.ok:
      postera_scores_ = [ 1-x["SAData"]["fastSAScore"] for x in r.json()["results"]]
    else:
      postera_scores_ = [-1]*len(reconstructed_smiles)
    print("AE_score RS_score SA_score SMI --> AE_SMI")
    for j, (ori_smi, smi) in enumerate(zip(input_smiles, reconstructed_smiles)):
        mol1= Chem.MolFromSmiles(ori_smi)
        mol2= Chem.MolFromSmiles(smi)
        if mol1 is None or mol2 is None:
          similarity = -1
          sa_score = -1
        else:
          similarity= fingerprint_sim(get_fingerprint(mol1), get_fingerprint(mol2))
          sa_score = 1 - sascorer.calculateScore( mol1 ) / 10 #is it 9 instead?
        print("%.3f\t: %.3f\t: %.3f : %s --> %s"%(similarity, postera_scores_[j], sa_score, ori_smi, smi))
        yield similarity, postera_scores_[j], sa_score


sa_scores = []
postera_scores = []
ae_scores = []
with torch.no_grad():
    for i in range(n_smiles // args.batch_size + int(bool(n_smiles % args.batch_size))):
        raw_batch = smiles[i*batch_size:(i+1)*batch_size]
        r = requests.post(url, data={"smilesList": raw_batch}, headers=headers)
        batch = tensorize(raw_batch, prepro_vocab)
        
        smiles_attempts_list = []
        print( raw_batch ); print("..........................")
        for j in range(50):
            smiles_list = model.reconstruct( batch, greedy=False )
            print(smiles_list)
            smiles_attempts_list.append( smiles_list )
        
        avg_sim, best_sim, best_matches = selecte_best_matches(raw_batch, smiles_attempts_list)
        smiles_list = best_matches
        
        similarity_postera_sa_list = score_one_batch(raw_batch, smiles_list)
        
        for k, (a,p,sa) in enumerate(similarity_postera_sa_list):
#            a = avg_sim[k]
            ae_scores.append( a )
            postera_scores.append( p )
            sa_scores.append( sa )
        print( "I=", i)
        if i == 200: break

from matplotlib import pyplot as plt
from scipy import stats

plt.hist(postera_scores, 50, alpha=0.5, label="postera"); plt.hist(sa_scores, 50, alpha=0.5, label="SAScore"); plt.hist(ae_scores, 50, alpha=0.5, label="AEScore"); plt.legend(); plt.show()

plt.scatter(postera_scores, sa_scores)
gradient, intercept, r_value, p_value, std_err = stats.linregress(postera_scores, sa_scores)
mn=np.min(postera_scores)
mx=np.max(postera_scores)
x1=np.linspace(mn,mx,500)
y1=gradient*x1+intercept
plt.plot(x1,y1,'-r', label="r=%f"%r_value)
plt.ylabel("sa_score")
plt.xlabel("postera_score")
plt.legend()
plt.show()


plt.scatter(postera_scores, ae_scores)
gradient, intercept, r_value, p_value, std_err = stats.linregress(postera_scores, ae_scores)
mn=np.min(postera_scores)
mx=np.max(postera_scores)
x1=np.linspace(mn,mx,500)
y1=gradient*x1+intercept
plt.plot(x1,y1,'-r', label="r=%f"%r_value)
plt.ylabel("ae_score")
plt.xlabel("postera_score")
plt.legend()
plt.show()


'''

echo -e "CCCOC\nIc1ccc2[nH]ncc2c1\nC1=CN=CC(NC(=O)N(CN2CCN(C)CC2)C2=CC=CC(Cl)=C2)=C1" | python anomaly_detection.py --vocab data/chembl/vocab.txt --model ckpt/chembl-pretrained/model.ckpt --query_smiles -

'''
