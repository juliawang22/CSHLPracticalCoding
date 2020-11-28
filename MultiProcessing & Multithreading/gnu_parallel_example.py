import numpy as np
import pandas as pd

from egad import run_egad
from rank import rank

from argparse import ArgumentParser

##Get iteration number from the command line
parser = ArgumentParser()
parser.add_argument('-i', type=int, help='ID for results',required=True)
args = parser.parse_args()


## Load list of datasets and permute the order of them
datasets = np.genfromtxt(
    '/data/bharris/biccn_paper/data/bulk_rna/datasets_used.csv', dtype=str)
np.random.shuffle(datasets) #Shuffle order of datasets

## Compute file paths for aggregation
networks_path = '/data/bharris/biccn_paper/data/bulk_rna/networks/'
file_names = [f'{networks_path}{ds}_pearson_nw.hdf5' for ds in datasets]
 
## Load in GO annotations
go = pd.read_hdf('~/GO_data/go_mouse_nw.hdf5', 'go')

##Create empty network to store aggregate in
genes = np.genfromtxt(
    '/data/bharris/biccn_paper/data/highly_expressed_7_datasets_75k.csv',
    dtype=str)
agg_nw = np.zeros([genes.shape[0],genes.shape[0]])

##Series to store result
results = pd.Series()
for ds, fn in zip(datasets, file_name):
    agg_nw += pd.read_hdf(fn,'nw').values
    agg_nw_copy = agg_nw.copy()
    rank(agg_nw_copy) #Ranking occurs inplace
    results[ds] = run_egad(go, 
                           pd.DataFrame(agg_nw_copy, 
                                            index=genes,
                                            columns=genes)).AUC.mean()

## Save output using ID
results.to_csv(f'permute_network_performance_{args.i}.csv')