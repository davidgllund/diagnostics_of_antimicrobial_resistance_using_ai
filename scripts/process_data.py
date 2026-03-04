#!/usr/bin/env python
import argparse
import torch
import pandas as pd
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from sys import argv

from aux import Vocab

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Format input data and generate vocabulary file.')
    parser.add_argument('--input', '-i', required=True,
                        help='Input data file.')
    parser.add_argument('--output', '-o', required=True,
                        help='Name of output data file.')
    parser.add_argument('--vocab', required=True,
                        help='Name of output vocabulary file.')

    arguments = parser.parse_args()
    
    return arguments

def format_ast(lst):
    uniq_items = lst.unique()

    ab = [x.split('_')[0] for x in uniq_items]
    result = [x.split('_')[1] for x in uniq_items]

    if len(set(ab)) == len(uniq_items):
        return list(uniq_items)
    
    else:
        inconclusive_ab = list({k: v for k, v in Counter(ab).items() if v != 1}.keys())
        index = [i for i, e in enumerate(ab) if e in inconclusive_ab]
        
        for idx in index:
            result[idx] = 'R'

        output = list(set(['_'.join([x, y]) for x, y in zip(ab, result)]))

        return output

def format_data(df):
    tqdm.pandas()

    df_formatted = (
        df.groupby(['LabID', 'IsolateID'])
        .progress_apply(
            lambda g: ' '.join(
                map(str, [g['Year'].iloc[0], g['Month'].iloc[0], g['Age'].iloc[0],
                    g['LabID'].iloc[0], g['Gender'].iloc[0], g['Pathogen'].iloc[0], 
                    g['Specimen'].iloc[0], *format_ast(g['Antibiotic'] + '_' + g['SIR'])
                ])
            )
        )
        .tolist()
    )

    return df_formatted

def count_antibiotics(sentence, n_metadata):
    n_ab = len(sentence.split(' ')) - n_metadata
    return n_ab

def main():
    arguments = parse_args(argv)
    data = pd.read_csv(arguments.input, sep='\t')

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Year'] = data['Date'].dt.year.astype(str)
    data['Month'] = data['Date'].dt.month.astype(str)

    print('Formatting data')
    sentence = format_data(data)

    fold = np.random.choice(['train', 'validation', 'calibration', 'test'], size=len(sentence), p=[0.4, 0.1, 0.4, 0.1])
    n_ab = [count_antibiotics(x, 7) for x in sentence]

    df_formatted = pd.DataFrame({'sentence': sentence, 'n_ab': n_ab, 'fold': fold})
    df_formatted.to_csv(arguments.output, sep='\t', index=False, header=True)

    print('Generating vocabulary')
    vocab = Vocab(tokens=list({x for lst in sentence for x in lst.split(' ')[3:]}), reserved_tokens=['<cls>','<pad>','<mask>','<sep>'])
    torch.save(vocab, arguments.vocab)

if __name__ == '__main__':
    main()