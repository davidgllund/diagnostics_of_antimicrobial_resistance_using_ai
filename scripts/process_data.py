#!/usr/bin/env python
import argparse
import torch
import pandas as pd 
import numpy as np
from collections import defaultdict
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

    grouped = defaultdict(list)
    for item in uniq_items:
            ab, res = item.split('_')
            grouped[ab].append(res)

    output = []
    for ab, results in grouped.items():
            if 'R' in results:
                final = 'R'
            elif 'I' in results:
                final = 'I'
            else:
                final = 'S'

            output.append(f"{ab}_{final}")

    return output

def format_data(df):
    tqdm.pandas()

    df_formatted = (
        df.groupby(['lab_id', 'isolate_id'])
        .progress_apply(
            lambda g: ' '.join(
                map(str, [g['year'].iloc[0], g['month'].iloc[0], g['age'].iloc[0],
                    g['lab_id'].iloc[0], g['gender'].iloc[0], g['pathogen'].iloc[0], 
                    g['specimen'].iloc[0], *format_ast(g['antibiotic'] + '_' + g['sir'])
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

    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['year'] = data['date'].dt.year.astype(str)
    data['month'] = data['date'].dt.month.astype(str)

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