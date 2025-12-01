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
    parser.add_argument('--ab_abbrev', required=True,
                        help='File containing antibiotic abrreviations.')
    parser.add_argument('--sp_abbrev', required=True,
                        help='File containing bacterial species abrreviations.')
    parser.add_argument('--sample_categ', required=True,
                        help='File containing types of sample and their respective categories.')
    parser.add_argument('--output', '-o', required=True,
                        help='Name of output data file.')
    parser.add_argument('--vocab', required=True,
                        help='Name of output vocabulary file.')

    arguments = parser.parse_args()

    return arguments

def read_dict(filename):
    dictionary = {}
    with open(filename, 'r') as f:
        for line in f:
            items = line.split('\t')
            dictionary[items[0]] = items[1].strip()      
    return dictionary

def preprocess_data(df, ab_abbrev, sp_abbrev, sample_categ): 
    ab_dict = read_dict(ab_abbrev)
    sp_dict = read_dict(sp_abbrev)
    sample_dict = read_dict(sample_categ)

    #df = df[df['antibiotikum_namn'].isin(ab_dict.keys())]
    #df = df[df['smittamne_namn'].isin(sp_dict.keys())]
    #df = df[df['provmaterial_namn'].isin(sample_dict.keys())]

    df['antibiotic_short'] = [ab_dict[x] if x in ab_dict.keys() else x for x in df['antibiotikum_namn']]
    df['species_short'] = [sp_dict[x] if x in sp_dict.keys() else x for x in df['smittamne_namn']]
    df['sample'] = [sample_dict[x] if x in sample_dict.keys() else x for x in df['provmaterial_namn']]

    df['provtagnings_datum'] = pd.to_datetime(df['provtagnings_datum'], errors='coerce')

    df['year'] = df['provtagnings_datum'].dt.year.astype(str)
    df['month'] = df['provtagnings_datum'].dt.month.astype(str)

    df['patient_alder'] = [x.split('.')[0] for x in df['patient_alder'].astype(str)]
    df['patient_kon'] = df['patient_kon'].astype(str)

    return df

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
        df.groupby('id')
        .progress_apply(
            lambda g: ' '.join(
                [g['year'].iloc[0], g['month'].iloc[0], g['patient_alder'].iloc[0], 
                 g['labb_kod'].iloc[0], g['patient_kon'].iloc[0], g['species_short'].iloc[0], 
                 g['sample'].iloc[0], *format_ast(g['antibiotic_short'] + '_' + g['sir'])]
            )
        )
        .tolist()
    )

    return df_formatted

def generate_balanced_labels(length, labels):
    labels = list(labels)
    n_labels = len(labels)

    base_count = length // n_labels
    remainder = length % n_labels

    balanced = []
    for i, label in enumerate(labels):
        count = base_count + (1 if i < remainder else 0)
        balanced.extend([label] * count)

    random.shuffle(balanced)
    return balanced

def count_antibiotics(sentence, n_metadata):
    n_ab = len(sentence.split(' ')) - n_metadata
    return n_ab

def main():
    arguments = parse_args(argv)
    data = pd.read_csv(arguments.input, sep='\t')
    data.replace(np.nan, '<unk>', inplace=True)

    print('Preprocessing data')
    df = preprocess_data(data, arguments.ab_abbrev, arguments.sp_abbrev, arguments.sample_categ)

    print('Formatting data')
    sentence = format_data(df)

    labels = ['1', '2', '3', '4', '5', 'test', 'val']
    fold = generate_balanced_labels(len(sentence), labels)
    n_ab = [count_antibiotics(x, 7) for x in sentence]

    df_formatted = pd.DataFrame({'sentence': sentence, 'n_ab': n_ab, 'fold': fold})
    df_formatted.to_csv(arguments.output, sep='\t', index=False, header=True)

    print('Generating vocabulary')
    vocab = Vocab(tokens=list({x for lst in sentence for x in lst.split(' ')[3:]}), reserved_tokens=['<cls>','<pad>','<mask>','<sep>'])
    torch.save(vocab, arguments.vocab)

if __name__ == '__main__':
    main()