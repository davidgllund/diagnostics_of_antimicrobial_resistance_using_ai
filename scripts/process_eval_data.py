#!/usr/bin/env python
import argparse
import torch
import polars as pl
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from sys import argv

from dire.aux import Vocab

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

    arguments = parser.parse_args(argv)
    
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

    df = df.with_columns(
        pl.col('antibiotikum_namn').map_elements(lambda x: ab_dict.get(x, x), return_dtype=pl.String).alias('antibiotic_short'),
        pl.col('smittamne_namn').map_elements(lambda x: sp_dict.get(x, x), return_dtype=pl.String).alias('species_short'),
        pl.col('provmaterial_namn').map_elements(lambda x: sample_dict.get(x, x), return_dtype=pl.String).alias('sample')
    )

    df = df.with_columns(
        pl.col('provtagnings_datum').str.strptime(pl.Datetime, strict=False).alias('provtagnings_datum')
    )
    df = df.with_columns(
        pl.col('provtagnings_datum').dt.year().cast(pl.String).alias('year'),
        pl.col('provtagnings_datum').dt.month().cast(pl.String).alias('month'),
        pl.col('patient_alder').cast(pl.String).map_elements(lambda x: x.split('.')[0] if x is not None else x, return_dtype=pl.String).alias('patient_alder'),
        pl.col('patient_kon').cast(pl.String).alias('patient_kon')
    )

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
    rows = []
    for _, group in df.group_by('id', maintain_order=True):
        first = group.row(0)
        row_map = dict(zip(group.columns, first))
        ast_values = [f"{ab}_{result}" for ab, result in zip(group['antibiotic_short'].to_list(), group['sir'].to_list())]
        sentence = ' '.join([
            row_map['year'], row_map['month'], row_map['patient_alder'],
            row_map['labb_kod'], row_map['patient_kon'], row_map['species_short'],
            row_map['sample'], *format_ast(pl.Series(ast_values))
        ])
        rows.append(sentence)
    return rows

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
    data = pl.read_csv(arguments.input, separator='\t')

    print('Preprocessing data')
    df = preprocess_data(data, arguments.ab_abbrev, arguments.sp_abbrev, arguments.sample_categ)

    print('Formatting data')
    sentence = format_data(df)

    labels = ['test']
    fold = generate_balanced_labels(len(sentence), labels)
    n_ab = [count_antibiotics(x, 7) for x in sentence]

    df_formatted = pl.DataFrame({'sentence': sentence, 'n_ab': n_ab, 'fold': fold})
    df_formatted.write_csv(arguments.output, separator='\t', include_header=True)

    print('Generating vocabulary')
    vocab = Vocab(tokens=list({x for lst in sentence for x in lst.split(' ')[3:]}), reserved_tokens=['<cls>','<pad>','<mask>','<sep>'])
    torch.save(vocab, arguments.vocab)

if __name__ == '__main__':
    main()