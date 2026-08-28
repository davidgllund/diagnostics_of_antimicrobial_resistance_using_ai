#!/usr/bin/env python
import argparse
import numpy as np
import polars as pl
import torch

from collections import defaultdict
from tqdm import tqdm
from sys import argv
from dire.aux import Vocab

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Format input data and generate vocabulary file.')
    parser.add_argument('--input', '-i', required=True,
                        help='Input data file.')
    parser.add_argument('--output', '-o', required=True,
                        help='Name of output data file.')
    parser.add_argument('--vocab', required=True,
                        help='Name of output vocabulary file.')

    arguments = parser.parse_args(argv)
    
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
    rows = []
    for _, group in df.group_by(['lab_id', 'isolate_id'], maintain_order=True):
        first = group.row(0)
        record = {
            'year': first[0],
            'month': first[1],
            'age': first[2],
            'lab_id': first[3],
            'gender': first[4],
            'pathogen': first[5],
            'specimen': first[6],
            'antibiotic': group['antibiotic'].to_list(),
            'sir': group['sir'].to_list(),
        }
        ast_items = [f"{ab}_{res}" for ab, res in zip(record['antibiotic'], record['sir'])]
        sentence = ' '.join(map(str, [record['year'], record['month'], record['age'],
            record['lab_id'], record['gender'], record['pathogen'], 
            record['specimen'], *format_ast(pl.Series(ast_items))
        ]))
        rows.append(sentence)
    return rows

def count_antibiotics(sentence, n_metadata):
    n_ab = len(sentence.split(' ')) - n_metadata
    return n_ab

def main():
    arguments = parse_args(argv)
    data = pl.read_csv(arguments.input, separator='\t')

    data = data.with_columns(
        pl.col('date').str.strptime(pl.Datetime, strict=False).alias('date')
    )
    data = data.with_columns(
        pl.col('date').dt.year().cast(pl.String).alias('year'),
        pl.col('date').dt.month().cast(pl.String).alias('month')
    )

    print('Formatting data')
    sentence = format_data(data)

    fold = np.random.choice(['train', 'validation', 'calibration', 'test'], size=len(sentence), p=[0.4, 0.1, 0.4, 0.1])
    n_ab = [count_antibiotics(x, 7) for x in sentence]

    df_formatted = pl.DataFrame({'sentence': sentence, 'n_ab': n_ab, 'fold': fold})
    df_formatted.write_csv(arguments.output, separator='\t', include_header=True)

    print('Generating vocabulary')
    vocab = Vocab(tokens=list({x for lst in sentence for x in lst.split(' ')[3:]}), reserved_tokens=['<cls>','<pad>','<mask>','<sep>'])
    torch.save(vocab, arguments.vocab)

if __name__ == '__main__':
    main()