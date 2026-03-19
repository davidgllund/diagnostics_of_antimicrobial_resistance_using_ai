#!/usr/bin/env python
import pandas as pd
import numpy as np
from pathlib import Path
import pyreadr

DATA_DIR = Path('/home/dlund/dire/data/svebar')
INDEX_DIR = Path('/home/dlund/dire/index_files')
TAXONOMY_FILE = Path('/storage_i/dlund/ncbi_taxonomy/bacterial_taxonomy_table.tsv')

RDATA_FILES = [
    'SE100.Rdata',
    'SE110.Rdata',
    'SE230.Rdata',
    'SE300.Rdata',
]

OUTPUT_FILE = DATA_DIR / 'svebar_filtered.tsv'


def read_rdata(data_dir, filenames):
    dfs = []
    for fname in filenames:
        result = pyreadr.read_r(str(data_dir / fname))
        dfs.append(result['test_data'])

    return pd.concat(dfs, ignore_index=True)

def load_lookup_table(path):
    return (
        pd.read_csv(path, header=None, sep="\t")
        .set_index(0)[1]
        .to_dict()
    )

def prefilter_dataset(df, taxonomy):
    df = df.copy()

    df = df[
        (df['smittamne_namn'] != 'NEGATIV') &
        (df['sir'].notna()) &
        (~df['patient_fodelse'].isin(['Okänd', 'Okï¿½n'])) &
        (df['antibiotikum_namn'] != 'Betalaktamas') &
        (df['antibiotikum_namn'] != 'ResNet')
    ]

    df = df.fillna('<unk>')
    genus = df['smittamne_namn'].str.split(" ", n=1).str[0]
    df = df[genus.isin(taxonomy)]

    species = df['smittamne_namn'].str.split(" ", n=1).str[1]
    df = df[species != "art"]

    return df

def format_dataset(df, antibiotic_map, species_map, sample_map):
    df = df.copy()

    df['provtagnings_datum'] = pd.to_datetime(
        df['provtagnings_datum'],
        errors='coerce'
    )

    year_sample = df['provtagnings_datum'].dt.year
    year_birth = pd.to_numeric(
        df['patient_fodelse'].astype(str).str[:4],
        errors='coerce'
    )

    df['patient_alder'] = year_sample - year_birth
    
    df['patient_kon'] = df['patient_kon'].map({
        1: 'M',
        2: 'F',
        3: '<unk>'
    }).fillna('<unk>')

    df['smittamne_namn'] = df['smittamne_namn'].replace({
        'EHEC (Enterohemorragisk E. coli)': 'Escherichia coli'
    })

    df['provtagnings_datum'] = df['provtagnings_datum'].dt.strftime("%Y-%m-%d")
    df['antibiotikum_namn'] = df['antibiotikum_namn'].str.replace(r"\s*\(.*\)", "", regex=True)
    df['antibiotic'] = df['antibiotikum_namn'].map(antibiotic_map)
    df['pathogen'] = df['smittamne_namn'].map(species_map)
    df['specimen'] = df['provmaterial_namn'].map(sample_map)

    df = df.loc[df['specimen'] != 'IGNORE'] 

    df = df.rename(columns={
        'provtagnings_datum': 'date',
        'labb_kod': 'lab_id',
        'patient_kon': 'gender',
        'patient_alder': 'age',
        'id': 'isolate_id'
    })[
        ['date', 'lab_id', 'specimen', 'gender', 'age', 
         'isolate_id', 'pathogen', 'antibiotic', 'sir']
    ]

    return df

def filter_ab_sp_combinations(df):
    df = df.copy()
    
    # Extract species  with sufficient data
    top_sp = (
        df[['lab_id', 'isolate_id', 'pathogen']]
        .drop_duplicates()
        .groupby('pathogen')
        .size()
        .reset_index(name='Freq')
    )

    top_sp = top_sp[top_sp['Freq'] > 10000]
    df = df.loc[df['pathogen'].isin(top_sp['pathogen'])]

    # Extract antibiotic-species combinations with sufficient data
    ab_keep_df = (
        df
        .assign(isS = df['sir'] == 'S')
        .assign(isI = df['sir'] == 'I')
        .assign(isR = df['sir'] == 'R')
        .groupby(['pathogen', 'antibiotic'])
        .agg(I_rate=('isI', 'mean'),
            S_rate=('isS', 'mean'), 
            R_rate=('isR', 'mean'), 
            n=('sir', 'size'))
        .reset_index()
    )

    ab_keep_df = ab_keep_df[
        (ab_keep_df['S_rate'] >= 0.05) &
        (ab_keep_df['R_rate'] >= 0.05) &
        (ab_keep_df['n'] >= 1000)
    ]

    df = df.merge(
        ab_keep_df[['pathogen', 'antibiotic']],
        on=['pathogen', 'antibiotic'],
        how='inner'
    )

    # In relevant cases, keep tests where the result is "I"
    tmp = ab_keep_df[ab_keep_df['I_rate'] >= 0.05]
    valid_i_pairs = set(zip(tmp['pathogen'], tmp['antibiotic']))
    df = df[
        ~((df['sir'] == 'I') & 
        (~df[['pathogen', 'antibiotic']].apply(tuple, axis=1).isin(valid_i_pairs)))
    ]

    return df

def main():
    # Load data
    svebar = read_rdata(DATA_DIR, RDATA_FILES)
    
   # Load auxiliary files
    taxonomy = set(
        pd.read_csv(TAXONOMY_FILE, sep="\t", usecols=['genus'])['genus']
    )

    antibiotic_map = load_lookup_table(INDEX_DIR / 'ab_abbrev.tsv')
    species_map = load_lookup_table(INDEX_DIR / 'species_abbrev.tsv')
    sample_map = load_lookup_table(INDEX_DIR / 'sample_categ.tsv')
    
    # Prefiltering
    svebar = prefilter_dataset(svebar, taxonomy)

    # Format relevant columns
    svebar = format_dataset(svebar, antibiotic_map, species_map, sample_map)

    # Keep only antibiotic-species combinations with sufficent data
    svebar = filter_ab_sp_combinations(svebar)

    # Write output
    svebar.to_csv(OUTPUT_FILE, sep='\t', index=False)

if __name__ == '__main__':
    main()