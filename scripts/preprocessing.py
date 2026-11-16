#!/usr/bin/env python
import polars as pl
import pyreadr

from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path('/home/dlund/dire/data/svebar')
INDEX_DIR = Path('/home/dlund/dire/auxiliary_files')
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
        dfs.append(pl.from_pandas(result['test_data']))
    return pl.concat(dfs, how='vertical_relaxed')


def load_lookup_table(path):
    df = pl.read_csv(path, separator='\t', has_header=False)
    mapping = dict(zip(df['column_1'].to_list(), df['column_2'].to_list()))
    return mapping


def load_antibiotic_class_map(path):
    df = pl.read_csv(path, separator='\t', has_header=False)
    mapping = dict(zip(df['column_2'].to_list(), df['column_3'].to_list()))
    return mapping


def prefilter_dataset(df, taxonomy):
    df = df.filter(
        (pl.col('smittamne_namn') != 'NEGATIV') &
        pl.col('sir').is_not_null() &
        (~pl.col('patient_fodelse').is_in(['Okänd', 'Okï¿½n'])) &
        (pl.col('antibiotikum_namn') != 'Betalaktamas') &
        (pl.col('antibiotikum_namn') != 'ResNet')
    )

    df = df.fill_null('<unk>')
    genus = pl.col('smittamne_namn').str.split(' ').list.get(0)
    species = pl.col('smittamne_namn').str.split(' ').list.get(1)
    df = df.filter(genus.is_in(taxonomy))
    df = df.filter(species != 'art')
    return df


def format_dataset(df, antibiotic_map, species_map, sample_map):
    df = df.with_columns(
        pl.col('provtagnings_datum')
        .cast(pl.String, strict=False)
        .str.strptime(pl.Datetime, strict=False)
        .alias('provtagnings_datum'),
        (pl.col('provtagnings_datum').dt.year() - pl.col('patient_fodelse').cast(pl.String).str.slice(0, 4).cast(pl.Int64, strict=False)).alias("patient_alder"),
        pl.col('patient_kon').map_elements(
            lambda x: {1: 'M', 2: 'F', 3: '<unk>'}.get(x, '<unk>'),
            return_dtype=pl.String,
        ).alias('patient_kon')
    )

    df = df.with_columns(
        pl.when(pl.col('smittamne_namn') == 'EHEC (Enterohemorragisk E. coli)')
        .then(pl.lit('Escherichia coli'))
        .otherwise(pl.col('smittamne_namn'))
        .alias('smittamne_namn'),
        pl.when(pl.col('labb_kod').is_in(['SE110', 'SE230']))
        .then(pl.lit('SE340'))
        .otherwise(pl.col('labb_kod'))
        .alias('labb_kod'),
        pl.col('provtagnings_datum').dt.strftime('%Y-%m-%d').alias('provtagnings_datum'),
        pl.col('antibiotikum_namn').str.replace(r'\s*\(.*\)', '', literal=False).alias('antibiotikum_namn')
    )

    df = df.with_columns(
        pl.col('antibiotikum_namn').map_elements(lambda x: antibiotic_map.get(x, x), return_dtype=pl.String).alias('antibiotic'),
        pl.col('smittamne_namn').map_elements(lambda x: species_map.get(x, x), return_dtype=pl.String).alias('pathogen'),
        pl.col('provmaterial_namn').map_elements(lambda x: sample_map.get(x, x), return_dtype=pl.String).alias('specimen'),
    )

    df = df.filter(pl.col('specimen') != 'IGNORE')
    df = df.rename({
        'provtagnings_datum': 'date',
        'labb_kod': 'lab_id',
        'patient_kon': 'gender',
        'patient_alder': 'age',
        'id': 'isolate_id'
    })
    return df.select([
        'date', 'lab_id', 'specimen', 'gender', 'age', 'isolate_id', 'pathogen', 'antibiotic', 'sir'
    ])


def filter_ab_sp_combinations(df, antibiotic_class_map):
    df = df.clone()

    ab_keep_df = (
        df.with_columns(
            (pl.col('sir') == 'S').cast(pl.Int8).alias('isS'),
            (pl.col('sir') == 'I').cast(pl.Int8).alias('isI'),
            (pl.col('sir') == 'R').cast(pl.Int8).alias('isR'),
        )
        .group_by(['pathogen', 'antibiotic'])
        .agg(
            pl.col('isS').mean().alias('S_rate'),
            pl.col('isI').mean().alias('I_rate'),
            pl.col('isR').mean().alias('R_rate'),
            pl.col('isS').sum().alias('S_count'),
            pl.col('isI').sum().alias('I_count'),
            pl.col('isR').sum().alias('R_count'),
            pl.len().alias('n')
        )
        .filter((pl.col('S_rate') >= 0.05) & (pl.col('S_count') >= 500) & (pl.col('R_rate') >= 0.05) & (pl.col('R_count') >= 500))
    )

    df = df.join(ab_keep_df.select(['pathogen', 'antibiotic']), on=['pathogen', 'antibiotic'], how='inner')

    top_sp = (
        df.select(['lab_id', 'isolate_id', 'pathogen']).unique().group_by('pathogen').len().filter(pl.col('len') >= 10000)
    )
    df = df.join(top_sp.select('pathogen'), on='pathogen', how='inner')

    valid_i_pairs = {(row[0], row[1]) for row in ab_keep_df.filter((pl.col('I_rate') >= 0.05) & (pl.col('I_count') >= 500)).select(['pathogen', 'antibiotic']).iter_rows()}
    rows = []
    for row in tqdm(df.iter_rows(named=True), total=df.height, desc='Filtering I-rate pairs'):
        if row['sir'] == 'I' and (row['pathogen'], row['antibiotic']) not in valid_i_pairs:
            continue
        rows.append(row)
    df = pl.DataFrame(rows)

    final_sp = (
        df.with_columns(
            pl.col('antibiotic').map_elements(lambda x: antibiotic_class_map.get(x, x), return_dtype=pl.String).alias('antibiotic_class')
        )
        .select(['pathogen', 'antibiotic_class'])
        .unique()
        .group_by('pathogen')
        .len()
        .filter(pl.col('len') >= 3)
    )
    df = df.join(final_sp.select('pathogen'), on='pathogen', how='inner')
    return df


def main():
    print(f'Reading Rdata files from {DATA_DIR}...')
    svebar = read_rdata(DATA_DIR, RDATA_FILES)
    print(f'  -> {svebar.height} rows loaded')

    taxonomy = set(pl.read_csv(TAXONOMY_FILE, separator='\t', columns=['genus']).get_column('genus').to_list())

    antibiotic_map = load_lookup_table(INDEX_DIR / 'antibiotic_abbrev.tsv')
    species_map = load_lookup_table(INDEX_DIR / 'species_abbrev.tsv')
    sample_map = load_lookup_table(INDEX_DIR / 'sample_categ.tsv')
    antibiotic_class_map = load_antibiotic_class_map(INDEX_DIR / 'antibiotic_abbrev.tsv')

    print('Prefiltering dataset...')
    svebar = prefilter_dataset(svebar, taxonomy)
    print(f'  -> {svebar.height} rows remaining')

    print('Formatting dataset...')
    svebar = format_dataset(svebar, antibiotic_map, species_map, sample_map)
    print(f'  -> {svebar.height} rows remaining')

    print('Filtering antibiotic/species combinations...')
    svebar = filter_ab_sp_combinations(svebar, antibiotic_class_map)
    print(f'  -> {svebar.height} rows remaining')

    print(f'Writing output to {OUTPUT_FILE}...')
    svebar.write_csv(OUTPUT_FILE, separator='\t')
    print('Done.')

if __name__ == '__main__':
    main()