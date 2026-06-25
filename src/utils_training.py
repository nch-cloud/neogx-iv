import os
import pandas as pd
import numpy as np
import re
from datetime import datetime

FEATURE_DIR = '../data/feature_matrices'

## FUNCTIONS FOR GRID / RANDOM SEARCH
## FUNCTIONS FOR HANDLING SEARCH RESULTS

def format_search_results(grid_df, config, target_var_name):
    # sort_metric = 'mean_test_pr_auc'
    param_cols = [c for c in grid_df.columns if 'param_' in c]
    score_cols = [c for c in grid_df.columns if 'mean_test' in c]

    grid_df['clf_name'] = grid_df['param_clf'].str.split(r'\(').str[0]
    other_cols = ['n_features', 'clf_name']
    grid_df[score_cols + param_cols + other_cols]
    #grid_df = grid_df.sort_values('mean_test_pr_auc', ascending=False)

    for prefix, subconfig in config.items():
        for key, val in subconfig.items():
            #grid_df[f"{prefix}_feature__{key}"] = val
            grid_df[f"config__{prefix}__{key}"] = val

    #grid_df['target_var_name'] = target_var_name
    

    return grid_df

def save_formatted_grid_df(grid_df, result_dir):
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_')

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    config_lookup = dict(
        grid_df[
            [c for c in grid_df.columns if c.startswith('config__')]
            ].iloc[0])

    config_tags = [
        f"{key.split('__')[-1]}={val}"
        for key, val in config_lookup.items()
        if 'config__pheno' in key
    ]

    search_fname = (
        'search_results--'
        + formatted_time + '--'
        + '-'.join(config_tags)
        + '.csv.gz'
    )

    grid_df.to_csv(
        os.path.join(result_dir, search_fname),
        compression='gzip'
    )

## FUNCTIONS FOR FEATURE MATRIX CONFIGURATION / SELECTION / RETRIEVAL

def get_pheno_filename(
    config,
    feature_dir=FEATURE_DIR,
):
    pheno_config = dict(config['pheno'])
    if pheno_config.get('vocab', None) not in ['hpo', 'phecode']:
        raise ValueError(f"invalid vocab: {pheno_config['vocab']}")

    if pheno_config['vocab'] == 'hpo':
        if 'dx_type' in pheno_config:
            del pheno_config['dx_type']

    if pheno_config['vocab'] == 'phecode':
        if 'note_filter' in pheno_config:
            del pheno_config['note_filter']

    pheno_fnames = os.listdir(feature_dir)

    for attr, val in pheno_config.items():
        prev_names = list(pheno_fnames)
        pheno_fnames = [
            fname for fname in pheno_fnames
            if (f"{attr}={val}-" in fname or f"{attr}={val}." in fname)
        ]
        if len(pheno_fnames) == 0:
            raise ValueError(
                'No pheno matrix fitting configuration\n'
                + f'latest restriction: {attr} = {val}\n'
                + 'names at previous step:\n'
                + '\n'.join(prev_names)
            )
    if len(pheno_fnames) > 1:
        raise ValueError(
            "Underspecified, too many candidates:\n"
            + '\n'.join(pheno_fnames)
        )
    else:
        return pheno_fnames[0]

def build_feature_matrix(
    config,
    feature_dir=FEATURE_DIR
):
    static_config = config['static']
    pheno_config = config['pheno']

    static_path = os.path.join(feature_dir, 'static_features.csv.gz')
    static_df = pd.read_csv(static_path, index_col='pat_id')
    for col in static_df.columns:
        if static_config.get(col, True):
            pass
        else:
            del static_df[col]

    if pheno_config['vocab'] in ['hpo', 'phecode']:
        pheno_fname = get_pheno_filename(config, feature_dir=feature_dir)
        pheno_df = pd.read_csv(
            os.path.join(feature_dir, pheno_fname),
            index_col='pat_id'
        )

    elif pheno_config['vocab'] == 'hpo+phecode':
        hpo_config = dict(pheno_config)
        hpo_config['vocab'] = 'hpo'
        hpo_config['representatives'] = hpo_config['representatives_hpo']
        hpo_fname = get_pheno_filename(
            {'static': static_config, 'pheno': hpo_config}
        )
        pheno_df_hpo = pd.read_csv(
            os.path.join(feature_dir, hpo_fname),
            index_col='pat_id'
        )

        phe_config = dict(pheno_config)
        phe_config['vocab'] = 'phecode'
        phe_config['representatives'] = phe_config['representatives_phecode']
        phe_fname = get_pheno_filename(
            {'static': static_config, 'pheno': phe_config}
        )
        pheno_df_phe = pd.read_csv(
            os.path.join(feature_dir, phe_fname),
            index_col='pat_id'
        )

        pheno_df = pd.concat([pheno_df_hpo, pheno_df_phe], axis=1)

    if pheno_config['encoding'] == 'binary':
        pheno_df = 1*(pheno_df > 0)

    return pd.concat([static_df, pheno_df.fillna(0)], axis=1)


def get_config_from_grid(grid_df):
    grid_config = dict()
    for c in grid_df.columns:
        if (
            c.startswith('config__')
            and grid_df[c].nunique() == 0
           ):
            grid_df[c] = 'None'

    # get distinct config__ prefixes
    config_prefixes = [c.split('__')[1] for c in grid_df.columns if c.startswith('config__')]
    #for prefix in ['static', 'pheno']:
    for prefix in config_prefixes:
        for c in grid_df.columns:
            grid_config[prefix] = dict(
            grid_df
            [[
                c for c in grid_df.columns
                if c.startswith(f'config__{prefix}__')
            ]]
            .rename(columns=lambda c: c.replace(f'config__{prefix}__', ''))
            .iloc[0]
        )
    return grid_config

def is_completed(config, search_dir):
    done_configs = []
    for fname in os.listdir(search_dir):
        if not fname.endswith('.csv.gz'):
            continue
        done_configs.append(
            get_config_from_grid(pd.read_csv(os.path.join(search_dir, fname)))
        )
    return (config in done_configs)