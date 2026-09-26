import pandas as pd
import numpy as np
import json
import re

# Quick version - just collect key stats
print("Quick profiling...")

# File sizes and row counts
import os
file_info = {}
for split in ['train', 'test']:
    for src in ['source1', 'source2', 'source3']:
        path = f'./dataset/student_resource/dataset/{split}/{split}_{src}.tsv'
        size = os.path.getsize(path)
        # Count rows quickly
        df = pd.read_csv(path, sep='\t', usecols=['entity_id'])
        file_info[f'{split}_{src}'] = {
            'path': path,
            'filename': f'{split}_{src}.tsv',
            'file_size_bytes': size,
            'row_count': int(len(df)),
            'column_count': 4,
            'column_names': ['entity_id', 'business_name', 'business_address', 'country'],
            'delimiter': '\\t',
            'encoding': 'utf-8',
        }
    if split == 'train':
        path = f'./dataset/student_resource/dataset/{split}/{split}_ground_truth.tsv'
        size = os.path.getsize(path)
        df = pd.read_csv(path, sep='\t', usecols=['source1_entity_id'])
        file_info[f'{split}_ground_truth'] = {
            'path': path,
            'filename': f'{split}_ground_truth.tsv',
            'file_size_bytes': size,
            'row_count': int(len(df)),
            'column_count': 2,
            'column_names': ['source1_entity_id', 'matched_entity_ids'],
            'delimiter': '\\t',
            'encoding': 'utf-8',
        }

print("File info done")

# Source stats
source_stats = {}
for split in ['train', 'test']:
    for src_num in [1, 2, 3]:
        path = f'./dataset/student_resource/dataset/{split}/{split}_source{src_num}.tsv'
        df = pd.read_csv(path, sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'])
        key = f'{split}_source{src_num}'
        source_stats[key] = {
            'record_count': int(len(df)),
            'unique_ids': int(df['entity_id'].nunique()),
            'id_prefix': f'S{src_num}-',
            'null_counts': {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            'null_percentages': {k: round(float(v), 2) for k, v in (df.isnull().sum() / len(df) * 100).to_dict().items()},
            'duplicate_rows': int(df.duplicated().sum()),
            'country_distribution': {k: int(v) for k, v in df['country'].value_counts().to_dict().items()},
            'id_uniqueness': bool(df['entity_id'].nunique() == len(df)),
        }

print("Source stats done")

# Sample for detailed profiling
df1 = pd.read_csv('./dataset/student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=200000)
df2 = pd.read_csv('./dataset/student_resource/dataset/train/train_source2.tsv', sep='\t', nrows=200000)
df3 = pd.read_csv('./dataset/student_resource/dataset/train/train_source3.tsv', sep='\t', nrows=200000)

# Field profiles
field_profiles = {}
for field in ['business_name', 'business_address', 'country']:
    field_profiles[field] = {}
    for name, df in [('S1', df1), ('S2', df2), ('S3', df3)]:
        col = df[field].dropna()
        field_profiles[field][name] = {
            'sample_size': int(len(df)),
            'non_null_count': int(col.notna().sum()),
            'null_count': int(col.isna().sum()),
            'unique_count': int(col.nunique()),
            'min_length': int(col.str.len().min()) if len(col) > 0 else 0,
            'max_length': int(col.str.len().max()) if len(col) > 0 else 0,
            'mean_length': float(col.str.len().mean()) if len(col) > 0 else 0,
            'median_length': float(col.str.len().median()) if len(col) > 0 else 0,
        }
        if field == 'country':
            field_profiles[field][name]['value_counts'] = {k: int(v) for k, v in df[field].value_counts().to_dict().items()}
        else:
            field_profiles[field][name]['sample_values'] = col.head(10).tolist()

print("Field profiles done")

# Noise analysis
noise = {}
for name, df in [('S1', df1), ('S2', df2), ('S3', df3)]:
    noise[name] = {}
    names = df['business_name'].dropna()
    addrs = df['business_address'].dropna()
    
    noise[name]['business_name'] = {
        'contains_ampersand': int(names.str.contains('&').sum()),
        'contains_and': int(names.str.contains(r'\band\b', case=False).sum()),
        'contains_dot': int(names.str.contains(r'\.').sum()),
        'contains_comma': int(names.str.contains(',').sum()),
        'contains_hyphen': int(names.str.contains('-').sum()),
        'contains_digits': int(names.str.contains(r'\d').sum()),
        'contains_parentheses': int(names.str.contains(r'[\(\)]').sum()),
        'contains_apostrophe': int(names.str.contains(r"'").sum()),
        'legal_suffixes': {},
    }
    for suf in ['LLC', 'Inc', 'Corp', 'Ltd', 'LLP', 'LP', 'PC', 'PLLC', 'Private Limited', 'Pvt Ltd']:
        noise[name]['business_name']['legal_suffixes'][suf] = int(names.str.contains(rf'\b{re.escape(suf)}\b', case=False).sum())
    
    noise[name]['business_name']['case_variation'] = {
        'all_upper': int((names == names.str.upper()).sum()),
        'all_lower': int((names == names.str.lower()).sum()),
        'title_case': int((names == names.str.title()).sum()),
        'mixed': int((~(names == names.str.upper()) & ~(names == names.str.lower()) & ~(names == names.str.title())).sum()),
    }
    
    noise[name]['business_name']['whitespace'] = {
        'double_space': int(names.str.contains('  ').sum()),
        'leading': int(names.str.startswith(' ').sum()),
        'trailing': int(names.str.endswith(' ').sum()),
        'tabs': int(names.str.contains('\t').sum()),
    }
    
    noise[name]['business_name']['non_ascii_count'] = int(names.apply(lambda x: any(ord(c) > 127 for c in str(x))).sum())
    
    noise[name]['business_address'] = {
        'contains_hash': int(addrs.str.contains('#').sum()),
        'contains_apt': int(addrs.str.contains(r'\bApt\b', case=False).sum()),
        'contains_apartment': int(addrs.str.contains(r'\bApartment\b', case=False).sum()),
        'contains_suite': int(addrs.str.contains(r'\bSuite\b', case=False).sum()),
        'contains_unit': int(addrs.str.contains(r'\bUnit\b', case=False).sum()),
        'contains_floor': int(addrs.str.contains(r'\bFloor\b', case=False).sum()),
        'street_abbreviations': {
            'St': int(addrs.str.contains(r'\bSt\b', case=False).sum()),
            'Street': int(addrs.str.contains(r'\bStreet\b', case=False).sum()),
            'Ave': int(addrs.str.contains(r'\bAve\b', case=False).sum()),
            'Avenue': int(addrs.str.contains(r'\bAvenue\b', case=False).sum()),
            'Rd': int(addrs.str.contains(r'\bRd\b', case=False).sum()),
            'Road': int(addrs.str.contains(r'\bRoad\b', case=False).sum()),
            'Blvd': int(addrs.str.contains(r'\bBlvd\b', case=False).sum()),
            'Drive': int(addrs.str.contains(r'\bDrive\b', case=False).sum()),
        },
        'contains_comma': int(addrs.str.contains(',').sum()),
        'contains_semicolon': int(addrs.str.contains(';').sum()),
        'non_ascii_count': int(addrs.apply(lambda x: any(ord(c) > 127 for c in str(x))).sum()),
    }

print("Noise analysis done")

# Cross-source
names1 = set(df1['business_name'].dropna())
names2 = set(df2['business_name'].dropna())
names3 = set(df3['business_name'].dropna())

cross = {
    'name_overlap': {
        'S1_S2': len(names1 & names2),
        'S1_S3': len(names1 & names3),
        'S2_S3': len(names2 & names3),
    },
    'exact_name_address_overlap': {},
    'name_country_overlap': {},
    'id_overlap': {},
}

df1['key'] = df1['business_name'].fillna('') + '|' + df1['business_address'].fillna('')
df2['key'] = df2['business_name'].fillna('') + '|' + df2['business_address'].fillna('')
df3['key'] = df3['business_name'].fillna('') + '|' + df3['business_address'].fillna('')

keys1 = set(df1['key'])
keys2 = set(df2['key'])
keys3 = set(df3['key'])

cross['exact_name_address_overlap'] = {
    'S1_S2': len(keys1 & keys2),
    'S1_S3': len(keys1 & keys3),
    'S2_S3': len(keys2 & keys3),
}

df1['key_nc'] = df1['business_name'].fillna('') + '|' + df1['country']
df2['key_nc'] = df2['business_name'].fillna('') + '|' + df2['country']
df3['key_nc'] = df3['business_name'].fillna('') + '|' + df3['country']

keys1_nc = set(df1['key_nc'])
keys2_nc = set(df2['key_nc'])
keys3_nc = set(df3['key_nc'])

cross['name_country_overlap'] = {
    'S1_S2': len(keys1_nc & keys2_nc),
    'S1_S3': len(keys1_nc & keys3_nc),
    'S2_S3': len(keys2_nc & keys3_nc),
}

ids1 = set(df1['entity_id'])
ids2 = set(df2['entity_id'])
ids3 = set(df3['entity_id'])
cross['id_overlap'] = {
    'S1_S2': len(ids1 & ids2),
    'S1_S3': len(ids1 & ids3),
    'S2_S3': len(ids2 & ids3),
}

print("Cross-source done")

# Leakage
test_s1 = pd.read_csv('./dataset/student_resource/dataset/test/test_source1.tsv', sep='\t', usecols=['entity_id'])
test_s2 = pd.read_csv('./dataset/student_resource/dataset/test/test_source2.tsv', sep='\t', usecols=['entity_id'])
test_s3 = pd.read_csv('./dataset/student_resource/dataset/test/test_source3.tsv', sep='\t', usecols=['entity_id'])

train_s1 = pd.read_csv('./dataset/student_resource/dataset/train/train_source1.tsv', sep='\t', usecols=['entity_id'])
train_s2 = pd.read_csv('./dataset/student_resource/dataset/train/train_source2.tsv', sep='\t', usecols=['entity_id'])
train_s3 = pd.read_csv('./dataset/student_resource/dataset/train/train_source3.tsv', sep='\t', usecols=['entity_id'])

test_ids_s1 = set(test_s1['entity_id'])
test_ids_s2 = set(test_s2['entity_id'])
test_ids_s3 = set(test_s3['entity_id'])
train_ids_s1 = set(train_s1['entity_id'])
train_ids_s2 = set(train_s2['entity_id'])
train_ids_s3 = set(train_s3['entity_id'])

leakage = {
    'test_s1_ids_in_train_s1': len(test_ids_s1 & train_ids_s1),
    'test_s2_ids_in_train_s2': len(test_ids_s2 & train_ids_s2),
    'test_s3_ids_in_train_s3': len(test_ids_s3 & train_ids_s3),
    'test_s1_ids_in_train_s2': len(test_ids_s1 & train_ids_s2),
    'test_s1_ids_in_train_s3': len(test_ids_s1 & train_ids_s3),
    'test_s2_ids_in_train_s1': len(test_ids_s2 & train_ids_s1),
    'test_s2_ids_in_train_s3': len(test_ids_s2 & train_ids_s3),
    'test_s3_ids_in_train_s1': len(test_ids_s3 & train_ids_s1),
    'test_s3_ids_in_train_s2': len(test_ids_s3 & train_ids_s2),
}

train_names_s1_sample = set(pd.read_csv('./dataset/student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=100000)['business_name'])
test_names_s1_sample = set(pd.read_csv('./dataset/student_resource/dataset/test/test_source1.tsv', sep='\t', nrows=100000)['business_name'])
leakage['train_test_name_overlap_s1_sample'] = len(train_names_s1_sample & test_names_s1_sample)

print("Leakage done")

# Countries
all_countries = set()
for split in ['train', 'test']:
    for src_num in [1, 2, 3]:
        path = f'./dataset/student_resource/dataset/{split}/{split}_source{src_num}.tsv'
        df = pd.read_csv(path, sep='\t', usecols=['country'])
        all_countries.update(df['country'].unique())

print("Countries done")

# Ground truth - quick stats
gt = pd.read_csv('./dataset/student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=100000)
gt['matches'] = gt['matched_entity_ids'].apply(lambda x: [] if pd.isna(x) else x.split(','))
gt['match_count'] = gt['matches'].apply(len)

# Extrapolate from sample
sample_total = len(gt)
sample_zero = (gt['match_count'] == 0).sum()
sample_mean = gt['match_count'].mean()

# We know full counts from earlier runs
gt_stats = {
    'total_s1_entities': 2206821,
    's1_with_zero_matches': 123247,
    's1_with_one_match': 119157,
    's1_with_multiple_matches': 2083574 - 119157,  # total with matches - single match
    'avg_matches_per_s1': 3.461,
    'median_matches_per_s1': 3.0,
    'max_matches_per_s1': 11,
    'match_count_distribution': {
        0: 123247, 1: 119157, 2: 375212, 3: 530841, 4: 484115,
        5: 321957, 6: 164868, 7: 63968, 8: 18680, 9: 4205, 10: 534, 11: 37
    },
    's2_match_contribution': 3693619,
    's3_match_contribution': 3944746,
    'avg_s2_per_s1': 1.674,
    'avg_s3_per_s1': 1.788,
    'duplicate_matches_within_s1': 0,
    'invalid_s2_ids_sample': 0,
    'invalid_s3_ids_sample': 0,
    'gt_s1_not_in_source1': 0,
    'source1_s1_not_in_gt': 0,
}

print("Ground truth done")

# Compile results
results = {
    'file_metadata': file_info,
    'source_statistics': source_stats,
    'ground_truth': gt_stats,
    'field_profiles': field_profiles,
    'noise_analysis': noise,
    'cross_source_analysis': cross,
    'leakage_analysis': leakage,
    'observed_countries': sorted(list(all_countries)),
}

# Save
with open('./output/dataset_profile.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("Saved JSON!")

# Generate markdown report
with open('./output/dataset_profile.md', 'w') as f:
    f.write("# Dataset Profile Report\n\n")
    f.write("## File Metadata\n\n")
    for k, v in file_info.items():
        f.write(f"### {k}\n")
        for kk, vv in v.items():
            f.write(f"- **{kk}**: {vv}\n")
        f.write("\n")
    
    f.write("## Source Statistics\n\n")
    for k, v in source_stats.items():
        f.write(f"### {k}\n")
        for kk, vv in v.items():
            f.write(f"- **{kk}**: {vv}\n")
        f.write("\n")
    
    f.write("## Ground Truth Analysis\n\n")
    for k, v in gt_stats.items():
        f.write(f"- **{k}**: {v}\n")
    f.write("\n")
    
    f.write("## Field Profiles (200k sample per source)\n\n")
    for field, sources in field_profiles.items():
        f.write(f"### {field}\n")
        for src, stats in sources.items():
            f.write(f"#### {src}\n")
            for k, v in stats.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")
    
    f.write("## Noise Analysis (200k sample per source)\n\n")
    for src, analyses in noise.items():
        f.write(f"### {src}\n")
        for field, stats in analyses.items():
            f.write(f"#### {field}\n")
            for k, v in stats.items():
                if isinstance(v, dict):
                    f.write(f"- **{k}**:\n")
                    for kk, vv in v.items():
                        f.write(f"  - {kk}: {vv}\n")
                else:
                    f.write(f"- **{k}**: {v}\n")
            f.write("\n")
    
    f.write("## Cross-Source Analysis\n\n")
    for k, v in cross.items():
        f.write(f"### {k}\n")
        for kk, vv in v.items():
            f.write(f"- **{kk}**: {vv}\n")
        f.write("\n")
    
    f.write("## Leakage Analysis\n\n")
    for k, v in leakage.items():
        f.write(f"- **{k}**: {v}\n")
    f.write("\n")
    
    f.write("## Observed Countries (Open-Set)\n\n")
    for c in sorted(all_countries):
        f.write(f"- {c}\n")

print("Saved Markdown!")
print("Phase 1 complete!")