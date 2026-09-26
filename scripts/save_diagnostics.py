"""
Save normalization diagnostics to JSON.
"""

import sys
sys.path.insert(0, 'src')

import json
import pandas as pd
from amazon_ml.normalization import (
    EntityNormalizer,
    normalization_diagnostics,
)
from amazon_ml.dataset import DatasetLoader


def run_and_save_diagnostics(sample_size=50000):
    """Run normalization diagnostics and save to JSON."""
    print(f"Loading {sample_size} records from each source...")
    loader = DatasetLoader()
    
    s1 = loader.load_train_source('source1', nrows=sample_size)
    s2 = loader.load_train_source('source2', nrows=sample_size)
    s3 = loader.load_train_source('source3', nrows=sample_size)
    
    normalizer = EntityNormalizer()
    all_diagnostics = {}
    
    for name, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
        print(f"Processing {name}...")
        all_diagnostics[name] = {}
        
        # Business name
        name_raw = df['business_name'].fillna('').tolist()
        name_norm = []
        for raw in name_raw:
            result = normalizer.normalize_business_name(raw)
            name_norm.append(result.field_specific_normalized)
        
        diag = normalization_diagnostics(name_raw, name_norm, f'{name}_business_name')
        all_diagnostics[name]['business_name'] = diag
        
        # Business address
        addr_raw = df['business_address'].fillna('').tolist()
        addr_norm = []
        for raw in addr_raw:
            result = normalizer.normalize_business_address(raw)
            addr_norm.append(result.field_specific_normalized)
        
        diag = normalization_diagnostics(addr_raw, addr_norm, f'{name}_business_address')
        all_diagnostics[name]['business_address'] = diag
        
        # Country
        country_raw = df['country'].fillna('').tolist()
        country_norm = []
        for raw in country_raw:
            result = normalizer.normalize_country(raw)
            country_norm.append(result.field_specific_normalized)
        
        diag = normalization_diagnostics(country_raw, country_norm, f'{name}_country')
        all_diagnostics[name]['country'] = diag
        
        # Before/after examples
        examples = []
        for i in range(min(10, len(name_raw))):
            if name_raw[i] != name_norm[i]:
                examples.append({
                    'raw': name_raw[i],
                    'normalized': name_norm[i]
                })
        all_diagnostics[name]['business_name_examples'] = examples
        
        examples = []
        for i in range(min(10, len(addr_raw))):
            if addr_raw[i] != addr_norm[i]:
                examples.append({
                    'raw': addr_raw[i],
                    'normalized': addr_norm[i]
                })
        all_diagnostics[name]['business_address_examples'] = examples
    
    # Save
    with open('./output/normalization_diagnostics.json', 'w', encoding='utf-8') as f:
        json.dump(all_diagnostics, f, indent=2, ensure_ascii=False)
    
    print("Diagnostics saved to ./output/normalization_diagnostics.json")
    
    # Also create a markdown summary
    with open('./output/normalization_diagnostics.md', 'w', encoding='utf-8') as f:
        f.write("# Normalization Diagnostics\n\n")
        f.write(f"Sample size per source: {sample_size}\n\n")
        
        for name, diags in all_diagnostics.items():
            f.write(f"## {name}\n\n")
            
            for field, diag in diags.items():
                if field.endswith('_examples'):
                    continue
                f.write(f"### {field}\n")
                f.write(f"- Total: {diag['total']}\n")
                f.write(f"- Changed: {diag['changed']} ({diag['changed_pct']}%)\n")
                f.write(f"- Empty normalized: {diag['empty_normalized']} ({diag['empty_normalized_pct']}%)\n")
                f.write(f"- Avg tokens: {diag['avg_tokens']}\n")
                f.write(f"- Max tokens: {diag['max_tokens']}\n")
                f.write(f"- Collisions: {diag['collision_count']}\n\n")
            
            f.write("#### Business Name Examples\n")
            for ex in diags.get('business_name_examples', [])[:5]:
                f.write(f"- `{ex['raw']}` -> `{ex['normalized']}`\n")
            f.write("\n")
            
            f.write("#### Business Address Examples\n")
            for ex in diags.get('business_address_examples', [])[:5]:
                f.write(f"- `{ex['raw']}` -> `{ex['normalized']}`\n")
            f.write("\n")
    
    print("Markdown summary saved to ./output/normalization_diagnostics.md")


if __name__ == "__main__":
    run_and_save_diagnostics(20000)