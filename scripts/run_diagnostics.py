"""
Run normalization diagnostics on actual dataset.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
from amazon_ml.normalization import (
    EntityNormalizer,
    normalization_diagnostics,
)
from amazon_ml.dataset import DatasetLoader


def run_diagnostics(sample_size=50000):
    """Run normalization diagnostics on sample data."""
    print(f"Loading {sample_size} records from each source...")
    loader = DatasetLoader()
    
    s1 = loader.load_train_source('source1', nrows=sample_size)
    s2 = loader.load_train_source('source2', nrows=sample_size)
    s3 = loader.load_train_source('source3', nrows=sample_size)
    
    normalizer = EntityNormalizer()
    
    for name, df in [('S1', s1), ('S2', s2), ('S3', s3)]:
        print(f"\n=== {name} Diagnostics ===")
        
        # Business name
        name_raw = df['business_name'].fillna('').tolist()
        name_norm = []
        for raw in name_raw:
            result = normalizer.normalize_business_name(raw)
            name_norm.append(result.field_specific_normalized)
        
        diag = normalization_diagnostics(name_raw, name_norm, f'{name}_business_name')
        print(f"  Business Name:")
        print(f"    Total: {diag['total']}")
        print(f"    Changed: {diag['changed']} ({diag['changed_pct']}%)")
        print(f"    Empty normalized: {diag['empty_normalized']} ({diag['empty_normalized_pct']}%)")
        print(f"    Avg tokens: {diag['avg_tokens']}")
        print(f"    Max tokens: {diag['max_tokens']}")
        print(f"    Collisions: {diag['collision_count']}")
        
        # Business address
        addr_raw = df['business_address'].fillna('').tolist()
        addr_norm = []
        for raw in addr_raw:
            result = normalizer.normalize_business_address(raw)
            addr_norm.append(result.field_specific_normalized)
        
        diag = normalization_diagnostics(addr_raw, addr_norm, f'{name}_business_address')
        print(f"  Business Address:")
        print(f"    Total: {diag['total']}")
        print(f"    Changed: {diag['changed']} ({diag['changed_pct']}%)")
        print(f"    Empty normalized: {diag['empty_normalized']} ({diag['empty_normalized_pct']}%)")
        print(f"    Avg tokens: {diag['avg_tokens']}")
        print(f"    Max tokens: {diag['max_tokens']}")
        print(f"    Collisions: {diag['collision_count']}")
        
        # Country
        country_raw = df['country'].fillna('').tolist()
        country_norm = []
        for raw in country_raw:
            result = normalizer.normalize_country(raw)
            country_norm.append(result.field_specific_normalized)
        
        diag = normalization_diagnostics(country_raw, country_norm, f'{name}_country')
        print(f"  Country:")
        print(f"    Total: {diag['total']}")
        print(f"    Changed: {diag['changed']} ({diag['changed_pct']}%)")
        print(f"    Empty normalized: {diag['empty_normalized']} ({diag['empty_normalized_pct']}%)")
        print(f"    Avg tokens: {diag['avg_tokens']}")
        print(f"    Collisions: {diag['collision_count']}")
        
        # Show some before/after examples
        print(f"  Before/After Examples (Business Name):")
        for i in range(min(5, len(name_raw))):
            if name_raw[i] != name_norm[i]:
                raw_safe = name_raw[i].encode('ascii', 'replace').decode()
                norm_safe = name_norm[i].encode('ascii', 'replace').decode()
                print(f"    '{raw_safe}' -> '{norm_safe}'")
        
        print(f"  Before/After Examples (Business Address):")
        for i in range(min(5, len(addr_raw))):
            if addr_raw[i] != addr_norm[i]:
                raw_safe = addr_raw[i].encode('ascii', 'replace').decode()
                norm_safe = addr_norm[i].encode('ascii', 'replace').decode()
                print(f"    '{raw_safe}' -> '{norm_safe}'")


if __name__ == "__main__":
    run_diagnostics(20000)