"""
Dataset loading utilities for Amazon ML Entity Resolution.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Iterator
from pathlib import Path

from .config import config


class DatasetLoader:
    """Load and manage dataset files."""
    
    def __init__(self, dataset_root: Optional[str] = None):
        self.dataset_root = dataset_root or config.dataset_root
        self._train_cache = {}
        self._test_cache = {}
        self._gt_cache = None
    
    def load_train_source(self, source: str, nrows: Optional[int] = None, 
                          usecols: Optional[List[str]] = None) -> pd.DataFrame:
        """Load train source file."""
        cache_key = (source, nrows, tuple(usecols) if usecols else None)
        if cache_key in self._train_cache:
            return self._train_cache[cache_key]
        
        path = config.get_train_path(source)
        df = pd.read_csv(path, sep='\t', nrows=nrows, usecols=usecols)
        self._train_cache[cache_key] = df
        return df
    
    def load_test_source(self, source: str, nrows: Optional[int] = None,
                         usecols: Optional[List[str]] = None) -> pd.DataFrame:
        """Load test source file."""
        cache_key = (source, nrows, tuple(usecols) if usecols else None)
        if cache_key in self._test_cache:
            return self._test_cache[cache_key]
        
        path = config.get_test_path(source)
        df = pd.read_csv(path, sep='\t', nrows=nrows, usecols=usecols)
        self._test_cache[cache_key] = df
        return df
    
    def load_ground_truth(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load training ground truth."""
        if self._gt_cache is not None and nrows is None:
            return self._gt_cache
        
        path = config.get_ground_truth_path()
        df = pd.read_csv(path, sep='\t', nrows=nrows)
        if nrows is None:
            self._gt_cache = df
        return df
    
    def load_all_train(self, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three train sources."""
        s1 = self.load_train_source('source1', nrows)
        s2 = self.load_train_source('source2', nrows)
        s3 = self.load_train_source('source3', nrows)
        return s1, s2, s3
    
    def load_all_test(self, nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all three test sources."""
        s1 = self.load_test_source('source1', nrows)
        s2 = self.load_test_source('source2', nrows)
        s3 = self.load_test_source('source3', nrows)
        return s1, s2, s3
    
    def iter_train_chunks(self, source: str, chunksize: int = 100000,
                          usecols: Optional[List[str]] = None) -> Iterator[pd.DataFrame]:
        """Iterate over train source in chunks."""
        path = config.get_train_path(source)
        return pd.read_csv(path, sep='\t', chunksize=chunksize, usecols=usecols)
    
    def iter_ground_truth_chunks(self, chunksize: int = 100000) -> Iterator[pd.DataFrame]:
        """Iterate over ground truth in chunks."""
        path = config.get_ground_truth_path()
        return pd.read_csv(path, sep='\t', chunksize=chunksize)
    
    def get_source_stats(self) -> Dict:
        """Get statistics for all sources."""
        stats = {}
        for split in ['train', 'test']:
            for src_num in [1, 2, 3]:
                source = f'source{src_num}'
                if split == 'train':
                    df = self.load_train_source(source, usecols=['entity_id', 'country'])
                else:
                    df = self.load_test_source(source, usecols=['entity_id', 'country'])
                key = f'{split}_{source}'
                stats[key] = {
                    'record_count': len(df),
                    'unique_ids': df['entity_id'].nunique(),
                    'country_distribution': df['country'].value_counts().to_dict(),
                }
        return stats


def parse_ground_truth_matches(matched_entity_ids: str) -> List[str]:
    """Parse comma-separated matched entity IDs."""
    if pd.isna(matched_entity_ids):
        return []
    return matched_entity_ids.split(',')


def expand_ground_truth(gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand ground truth from one row per S1 to one row per (S1, matched_entity) pair.
    """
    rows = []
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matches = parse_ground_truth_matches(row['matched_entity_ids'])
        for match_id in matches:
            rows.append({
                'source1_entity_id': s1_id,
                'matched_entity_id': match_id,
                'match_source': match_id.split('-')[0],  # S2 or S3
            })
    return pd.DataFrame(rows)


def get_source_from_id(entity_id: str) -> str:
    """Extract source prefix from entity ID (S1, S2, S3)."""
    if entity_id.startswith('S1-'):
        return 'source1'
    elif entity_id.startswith('S2-'):
        return 'source2'
    elif entity_id.startswith('S3-'):
        return 'source3'
    return 'unknown'