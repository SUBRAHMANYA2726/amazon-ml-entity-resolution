"""
Configuration module for Amazon ML Entity Resolution.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Main configuration class."""
    
    # Dataset paths (relative to project root)
    dataset_root: str = "./dataset/student_resource/dataset"
    
    # Train files
    train_source1: str = "train/train_source1.tsv"
    train_source2: str = "train/train_source2.tsv"
    train_source3: str = "train/train_source3.tsv"
    train_ground_truth: str = "train/train_ground_truth.tsv"
    
    # Test files
    test_source1: str = "test/test_source1.tsv"
    test_source2: str = "test/test_source2.tsv"
    test_source3: str = "test/test_source3.tsv"
    
    # Output paths
    output_dir: str = "./output"
    models_dir: str = "./models"
    logs_dir: str = "./logs"
    
    # Normalization settings
    unicode_form: str = "NFKC"
    case_lower: bool = True
    keep_punctuation_name: str = ".,-'&"
    keep_punctuation_address: str = ".,-#/"
    
    # Tokenization
    char_ngram_n: int = 3
    word_ngram_n: int = 2
    
    # Random seed
    random_seed: int = 42
    
    def get_train_path(self, source: str) -> str:
        """Get full path for train source file."""
        mapping = {
            'source1': self.train_source1,
            'source2': self.train_source2,
            'source3': self.train_source3,
        }
        return os.path.join(self.dataset_root, mapping[source])
    
    def get_test_path(self, source: str) -> str:
        """Get full path for test source file."""
        mapping = {
            'source1': self.test_source1,
            'source2': self.test_source2,
            'source3': self.test_source3,
        }
        return os.path.join(self.dataset_root, mapping[source])
    
    def get_ground_truth_path(self) -> str:
        """Get full path for ground truth file."""
        return os.path.join(self.dataset_root, self.train_ground_truth)
    
    def ensure_dirs(self):
        """Ensure output directories exist."""
        for dir_path in [self.output_dir, self.models_dir, self.logs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()

# Allow environment variable override
if 'AMZ_ML_DATASET_ROOT' in os.environ:
    config.dataset_root = os.environ['AMZ_ML_DATASET_ROOT']
if 'AMZ_ML_OUTPUT_DIR' in os.environ:
    config.output_dir = os.environ['AMZ_ML_OUTPUT_DIR']