"""Tests for dataset loading, discovery, ground-truth analysis, and normalization pipeline."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from business_entity_resolution.config.settings import NormalizationSettings, load_settings
from business_entity_resolution.ingestion.dataset import LoadedFile, iter_chunks, read_header
from business_entity_resolution.ingestion.discovery import DiscoveredFile, discover_dataset
from business_entity_resolution.ingestion.ground_truth import analyse_ground_truth
from business_entity_resolution.normalization.fields import (
    FieldResult,
    NormalizerRegistry,
    classify_column,
)
from business_entity_resolution.normalization.pipeline import (
    NormalizationPipeline,
    representations_for,
)
from business_entity_resolution.normalization.text import (
    PunctuationTranslator,
    collapse_whitespace,
    fold_punctuation,
    normalize_punctuation,
    strip_control_characters,
    to_unicode,
)
from business_entity_resolution.normalization.tokens import (
    alphanumeric_tokens,
    char_ngrams,
    normalized_tokens,
    whitespace_tokens,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestDatasetDiscoveryAndLoading:
    """Verify dataset discovery and chunked loading ignore OS metadata and respect splits."""

    def test_discover_dataset_finds_challenge_files(self) -> None:
        settings = load_settings(PROJECT_ROOT / "config.yaml")
        dataset = discover_dataset(settings.dataset, PROJECT_ROOT)

        filenames = [f.filename for f in dataset.files]
        assert "train_source1.tsv" in filenames
        assert "train_source2.tsv" in filenames
        assert "train_source3.tsv" in filenames
        assert "train_ground_truth.tsv" in filenames

        # Verify OS metadata is excluded
        for f in dataset.files:
            assert "__MACOSX" not in str(f.path)
            assert not f.filename.startswith("._")
            assert f.filename != ".DS_Store"

        # Verify test files are discovered but flagged
        test_files = [f for f in dataset.files if f.split == "test"]
        assert len(test_files) == 3

    def test_read_header_detects_tsv_columns(self) -> None:
        settings = load_settings(PROJECT_ROOT / "config.yaml")
        dataset = discover_dataset(settings.dataset, PROJECT_ROOT)
        train_s1 = next(f for f in dataset.files if f.filename == "train_source1.tsv")

        loaded = read_header(train_s1, settings.dataset)
        assert loaded.delimiter == "\t"
        assert loaded.columns == ("entity_id", "business_name", "business_address", "country")

    def test_iter_chunks_loads_records(self) -> None:
        settings = load_settings(PROJECT_ROOT / "config.yaml")
        dataset = discover_dataset(settings.dataset, PROJECT_ROOT)
        train_s1 = next(f for f in dataset.files if f.filename == "train_source1.tsv")
        loaded = read_header(train_s1, settings.dataset)

        first_chunk = next(iter_chunks(loaded, settings.dataset, chunksize=100))
        assert len(first_chunk) == 100
        assert "entity_id" in first_chunk.columns
        assert first_chunk["entity_id"].iloc[0].startswith("S1-")


class TestGroundTruthAnalysis:
    """Verify ground truth analysis handles singletons, multiple matches, and source contributions."""

    def test_ground_truth_stats(self, tmp_path: Path) -> None:
        gt_content = (
            "source1_entity_id\tmatched_entity_ids\n"
            "S1-1\tS2-10,S3-20\n"
            "S1-2\t\n"
            "S1-3\tS2-30\n"
        )
        gt_file = tmp_path / "train_ground_truth.tsv"
        gt_file.write_text(gt_content, encoding="utf-8")
        settings = load_settings(PROJECT_ROOT / "config.yaml")

        disc_file = DiscoveredFile(
            path=gt_file,
            relative_path="train_ground_truth.tsv",
            filename="train_ground_truth.tsv",
            extension=".tsv",
            split="train",
            role="ground_truth",
            source_index=None,
            size_bytes=len(gt_content),
            is_readable=True,
        )
        loaded_gt = LoadedFile(
            file=disc_file,
            columns=("source1_entity_id", "matched_entity_ids"),
            delimiter="\t",
            encoding="utf-8",
        )

        stats = analyse_ground_truth(
            loaded_gt,
            settings.dataset,
            source_column="source1_entity_id",
            matched_column="matched_entity_ids",
        )
        assert stats["total_rows"] == 3
        assert stats["distinct_source1_ids"] == 3
        assert stats["s1_entities_with_zero_matches"] == 1
        assert stats["s1_entities_with_one_match"] == 1
        assert stats["s1_entities_with_multiple_matches"] == 1
        assert stats["match_contribution_by_prefix"] == {"S2": 2, "S3": 1}
        assert stats["max_matches_per_s1"] == 2


class TestCoreNormalization:
    """Verify individual normalization operations."""

    def test_unicode_normalization(self) -> None:
        # Fullwidth to halfwidth
        assert to_unicode("\uff21\uff42\uff43", form="NFKC") == "Abc"
        # Decomposed to composed
        assert to_unicode("cafe\u0301", form="NFKC") == "caf\u00e9"

    def test_case_and_whitespace(self) -> None:
        text = "  HELLO   \t WORLD \n "
        assert collapse_whitespace(text.lower(), True) == "hello world"

    def test_punctuation_normalization(self) -> None:
        # Smart quotes and dashes
        smart = "\u201cHello\u201d \u2014 World"
        cleaned = fold_punctuation(smart)
        assert '"' in cleaned
        assert "-" in cleaned

    def test_tokenization_utilities(self) -> None:
        text = "Acme & Co. (USA) #123"
        ws_tokens = whitespace_tokens(text)
        assert ws_tokens == ["Acme", "&", "Co.", "(USA)", "#123"]

        alnum = alphanumeric_tokens(text)
        assert alnum == ["Acme", "Co", "USA", "123"]

        norm_tok = normalized_tokens(text.lower())
        assert norm_tok == ["acme", "co", "usa", "123"]

        ngrams = char_ngrams("acme", (3,))
        assert ngrams == ["acm", "cme"]


class TestNormalizationPipelineAndRawPreservation:
    """Verify that normalization preserves raw fields and adds representations side-by-side."""

    def test_pipeline_preserves_raw_dataframe(self) -> None:
        settings = load_settings(PROJECT_ROOT / "config.yaml")
        pipeline = NormalizationPipeline(settings.normalization)

        df = pd.DataFrame([
            {
                "entity_id": "S1-9999",
                "business_name": "Acme & Co. LLC",
                "business_address": "123 Main St, Apt 4B",
                "country": "US",
            }
        ])

        norm_df = pipeline.normalize_frame(df)

        # Raw columns must remain intact and unchanged
        assert norm_df["business_name"].iloc[0] == "Acme & Co. LLC"
        assert norm_df["business_address"].iloc[0] == "123 Main St, Apt 4B"
        assert norm_df["country"].iloc[0] == "US"
        assert norm_df["entity_id"].iloc[0] == "S1-9999"

        # Side-by-side representation columns must be present
        assert "business_name__case_normalized" in norm_df.columns
        assert "business_name__tokens" in norm_df.columns
        assert "business_address__tokens" in norm_df.columns
        assert "country__country" in norm_df.columns

    def test_normalization_determinism(self) -> None:
        """Calling normalization twice on the same input produces identical output."""
        settings = load_settings(PROJECT_ROOT / "config.yaml")
        pipeline = NormalizationPipeline(settings.normalization)

        df = pd.DataFrame([
            {
                "entity_id": "S2-12345",
                "business_name": "  Global & Tech   Inc.  ",
                "business_address": "456 Oak Avenue, Suite 100",
                "country": "India",
            }
        ])

        run1 = pipeline.normalize_frame(df)
        run2 = pipeline.normalize_frame(df)

        pd.testing.assert_frame_equal(run1, run2)

    def test_open_set_country_behavior(self) -> None:
        """Country normalization allows unseen countries to pass through cleanly."""
        settings = load_settings(PROJECT_ROOT / "config.yaml")
        registry = NormalizerRegistry(settings.normalization)

        country_normalizer = registry.for_column("country")
        assert country_normalizer.kind == "country"

        result_known = country_normalizer.normalize("country", "US")
        assert result_known.extras.get("country") == "us"

        result_unseen = country_normalizer.normalize("country", "Madagascar")
        # Unseen country passes through, cleaned and case-folded, never rejected
        assert result_unseen.extras.get("country") == "madagascar"
