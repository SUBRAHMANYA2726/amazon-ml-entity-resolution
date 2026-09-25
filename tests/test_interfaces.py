"""Tests for importable, intentionally abstract Phase 0 contracts."""

from __future__ import annotations

import importlib

import pytest

from business_entity_resolution.blocking import Blocker
from business_entity_resolution.contracts import ValidationResult
from business_entity_resolution.entity import EntityConsistencyChecker, SingletonAuditor
from business_entity_resolution.evaluation import Evaluator, ThresholdOptimizer
from business_entity_resolution.features import PairFeatureGenerator
from business_entity_resolution.inference import InferencePipeline
from business_entity_resolution.ingestion import DataIngestor, DatasetProfiler, SchemaValidator
from business_entity_resolution.models import Matcher
from business_entity_resolution.normalization import Normalizer
from business_entity_resolution.retrieval import CandidateRetriever
from business_entity_resolution.submission import SubmissionGenerator
from business_entity_resolution.training import HardNegativeMiner, Trainer
from business_entity_resolution.validation import SubmissionValidator


@pytest.mark.parametrize(
    "module_name",
    [
        "business_entity_resolution.config",
        "business_entity_resolution.ingestion",
        "business_entity_resolution.normalization",
        "business_entity_resolution.blocking",
        "business_entity_resolution.retrieval",
        "business_entity_resolution.features",
        "business_entity_resolution.models",
        "business_entity_resolution.training",
        "business_entity_resolution.evaluation",
        "business_entity_resolution.entity",
        "business_entity_resolution.inference",
        "business_entity_resolution.submission",
        "business_entity_resolution.validation",
        "business_entity_resolution.utils",
    ],
)
def test_foundation_packages_import(module_name: str) -> None:
    """All Phase 0 packages remain importable without ML dependencies."""

    assert importlib.import_module(module_name)


@pytest.mark.parametrize(
    "interface",
    [
        DataIngestor,
        DatasetProfiler,
        SchemaValidator,
        Normalizer,
        Blocker,
        CandidateRetriever,
        PairFeatureGenerator,
        Matcher,
        Trainer,
        HardNegativeMiner,
        Evaluator,
        ThresholdOptimizer,
        EntityConsistencyChecker,
        SingletonAuditor,
        InferencePipeline,
        SubmissionGenerator,
        SubmissionValidator,
    ],
)
def test_interfaces_require_concrete_implementation(interface: type[object]) -> None:
    """Concrete behavior cannot be accidentally used before it is implemented."""

    with pytest.raises(TypeError):
        interface()


def test_validation_contract_retains_generic_messages() -> None:
    """Generic validation results require no imaginary dataset schema."""

    result = ValidationResult(is_valid=False, messages=("future contract required",))

    assert result.is_valid is False
    assert result.messages == ("future contract required",)
