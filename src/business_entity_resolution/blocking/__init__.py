"""Candidate-blocking interfaces and multi-strategy retrieval pipeline."""

from business_entity_resolution.blocking.base import Blocker
from business_entity_resolution.blocking.evaluation import (
    CandidateRecallEvaluator,
    CandidateRecallReport,
    parse_ground_truth,
)
from business_entity_resolution.blocking.exact import ExactBlocker
from business_entity_resolution.blocking.ngram import CharNgramBlocker
from business_entity_resolution.blocking.pipeline import (
    BlockingConfig,
    MultiStrategyBlockingPipeline,
)
from business_entity_resolution.blocking.tfidf import TfidfBlocker
from business_entity_resolution.blocking.token import AddressTokenBlocker, NameTokenBlocker
from business_entity_resolution.blocking.union import (
    AnchorCandidates,
    CandidateUnion,
    ProvenancedCandidate,
)

__all__ = [
    "AddressTokenBlocker",
    "AnchorCandidates",
    "Blocker",
    "BlockingConfig",
    "CandidateRecallEvaluator",
    "CandidateRecallReport",
    "CandidateUnion",
    "CharNgramBlocker",
    "ExactBlocker",
    "MultiStrategyBlockingPipeline",
    "NameTokenBlocker",
    "ProvenancedCandidate",
    "TfidfBlocker",
    "parse_ground_truth",
]
