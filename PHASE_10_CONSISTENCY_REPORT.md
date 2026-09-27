# Phase 10 — Entity Consistency + Singleton Audit Report

**Project:** Amazon ML Challenge 2026 — Business Entity Resolution
**Phase:** 10 (Entity Consistency + Singleton Audit)
**Execution Timestamp:** 2026-09-27
**Operating Threshold:** τ = 0.922 (Phase 9)
**Runner:** `scripts/run_phase10_consistency_audit.py`

All measurements on the 500 held-out validation S1 entities (34,811 pairs).
No test data used.

---

## 1. Singleton analysis (27 ground-truth singletons, 5.4%)

| Statistic | Measured value |
|---|---|
| GT singletons | 27 / 500 |
| Candidates per singleton: min / median / mean / max | 14 / 74 / 69.6 / 100 (same blocking depth as matched entities) |
| Max candidate probability: min / median / mean / max | 0.0000 / **0.0150** / — / **0.9753** |
| Singletons with ≥1 candidate above τ | 2 |
| Singleton false-positive entities at τ | **2 / 27 (rate 0.074)** |

The model cleanly suppresses 25/27 singletons (median max-probability 0.015).
The 2 residual FPs are high-confidence lookalikes of the same family as the
pair-level FPs (adjacent commercial units / shared buildings). No additional
singleton veto was found that removes them without harming recall, so **no
extra zero-match rule is retained** beyond τ = 0.922 thresholding.

Artifact: `output/phase10/singleton_analysis.{csv,json}` (per-singleton rows:
candidate count, max probability, count above threshold, top baseline score,
top provenance flag).

## 2. Zero-match rule decision

**No hard-coded zero-match override.** A singleton must not receive a match
merely because one candidate scores moderately — and at τ = 0.922 it does not:
only candidates ≥ 0.922 are assigned, which suppresses 92.6% of singletons
with zero extra machinery. Evidence does not support any further rule.

## 3. Margin analysis

Per-S1 top-1 / top-2 / gap computed for all 500 entities
(`output/phase10/margin_analysis.{csv,json}`):

- Near-zero gaps (top1 − top2 < 0.001) overwhelmingly join **two true matches**
  (distinct S2/S3 records of the same business), consistent with Phase 8's
  98.7% finding.
- A margin veto (e.g. "require gap > δ") would therefore delete true
  co-matches and destroy recall. **No margin rule retained.**

## 4. Multiple matches — top-1 rule test

- Ground truth entities with ≥2 matches: **447 / 473** (94.5%)
- Predicted entities with ≥2 matches at τ: **444 / 473** (93.9%)
- Counterfactual "take only top-1" macro F0.5 delta: **−0.280398**

**Top-1-only is emphatically rejected.** The challenge allows 0/1/N matches and
the data demands N. All candidates ≥ τ are kept (after dedup/ordering).

## 5. Duplicate prediction removal

Validation check: **0** duplicated (S1, candidate) pairs — the union engine
already dedups. For test inference the pipeline still enforces per-S1
deduplication with deterministic ordering (probability DESC,
candidate_entity_id ASC). Retained as a safety invariant.

## 6. Source consistency (S2 vs S3)

| | Predicted (τ) | Ground truth |
|---|---|---|
| Entities with both S2 and S3 matches | 394 | 405 |
| Entities only S2 / only S3 | measured in `phase10_audit.json` | 36 / 45-type split |

S2 and S3 are scored independently with **no per-source quota** — retained.

## 7. Contradiction checks (country conflict)

Joining predicted pairs (≥ τ) to Phase 5 features:

- Predicted pairs with `country_conflict = 1`: **0**
- (Of which ground-truth positive: 0)

Country partitioning in Phase 3 blocking already prevents cross-country pairs,
so a country-conflict veto is a **no-op on validation (Δ macro = +0.000000)**
and is **rejected** as unnecessary machinery. The open-set country handling
(lowercased labels, no fixed universe) is preserved for test France entities.

## 8. Entity consistency / fan-in

- Candidates assigned to >1 S1 at τ: **0**
- Invalid source IDs / cross-source candidate IDs: 0 (S2/S3 prefixes disjoint)
- Inference still runs a fan-in safety check; no reassignment rule is needed.

## 9. Before / after Phase 10

| Metric | Before (τ only) | After (retained rules) | Δ |
|---|---|---|---|
| Macro F0.5 | 0.970997 | 0.970997 | +0.000000 |
| Exact-set matches | 420 | 420 | 0 |
| Singleton FP entities | 2 | 2 | 0 |
| Matched FN entities | 2 | 2 | 0 |
| Predicted pairs | 1682 | 1682 | 0 |

Retained rules are deliberately **safety/ordering invariants** (dedup,
deterministic ordering, fan-in check), not score-changing filters — validation
evidence supports no filter beyond the Phase 9 threshold.

## 10. Retained vs rejected rules

**Retained for test inference** (`output/phase10/consistency_config.json`):

1. τ = 0.922 thresholding
2. Intra-S1 duplicate removal + deterministic ordering
3. Fan-in safety check (keep all above threshold)

**Rejected (with measured deltas):** top-1-only (−0.280398), country-conflict
veto (+0.000000, no-op), margin veto (destroys true co-matches), extra
singleton veto (unjustified recall risk), per-source quotas.

## 11. Artifacts

`output/phase10/singleton_analysis.{csv,json}`, `margin_analysis.{csv,json}`,
`phase10_audit.json`, `consistency_config.json`.

**Phase 10 complete.**
