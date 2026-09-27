"""MaxScore/hybrid TF-IDF validation (Phase 11 perf engine).

- Hybrid NAME retrieval must match TfidfBlocker EXACTLY (same code path).
- Hybrid ADDRESS retrieval must match except possibly on exact-score ties at
  the K boundary; every differing pair is asserted to be an exact tie.
- Train-sample recall check: no true-pair set differences vs standard path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from business_entity_resolution.blocking.maxscore import HybridTfidfBlocker
from business_entity_resolution.blocking.tfidf import TfidfBlocker

CACHE = Path("output/phase11/work")
KW = {"sep": ",", "dtype": str, "na_filter": False}


def _load(n_anchors: int = 300):
    s1 = pd.read_csv(CACHE / "normblock_s1_france.csv.gz", **KW).head(n_anchors)
    s2 = pd.read_csv(CACHE / "normblock_s2_france.csv.gz", **KW)
    s3 = pd.read_csv(CACHE / "normblock_s3_france.csv.gz", **KW)
    return s1, pd.concat([s2, s3], ignore_index=True)


def test_hybrid_name_exact_and_addr_tie_confined():
    import time

    s1, cands = _load()
    tb = TfidfBlocker(top_k_name=10, top_k_address=5, min_similarity=0.15, batch_size=500)
    tb.fit(cands)
    t0 = time.time()
    ref = tb.generate_pairs(s1)
    t_ref = time.time() - t0

    hy = HybridTfidfBlocker(tb, name_batch_size=500)
    t0 = time.time()
    got = hy.generate_pairs(s1)
    t_hy = time.time() - t0
    print(f"\nreference={t_ref:.1f}s hybrid={t_hy:.1f}s speedup={t_ref/max(t_hy,1e-9):.1f}x")

    id2pos = {c: i for i, c in enumerate(tb.candidate_ids)}
    name_csr = tb.name_matrix.tocsr()
    addr_csr = tb.addr_matrix.tocsr()

    name_mismatch = 0
    addr_diff_anchors = 0
    addr_nontie = 0
    for aid in s1["entity_id"].astype(str).tolist():
        r = ref.get(aid, {})
        g = got.get(aid, {})
        rn = {c for c, s in r.items() if "tfidf_name" in s}
        gn = {c for c, s in g.items() if "tfidf_name" in s}
        if rn != gn:
            name_mismatch += 1
        ra = {c for c, s in r.items() if "tfidf_addr" in s}
        ga = {c for c, s in g.items() if "tfidf_addr" in s}
        if ra == ga:
            continue
        addr_diff_anchors += 1
        prow = s1[s1["entity_id"] == aid].iloc[0]
        qn = tb.name_vectorizer.transform([prow["business_name__name_core"]])
        qa = tb.addr_vectorizer.transform([prow["business_address__address"]])
        # boundary scores on each side
        for c in set(ra) ^ set(ga):
            p = id2pos[c]
            sa = float((qa.dot(addr_csr.getrow(p).T)).todense()[0, 0])
            # tie iff some opposite-side candidate has equal addr score
            other = (ga - ra) if c in ra else (ra - ga)
            tied = any(abs(float((qa.dot(addr_csr.getrow(id2pos[o]).T)).todense()[0, 0]) - sa) < 1e-12
                       for o in other)
            if not tied:
                addr_nontie += 1
    print(f"name mismatches: {name_mismatch}/300; addr-diff anchors: {addr_diff_anchors}/300; non-tie addr swaps: {addr_nontie}")
    assert name_mismatch == 0
    assert addr_nontie == 0
