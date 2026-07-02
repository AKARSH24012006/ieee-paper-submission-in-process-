"""
agents/career_agent_v2.py — Career Guidance Agent v2 (Semantic Embeddings Upgrade)

CRITICAL WEAKNESS FIXED from v1:
  v1 _skill_match_score: Jaccard keyword overlap → PURE RULE-BASED
     Problem: "Machine Learning" ≠ "ML", "Python Programming" ≠ "Python"
     NDCG scores are inflated because ground truth uses same keyword logic.

  v2 _skill_match_score: sentence-transformer cosine similarity → GENUINE ML
     "Machine Learning" ≈ "ML" ≈ "Statistical Modeling" (cosine ~0.82)
     Differentiates YUVA-AI from a simple lookup table.

v1 _interest_alignment_score: substring matching → BUGGY
     "AI" substring-matches "EMAIL", "GMAIL" falsely.

v2 _interest_alignment_score: semantic cosine → CORRECT

Citation for paper Section IV-B:
  Reimers & Gurevych (2019). Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks. EMNLP 2019. doi:10.18653/v1/D19-1410

Model: sentence-transformers/all-MiniLM-L6-v2
  - 80MB download, free, offline after first use
  - ~12ms encode time on CPU (negligible for latency budget)
  - 384-dimensional embeddings, cosine similarity works well

ABLATION STUDY SUPPORT:
  score_career(variant=X) supports 5 variants for Table IV in the paper:
    A_random, B_keyword_skill, C_v1_full, D_semantic_skill, E_full_v2
"""
import json
import time
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# ── Lazy-load embedding model ──────────────────────────────────
_embed_model   = None
EMBEDDINGS_OK  = False

def _load_embedder():
    global _embed_model, EMBEDDINGS_OK
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model  = SentenceTransformer("all-MiniLM-L6-v2")
        EMBEDDINGS_OK = True
        print("[CGA-v2] Loaded sentence-transformer: all-MiniLM-L6-v2")
    except ImportError:
        print("[CGA-v2] WARN: pip install sentence-transformers  →  keyword fallback active")
    return _embed_model


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically stable cosine similarity."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if (na > 1e-9 and nb > 1e-9) else 0.0


# ── Career data ───────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "data" / "career_profiles.json"

def _load_careers() -> List[Dict]:
    with open(DATA_PATH) as f:
        return json.load(f)["careers"]

CAREERS = _load_careers()

# Module-level embedding caches (populated once at import)
_skill_emb_cache:    Dict[str, np.ndarray] = {}
_interest_emb_cache: Dict[str, np.ndarray] = {}


def _precompute():
    """Pre-encode all career skill + interest descriptions. Called once at import."""
    model = _load_embedder()
    if model is None:
        return
    for c in CAREERS:
        skill_text = ", ".join(c["required_skills"] + c.get("preferred_skills", []))
        cat_text   = (f"{c['category']} {c['title']} "
                      f"{' '.join(c.get('related_careers', []))}")
        _skill_emb_cache[c["title"]]    = model.encode(skill_text,    show_progress_bar=False)
        _interest_emb_cache[c["title"]] = model.encode(cat_text, show_progress_bar=False)
    print(f"[CGA-v2] Pre-encoded {len(CAREERS)} careers")


# ══════════════════════════════════════════════════════════════
#  SCORING — v2 (semantic) and v1 (keyword) side by side
# ══════════════════════════════════════════════════════════════

def skill_match_v2(user_skills: List[str], career: Dict) -> float:
    """
    Semantic skill match via sentence-transformer cosine similarity.

    Encode user's combined skill list as one sentence, compare against
    pre-encoded career skills sentence. Apply mild sharpening (^0.7)
    to spread score distribution for ranking.

    Range: [0, 1].  Higher = user's skills are semantically closer to role requirements.
    """
    model = _load_embedder()
    if model is None or not _skill_emb_cache:
        return skill_match_v1(user_skills, career)
    if not user_skills:
        return 0.0
    user_emb   = model.encode(", ".join(user_skills), show_progress_bar=False)
    career_emb = _skill_emb_cache.get(career["title"])
    if career_emb is None:
        return skill_match_v1(user_skills, career)
    raw = _cosine_sim(user_emb, career_emb)
    return float(max(0.0, raw) ** 0.7)


def interest_match_v2(user_interests: List[str], career: Dict) -> float:
    """
    Semantic interest alignment.

    Fixes v1's substring bug: "AI" no longer falsely matches "EMAIL" category.
    "Artificial Intelligence" correctly aligns with "AI/ML" category.

    Range: [0, 1].
    """
    model = _load_embedder()
    if model is None or not _interest_emb_cache:
        return interest_match_v1(user_interests, career)
    if not user_interests:
        return 0.3
    user_emb     = model.encode(" ".join(user_interests), show_progress_bar=False)
    interest_emb = _interest_emb_cache.get(career["title"])
    if interest_emb is None:
        return interest_match_v1(user_interests, career)
    return float(_cosine_sim(user_emb, interest_emb))


def skill_match_v1(user_skills: List[str], career: Dict) -> float:
    """Original Jaccard keyword overlap (v1). Kept for ablation baseline."""
    if not career["required_skills"]:
        return 0.0
    ul = {s.lower() for s in user_skills}
    rl = {s.lower() for s in career["required_skills"]}
    pl = {s.lower() for s in career.get("preferred_skills", [])}
    r  = len(ul & rl) / len(rl)
    p  = len(ul & pl) / max(len(pl), 1)
    return min(0.75 * r + 0.25 * p, 1.0)


def interest_match_v1(user_interests: List[str], career: Dict) -> float:
    """Original substring match (v1). Kept for ablation baseline."""
    if not user_interests:
        return 0.3
    il  = {i.lower() for i in user_interests}
    cat = career.get("category", "").lower()
    tit = career.get("title", "").lower()
    rel = [r.lower() for r in career.get("related_careers", [])]
    for i in il:
        if i in cat or cat in i:
            return 1.0
        if i in tit:
            return 0.9
        if any(i in r for r in rel):
            return 0.7
    return 0.2


def _demand_score(career: Dict) -> float:
    return career.get("demand_score", 5.0) / 10.0


# ══════════════════════════════════════════════════════════════
#  COMPOSITE SCORING WITH ABLATION SUPPORT
# ══════════════════════════════════════════════════════════════

W_S, W_I, W_D = 0.45, 0.30, 0.25   # tunable hyperparameters

def score_career(user_skills: List[str], user_interests: List[str],
                 career: Dict, variant: str = "E_full_v2") -> Dict:
    """
    Score one career for one user.

    variant param enables ablation study (Table IV in paper):
      A_random        → Random baseline (shuffle scores)
      B_keyword_skill → Skill keyword only, no interest, no demand
      C_v1_full       → Original v1: keyword skill + substring interest + demand
      D_semantic_skill→ Semantic skill + keyword interest + demand (partial upgrade)
      E_full_v2       → Full proposed system: semantic skill + semantic interest + demand
    """
    import random

    if variant == "A_random":
        s, i_s, d_s = random.random(), random.random(), _demand_score(career)
        composite   = random.random()
    elif variant == "B_keyword_skill":
        s   = skill_match_v1(user_skills, career)
        i_s = 0.5
        d_s = 0.5
        composite = s
    elif variant == "C_v1_full":
        s   = skill_match_v1(user_skills, career)
        i_s = interest_match_v1(user_interests, career)
        d_s = _demand_score(career)
        composite = W_S * s + W_I * i_s + W_D * d_s
    elif variant == "D_semantic_skill":
        s   = skill_match_v2(user_skills, career)
        i_s = interest_match_v1(user_interests, career)
        d_s = _demand_score(career)
        composite = W_S * s + W_I * i_s + W_D * d_s
    else:  # E_full_v2 (proposed)
        s   = skill_match_v2(user_skills, career)
        i_s = interest_match_v2(user_interests, career)
        d_s = _demand_score(career)
        composite = W_S * s + W_I * i_s + W_D * d_s

    return {
        "title":          career["title"],
        "category":       career["category"],
        "match_score":    round(composite, 4),
        "skill_score":    round(s,         4),
        "interest_score": round(i_s,       4),
        "demand_score":   round(d_s,       4),
        "salary_range":   career.get("salary_range_india", {}),
        "growth_outlook": career.get("growth_outlook", ""),
        "free_resources": career.get("free_resources", []),
        "certifications": career.get("certifications", []),
        "reason":         "",
        "variant":        variant,
    }


def rank_careers(user_skills: List[str], user_interests: List[str],
                 variant: str = "E_full_v2", top_k: int = 5) -> List[Dict]:
    """Rank all careers for a user under a given variant."""
    scored = [score_career(user_skills, user_interests, c, variant) for c in CAREERS]
    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored[:top_k]


# ── Initialise on import ───────────────────────────────────────
_precompute()
