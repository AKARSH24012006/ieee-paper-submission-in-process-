"""
experiments/user_study.py — User Study Tools for YUVA-AI

TWO tools in one file:
  1. annotation_cli()  — Command-line annotation tool for ground truth
  2. sus_collector()   — SUS questionnaire data entry + scoring

HOW TO USE:
  Annotation (you + 2 friends, ~1 hour total):
    python experiments/user_study.py --mode annotate --profiles 30

  SUS collection (after user study sessions):
    python experiments/user_study.py --mode sus --add

  Scoring:
    python experiments/user_study.py --mode sus --score
"""
import json
import sys
import argparse
from pathlib import Path

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ANN_FILE = DATA_DIR / "human_annotations.json"
SUS_FILE = RESULTS_DIR / "sus_responses.json"


# ══════════════════════════════════════════════════════════════
#  ANNOTATION TOOL
# ══════════════════════════════════════════════════════════════

RUBRIC = """
ANNOTATION RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3 = EXCELLENT fit  (strong skills + strong interest + realistic)
2 = GOOD fit       (moderate skills OR strong interest)
1 = MARGINAL fit   (possible with significant upskilling)
0 = POOR fit       (incompatible or contradictory)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rate each career for this student profile.
"""


def annotation_cli(n_profiles: int = 30):
    """Interactive annotation CLI for ground truth labelling."""
    profiles_path = DATA_DIR / "synthetic_users.json"
    if not profiles_path.exists():
        print("[ERROR] Run: python data/generate_synthetic.py first")
        return

    with open(profiles_path) as f:
        all_profiles = json.load(f)[:n_profiles]

    # Load existing annotations
    annotations = {}
    if ANN_FILE.exists():
        with open(ANN_FILE) as f:
            annotations = json.load(f)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "yuva_ai"))
    from agents.career_agent_v2 import CAREERS

    print(RUBRIC)
    CAREERS_TO_RATE = [c["title"] for c in CAREERS[:5]]  # Rate top-5 per profile

    annotated_count = 0
    for p in all_profiles:
        uid = p.get("user_id", "")
        if uid in annotations:
            print(f"[SKIP] {uid} already annotated.")
            continue

        profile = p.get("profile", p)
        print(f"\n{'='*60}")
        print(f"Profile ID: {uid}")
        print(f"Education: {profile.get('education_level')} @ {profile.get('institution','—')}")
        print(f"Skills:    {', '.join(profile.get('skills',[]))}")
        print(f"Interests: {', '.join(profile.get('interests',[]))}")
        print(f"Goals:     {profile.get('goals','')[:150]}")
        print(f"{'='*60}")

        user_ratings = {}
        for career in CAREERS_TO_RATE:
            while True:
                try:
                    r = input(f"  Rate '{career}' [0/1/2/3]: ").strip()
                    if r in ("0","1","2","3"):
                        user_ratings[career] = int(r)
                        break
                    print("  Please enter 0, 1, 2, or 3")
                except KeyboardInterrupt:
                    print("\n[Saving and exiting...]")
                    _save_annotations(annotations, ANN_FILE)
                    return

        annotations[uid] = user_ratings
        annotated_count += 1
        _save_annotations(annotations, ANN_FILE)
        print(f"  ✓ Saved ({annotated_count}/{n_profiles})")

    print(f"\n✅ Annotation complete: {annotated_count} profiles annotated.")
    print(f"   Saved to: {ANN_FILE}")
    print(f"\nNext step: Ask 2 friends to repeat this annotation independently.")
    print(f"Then compute Cohen's κ using: python experiments/user_study.py --mode kappa")


def _save_annotations(annotations: dict, path: Path):
    with open(path, "w") as f:
        json.dump(annotations, f, indent=2)


def compute_kappa_from_files(ann_file_1: str, ann_file_2: str):
    """
    Compute Cohen's κ between two annotation files.

    Usage:
      python experiments/user_study.py --mode kappa
              --ann1 data/ann_rater1.json --ann2 data/ann_rater2.json
    """
    with open(ann_file_1) as f: ann1 = json.load(f)
    with open(ann_file_2) as f: ann2 = json.load(f)

    rater1, rater2 = [], []
    for uid in ann1:
        if uid in ann2:
            for career in ann1[uid]:
                if career in ann2[uid]:
                    rater1.append(ann1[uid][career])
                    rater2.append(ann2[uid][career])

    # Cohen's kappa
    n          = len(rater1)
    categories = list(set(rater1 + rater2))
    p_o        = sum(r1 == r2 for r1, r2 in zip(rater1, rater2)) / n
    p_e        = sum(
        (rater1.count(c)/n) * (rater2.count(c)/n) for c in categories
    )
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-9 else 1.0

    grade = ("Almost perfect" if kappa>=0.81 else "Substantial" if kappa>=0.61
             else "Moderate" if kappa>=0.41 else "Fair" if kappa>=0.21 else "Slight")
    print(f"\nInter-Rater Agreement (Cohen's κ)")
    print(f"  n pairs:  {n}")
    print(f"  κ = {kappa:.4f}  ({grade} agreement)")
    print(f"  Report in paper as: κ = {kappa:.2f} ({grade.lower()})")
    print(f"\n  Interpretation (Landis & Koch, 1977):")
    print(f"  κ ≥ 0.61 = Substantial  ← minimum acceptable for IEEE paper")
    print(f"  κ ≥ 0.81 = Almost perfect ← ideal")
    return kappa


# ══════════════════════════════════════════════════════════════
#  SUS DATA COLLECTION
# ══════════════════════════════════════════════════════════════

SUS_QUESTIONS = [
    "Q1:  I think that I would like to use this system frequently.",
    "Q2:  I found the system unnecessarily complex.",
    "Q3:  I thought the system was easy to use.",
    "Q4:  I think that I would need the support of a technical person to use this.",
    "Q5:  I found the various functions in this system were well integrated.",
    "Q6:  I thought there was too much inconsistency in this system.",
    "Q7:  I would imagine that most people would learn this system quickly.",
    "Q8:  I found the system very cumbersome to use.",
    "Q9:  I felt very confident using the system.",
    "Q10: I needed to learn a lot of things before I could get going.",
]
SUS_ODD  = {1,3,5,7,9}
SUS_EVEN = {2,4,6,8,10}

GOOGLE_FORM_TEMPLATE = """
GOOGLE FORM TEMPLATE FOR SUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Create a Google Form with these questions.
Each question: Linear Scale 1-5
  1 = Strongly Disagree → 5 = Strongly Agree

Title: YUVA-AI User Experience Survey
Description: Thank you for testing YUVA-AI! Please answer honestly.
             This takes ~5 minutes.

Questions (copy-paste into Google Form):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q1: I think that I would like to use this system frequently.
Q2: I found the system unnecessarily complex.
Q3: I thought the system was easy to use.
Q4: I think that I would need help from a technical person to use this system.
Q5: I found the various functions in this system were well integrated.
Q6: I thought there was too much inconsistency in this system.
Q7: I would imagine that most people would learn to use this system quickly.
Q8: I found the system very cumbersome to use.
Q9: I felt very confident using the system.
Q10: I needed to learn a lot of things before I could get started.

Additional demographics (required for Table V in paper):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
D1: Year of study (2nd/3rd/4th)
D2: Branch (CSE / ECE / Other)
D3: Gender (Male/Female/Other/Prefer not to say)
D4: Have you used NCS Portal or similar? (Yes/No)
D5: Did you find the career recommendations relevant? (5-point scale)
D6: Did you find the skill gap analysis helpful? (5-point scale)
D7: Would you recommend this app to a friend? (5-point scale)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Export as CSV → import via sus_from_csv()
"""


def sus_add_response():
    """CLI to manually add one SUS response."""
    responses = []
    print("\nSUS Data Entry (enter 1-5 for each question)")
    print("─" * 50)
    for i, q in enumerate(SUS_QUESTIONS, 1):
        while True:
            try:
                val = int(input(f"{q} [1-5]: ").strip())
                if 1 <= val <= 5:
                    responses.append(val)
                    break
                print("Please enter 1, 2, 3, 4, or 5")
            except ValueError:
                print("Please enter a number")

    score = _calculate_sus(responses)
    print(f"\n  SUS Score: {score:.1f}")

    # Load existing
    all_resp = []
    if SUS_FILE.exists():
        with open(SUS_FILE) as f:
            all_resp = json.load(f)

    all_resp.append({"responses": responses, "score": score})
    with open(SUS_FILE, "w") as f:
        json.dump(all_resp, f, indent=2)
    print(f"  Saved to: {SUS_FILE}")


def sus_from_csv(csv_path: str):
    """
    Import SUS responses from Google Form CSV export.

    Expects columns Q1..Q10 with values 1-5.
    """
    import csv
    all_resp = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                responses = [int(row[f"Q{i}"].strip()) for i in range(1, 11)]
                score     = _calculate_sus(responses)
                all_resp.append({"responses": responses, "score": score})
            except (KeyError, ValueError) as e:
                print(f"[WARN] Skipping row: {e}")

    with open(SUS_FILE, "w") as f:
        json.dump(all_resp, f, indent=2)
    print(f"Imported {len(all_resp)} responses → {SUS_FILE}")
    sus_score_report()


def sus_score_report():
    """Print aggregate SUS report from stored responses."""
    if not SUS_FILE.exists():
        print("No SUS data yet. Add responses first.")
        return

    with open(SUS_FILE) as f:
        data = json.load(f)

    scores = [d["score"] for d in data]
    n      = len(scores)
    if n == 0:
        print("No SUS responses found.")
        return

    import math
    mean = sum(scores) / n
    std  = math.sqrt(sum((s-mean)**2 for s in scores) / n)
    srt  = sorted(scores)
    med  = srt[n//2]

    grade = ("Excellent" if mean >= 80 else "Good" if mean >= 70
             else "OK" if mean >= 60 else "Poor")

    print(f"\n{'='*50}")
    print(f"  YUVA-AI USER STUDY — SUS RESULTS")
    print(f"{'='*50}")
    print(f"  Participants:  n = {n}")
    print(f"  Mean SUS:      {mean:.2f} ± {std:.2f}")
    print(f"  Median:        {med:.1f}")
    print(f"  Min/Max:       {min(scores):.1f} / {max(scores):.1f}")
    print(f"  Grade:         {grade}")
    print(f"  Report as:     SUS = {mean:.1f} ± {std:.1f} ({grade})")
    print(f"\n  Comparison:")
    print(f"  NCS Portal baseline: ~52 (needs improvement)")
    print(f"  YUVA-AI target:      ≥70 (good)")
    print(f"{'='*50}")
    print(f"\n  LaTeX: \\textbf{{SUS}} = ${mean:.1f} \\pm {std:.1f}$ ({grade})")
    return {"n": n, "mean": round(mean,2), "std": round(std,2), "grade": grade}


def _calculate_sus(responses):
    total = 0
    for i, r in enumerate(responses, 1):
        total += (r - 1) if i in SUS_ODD else (5 - r)
    return total * 2.5


def print_google_form():
    print(GOOGLE_FORM_TEMPLATE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YUVA-AI User Study Tools")
    parser.add_argument("--mode", choices=["annotate","sus","kappa","form"], required=True)
    parser.add_argument("--profiles", type=int, default=30)
    parser.add_argument("--add",   action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--csv",   type=str, default=None)
    parser.add_argument("--ann1",  type=str, default=None)
    parser.add_argument("--ann2",  type=str, default=None)
    args = parser.parse_args()

    if args.mode == "annotate":
        annotation_cli(args.profiles)
    elif args.mode == "sus":
        if args.add:
            sus_add_response()
        elif args.csv:
            sus_from_csv(args.csv)
        else:
            sus_score_report()
    elif args.mode == "kappa":
        if args.ann1 and args.ann2:
            compute_kappa_from_files(args.ann1, args.ann2)
        else:
            print("Usage: --mode kappa --ann1 FILE1 --ann2 FILE2")
    elif args.mode == "form":
        print_google_form()
