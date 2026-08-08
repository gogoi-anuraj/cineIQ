# src/explain/meta_explain.py — Part 1

import json
import pickle
import pandas as pd

def load_meta_model_coefficients(results_path: str) -> dict:
    with open(results_path, "r") as f:
        results = json.load(f)
    return results["coefficients"]


def explain_meta_model_contribution(coefficients: dict) -> str:
    """
    Turns the meta-model's learned coefficients into a human-readable
    explanation of how much each signal generally contributes to
    recommendations. This is a GLOBAL explanation (how the model weighs
    signals overall), not a per-recommendation one -- see
    explain_single_recommendation() in Part 2 for the per-movie version.
    """
    # Normalize to percentages for readability (coefficients are on a
    # standardized scale, so relative magnitude is meaningful)
    total = sum(abs(v) for v in coefficients.values())
    shares = {k: abs(v) / total for k, v in coefficients.items()}

    ranked = sorted(shares.items(), key=lambda x: x[1], reverse=True)

    lines = ["Overall, this system weighs recommendation signals as follows:"]
    signal_names = {
        "svd_score": "collaborative filtering (what similar users liked)",
        "content_score": "content similarity (genre, keywords, cast)",
        "gru_score": "your recent watch sequence",
    }
    for name, share in ranked:
        readable = signal_names.get(name, name)
        lines.append(f"  - {readable}: {share*100:.0f}% of the blend")

    return "\n".join(lines)

# def explain_single_recommendation(svd_score: float, content_score: float, gru_score: float,
#                                    coefficients: dict, score_context: dict = None) -> str:
#     """
#     Explains one specific recommendation by comparing its 3 raw scores
#     against typical/reference ranges, then attributing credit using the
#     coefficients as weights. score_context (optional) can supply
#     dataset-wide percentiles for more calibrated language (e.g. "well
#     above average" vs. just printing raw numbers).
#     """
#     # Weighted contribution of each signal to THIS recommendation
#     contributions = {
#         "svd_score": svd_score * coefficients["svd_score"],
#         "content_score": content_score * coefficients["content_score"],
#         "gru_score": gru_score * coefficients["gru_score"],
#     }
#     total_contribution = sum(abs(v) for v in contributions.values())
#     shares = {k: abs(v) / total_contribution if total_contribution > 0 else 0 for k, v in contributions.items()}

#     ranked = sorted(shares.items(), key=lambda x: x[1], reverse=True)
#     top_signal, top_share = ranked[0]

#     signal_explanations = {
#         "svd_score": "users with similar taste to yours rated this highly",
#         "content_score": "it closely matches the genres, themes, and cast of movies you've enjoyed",
#         "gru_score": "it fits the pattern of what you've been watching recently",
#     }

#     primary_reason = signal_explanations[top_signal]
#     explanation = f"Recommended primarily because {primary_reason} ({top_share*100:.0f}% of the reasoning)."

#     if len(ranked) > 1 and ranked[1][1] > 0.15:  # mention a secondary reason if it's non-trivial
#         second_signal, second_share = ranked[1]
#         explanation += f" {signal_explanations[second_signal].capitalize()} also contributed ({second_share*100:.0f}%)."

#     return explanation
# src/explain/meta_explain.py — Part 2 (corrected)

def explain_single_recommendation(svd_score: float, content_score: float, gru_score: float,
                                   coefficients: dict, scaler) -> str:
    """
    Explains one recommendation using STANDARDIZED scores (via the same
    scaler the meta-model was trained on) multiplied by coefficients --
    this correctly reflects each signal's contribution relative to its
    own typical range, rather than being dominated by whichever signal
    happens to have the largest raw scale (SVD, in this case).
    """
    import numpy as np
    raw = pd.DataFrame([[svd_score, content_score, gru_score]], columns=["svd_score", "content_score", "gru_score"])
    standardized = scaler.transform(raw)[0]  # [svd_z, content_z, gru_z]

    contributions = {
        "svd_score": standardized[0] * coefficients["svd_score"],
        "content_score": standardized[1] * coefficients["content_score"],
        "gru_score": standardized[2] * coefficients["gru_score"],
    }
    total = sum(abs(v) for v in contributions.values())
    shares = {k: abs(v) / total if total > 0 else 0 for k, v in contributions.items()}
    ranked = sorted(shares.items(), key=lambda x: x[1], reverse=True)
    top_signal, top_share = ranked[0]

    signal_explanations = {
        "svd_score": "users with similar taste to yours rated this highly",
        "content_score": "it closely matches the genres, themes, and cast of movies you've enjoyed",
        "gru_score": "it fits the pattern of what you've been watching recently",
    }

    explanation = f"Recommended primarily because {signal_explanations[top_signal]} ({top_share*100:.0f}% of the reasoning)."
    if len(ranked) > 1 and ranked[1][1] > 0.15:
        second_signal, second_share = ranked[1]
        explanation += f" {signal_explanations[second_signal].capitalize()} also contributed ({second_share*100:.0f}%)."
    return explanation


if __name__ == "__main__":
    import pickle
    coefficients = load_meta_model_coefficients("data/interim/meta_model_results.json")
    with open("data/interim/meta_model_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    test_cases = [
        {"svd_score": 4.8, "content_score": 0.15, "gru_score": 8.0, "label": "high on all 3"},
        {"svd_score": 4.5, "content_score": 0.01, "gru_score": -2.0, "label": "SVD-driven only"},
        {"svd_score": 3.0, "content_score": 0.16, "gru_score": 1.0, "label": "content-driven"},
    ]
    for case in test_cases:
        label = case.pop("label")
        print(f"--- {label} ---")
        print(explain_single_recommendation(**case, coefficients=coefficients, scaler=scaler))
        print()

if __name__ == "__main__":
    coefficients = load_meta_model_coefficients("data/interim/meta_model_results.json")
    with open("data/interim/meta_model_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("raw coefficients:", coefficients)
    print()
    print(explain_meta_model_contribution(coefficients))

    test_cases = [
        {"svd_score": 4.8, "content_score": 0.15, "gru_score": 8.0, "label": "high on all 3"},
        {"svd_score": 4.5, "content_score": 0.01, "gru_score": -2.0, "label": "SVD-driven only"},
        {"svd_score": 3.0, "content_score": 0.16, "gru_score": 1.0, "label": "content-driven"},
    ]

    for case in test_cases:
        label = case.pop("label")
        print(f"--- {label} ---")
        print(f"scores: {case}")
        print(explain_single_recommendation(**case, coefficients=coefficients, scaler=scaler))     
        print()