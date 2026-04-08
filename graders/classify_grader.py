from typing import Optional, Literal
from pydantic import BaseModel

EPSILON = 1e-4


class ClassifyGrader:
    def grade(self, action, email: dict) -> tuple[float, dict, str]:
        """
        Grade classification action.
        Returns: (score, breakdown_dict, reason_str)
        Score is ALWAYS strictly between 0 and 1 (exclusive).
        """
        breakdown = {}
        reason = ""

        if not action.priority or not action.category:
            score = EPSILON
            breakdown = {"priority": 0.0, "category": 0.0, "total": score}
            return score, breakdown, "Missing priority or category"

        priority_score = 0.4 if action.priority == email["ground_truth_priority"] else 0.0

        fuzzy_map = {
            ("bug_report", "support"): 0.5,
            ("support", "bug_report"): 0.5,
            ("sales", "support"): 0.2,
            ("support", "sales"): 0.2,
        }

        category_score = (
            0.3
            if action.category == email["ground_truth_category"]
            else fuzzy_map.get((action.category, email["ground_truth_category"]), 0.0) * 0.3
        )

        bonus_score = 0.0
        if action.priority == email["ground_truth_priority"] and action.category == email["ground_truth_category"]:
            bonus_score = 0.3

        total = priority_score + category_score + bonus_score

        penalty = 0.0
        if email["ground_truth_priority"] == "urgent" and action.priority == "low":
            penalty = -0.5

        raw_score = max(0.0, total + penalty)

        # CLAMP: score must never be exactly 0.0 or 1.0
        score = max(EPSILON, min(1.0 - EPSILON, raw_score))
        score = round(score, 4)

        breakdown = {
            "priority_correct": priority_score,
            "category_correct": category_score,
            "bonus": bonus_score,
            "penalty": penalty,
            "total": score,
        }

        reason = (
            f"Priority: {action.priority} (expected {email['ground_truth_priority']}). "
            f"Category: {action.category} (expected {email['ground_truth_category']})."
        )

        return score, breakdown, reason


if __name__ == "__main__":
    from types import SimpleNamespace

    grader = ClassifyGrader()
    email = {
        "ground_truth_priority": "urgent",
        "ground_truth_category": "bug_report",
    }

    # Worst case: missing fields → EPSILON
    class BadAction:
        priority = None
        category = None

    score = grader.grade(BadAction(), email)[0]
    assert 0.0 < score < 1.0, f"Score {score} out of range!"

    # Wrong answer
    wrong = SimpleNamespace(priority="low", category="sales")
    score = grader.grade(wrong, email)[0]
    assert 0.0 < score < 1.0, f"Score {score} out of range!"

    # Perfect answer
    perfect = SimpleNamespace(priority="urgent", category="bug_report")
    score = grader.grade(perfect, email)[0]
    assert 0.0 < score < 1.0, f"Score {score} out of range!"

    print("All grader bounds OK")
