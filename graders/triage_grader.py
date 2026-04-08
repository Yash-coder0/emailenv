from typing import Optional
from pydantic import BaseModel
# FIXED: Removed 'emailenv.' prefix to match your flat directory structure
from graders.classify_grader import ClassifyGrader
from graders.draft_grader import DraftGrader

EPSILON = 1e-4


class TriageGrader:
    def __init__(self):
        self.classify_grader = ClassifyGrader()
        self.draft_grader = DraftGrader()

    def grade_action(self, action, email: dict) -> tuple[float, dict, str]:
        """
        Route to appropriate grader based on action type.
        Score is ALWAYS strictly between 0 and 1 (exclusive).
        """
        if action.action_type == "classify":
            score, breakdown, reason = self.classify_grader.grade(action, email)
        elif action.action_type == "draft":
            score, breakdown, reason = self.draft_grader.grade(action, email)
        elif action.action_type in ["archive", "flag", "skip"]:
            score = 0.11
            breakdown = {"action": score, "total": score}
            reason = f"Action {action.action_type} scored {score}"
        else:
            score = EPSILON
            breakdown = {"action": score, "total": score}
            reason = "Unknown action type"

        # FINAL SAFETY CLAMP: Ensure individual action scores are strictly (0, 1)
        score = max(EPSILON, min(1.0 - EPSILON, score))
        score = round(score, 4)
        return score, breakdown, reason

    def compute_final_score(self, scores: list, processed: int, steps: int, max_steps: int) -> float:
        """
        Compute final triage score with bonuses.
        Must be strictly between 0 and 1.
        """
        if not scores:
            return EPSILON

        mean_score = sum(scores) / len(scores)

        completion_bonus = 0.3 if processed == 10 else 0.0

        efficiency_bonus = 0.0
        if processed == 10:
            efficiency_bonus = min(0.2, 0.2 * (1.0 - steps / max_steps))

        raw_final = mean_score + completion_bonus + efficiency_bonus

        # CLAMP: score must never be exactly 0.0 or 1.0
        final = max(EPSILON, min(1.0 - EPSILON, raw_final))
        return round(final, 4)


if __name__ == "__main__":
    from types import SimpleNamespace

    grader = TriageGrader()
    email = {
        "ground_truth_priority": "urgent",
        "ground_truth_category": "bug_report",
        "key_questions_to_address": [],
        "expected_response_tone": "professional",
    }

    # Worst case: unknown action
    worst = SimpleNamespace(action_type="unknown", priority=None, category=None, response_text=None)
    score = grader.grade_action(worst, email)[0]
    assert 0.0 < score < 1.0, f"Score {score} out of range!"

    # Best case: perfect classify
    best = SimpleNamespace(action_type="classify", priority="urgent", category="bug_report", response_text=None)
    score = grader.grade_action(best, email)[0]
    assert 0.0 < score < 1.0, f"Score {score} out of range!"

    # compute_final_score edge cases
    score = grader.compute_final_score([], 0, 0, 30)
    assert 0.0 < score < 1.0, f"Score {score} out of range!"
    score = grader.compute_final_score([0.9] * 10, 10, 5, 30)
    assert 0.0 < score < 1.0, f"Score {score} out of range!"

    print("All grader bounds OK")