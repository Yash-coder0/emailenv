from typing import Optional
from pydantic import BaseModel
# FIXED: Removed 'emailenv.' prefix to match your flat directory structure
from graders.classify_grader import ClassifyGrader
from graders.draft_grader import DraftGrader

class TriageGrader:
    def __init__(self):
        self.classify_grader = ClassifyGrader()
        self.draft_grader = DraftGrader()
    
    def grade_action(self, action, email: dict) -> tuple[float, dict, str]:
        """
        Route to appropriate grader based on action type.
        """
        if action.action_type == "classify":
            score, breakdown, reason = self.classify_grader.grade(action, email)
        elif action.action_type == "draft":
            score, breakdown, reason = self.draft_grader.grade(action, email)
        elif action.action_type in ["archive", "flag", "skip"]:
            # NUDGED: Changed from 0.1 to 0.11 to stay away from strict boundaries
            score = 0.11
            breakdown = {"action": score, "total": score}
            reason = f"Action {action.action_type} scored {score}"
        else:
            # NUDGED: Changed from 0.0 to 0.01
            score = 0.01
            breakdown = {"action": score, "total": score}
            reason = "Unknown action type"
        
        # FINAL SAFETY NUDGE: Ensure individual action scores are strictly (0, 1)
        score = max(0.01, min(0.99, score))
        return score, breakdown, reason
    
    def compute_final_score(self, scores: list, processed: int, steps: int, max_steps: int) -> float:
        """
        Compute final triage score with bonuses.
        Must be strictly between 0 and 1.
        """
        if not scores:
            # NUDGED: Changed from 0.0 to 0.01
            return 0.01
        
        mean_score = sum(scores) / len(scores)
        
        completion_bonus = 0.3 if processed == 10 else 0.0
        
        efficiency_bonus = 0.0
        if processed == 10:
            efficiency_bonus = min(0.2, 0.2 * (1.0 - steps / max_steps))
        
        final = mean_score + completion_bonus + efficiency_bonus
        
        # CRITICAL FIX: The "Scaler Nudge"
        # This ensures the score is never 0.0 and never 1.0
        final = max(0.01, min(0.99, final))
        
        return final