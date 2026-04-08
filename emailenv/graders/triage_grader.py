from emailenv.graders.classify_grader import ClassifyGrader
from emailenv.graders.draft_grader import DraftGrader


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
            score = 0.1
            breakdown = {"action": score, "total": score}
            reason = f"Action {action.action_type} scored {score}"
        else:
            score = 0.0
            breakdown = {"action": score, "total": score}
            reason = "Unknown action type"
        
        return score, breakdown, reason
    
    def compute_final_score(self, scores: list, processed: int, steps: int, max_steps: int) -> float:
        """
        Compute final triage score with bonuses.
        """
        if not scores:
            return 0.0
        
        mean_score = sum(scores) / len(scores)
        
        completion_bonus = 0.3 if processed == 10 else 0.0
        
        efficiency_bonus = 0.0
        if processed == 10:
            efficiency_bonus = min(0.2, 0.2 * (1.0 - steps / max_steps))
        
        final = mean_score + completion_bonus + efficiency_bonus
        final = min(1.0, final)
        
        return final
