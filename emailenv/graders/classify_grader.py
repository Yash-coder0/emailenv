from typing import Optional, Literal
from pydantic import BaseModel


class ClassifyGrader:
    def grade(self, action, email: dict) -> tuple[float, dict, str]:
        """
        Grade classification action.
        Returns: (score, breakdown_dict, reason_str)
        """
        breakdown = {}
        reason = ""
        
        if not action.priority or not action.category:
            return 0.0, {"priority": 0.0, "category": 0.0, "total": 0.0}, "Missing priority or category"
        
        priority_score = 0.4 if action.priority == email["ground_truth_priority"] else 0.0
        
        fuzzy_map = {
            ("bug_report", "support"): 0.5,
            ("support", "bug_report"): 0.5,
            ("sales", "support"): 0.2,
            ("support", "sales"): 0.2,
        }
        
        category_score = 0.3 if action.category == email["ground_truth_category"] else fuzzy_map.get((action.category, email["ground_truth_category"]), 0.0) * 0.3
        
        bonus_score = 0.0
        if action.priority == email["ground_truth_priority"] and action.category == email["ground_truth_category"]:
            bonus_score = 0.3
        
        total = priority_score + category_score + bonus_score
        
        penalty = 0.0
        if email["ground_truth_priority"] == "urgent" and action.priority == "low":
            penalty = -0.5
        
        total = max(0.0, total + penalty)
        total = min(1.0, total)
        
        breakdown = {
            "priority_correct": priority_score,
            "category_correct": category_score,
            "bonus": bonus_score,
            "penalty": penalty,
            "total": total
        }
        
        reason = f"Priority: {action.priority} (expected {email['ground_truth_priority']}). Category: {action.category} (expected {email['ground_truth_category']})."
        
        return total, breakdown, reason
