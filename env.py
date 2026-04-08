import json
import random
from typing import Optional, Literal
from pydantic import BaseModel, Field
from pathlib import Path

EPSILON = 1e-4


class EmailObservation(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    timestamp: str
    thread_length: int
    attachments: list[str]
    task: str
    step: int
    inbox_size: int = 0
    emails_remaining: int = 0


class EmailAction(BaseModel):
    action_type: Literal["classify", "draft", "archive", "flag", "respond", "skip"]
    priority: Optional[Literal["urgent", "high", "normal", "low"]] = None
    category: Optional[Literal["bug_report", "sales", "hr", "finance", "support", "spam", "other"]] = None
    response_text: Optional[str] = None
    email_id: Optional[str] = None


class EmailReward(BaseModel):
    value: float
    breakdown: dict
    reason: str


class EmailEnv:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.emails_data = self._load_emails()
        self.task_config = {
            "email-classify": {"max_steps": 3, "pool_size": 10},
            "email-draft": {"max_steps": 5, "pool_size": 5},
            "email-triage": {"max_steps": 30, "pool_size": 50}
        }
        self.config = self.task_config.get(task_name, {"max_steps": 10, "pool_size": 10})
        self.reset()

    def _load_emails(self) -> list:
        data_path = Path(__file__).parent / "data" / "emails.json"
        with open(data_path, "r") as f:
            return json.load(f)

    def reset(self) -> EmailObservation:
        pool_size = min(self.config["pool_size"], len(self.emails_data))
        self.email_pool = random.sample(self.emails_data, pool_size)
        self.current_idx = 0
        self.step_count = 0
        self.scores = []
        self.processed_count = 0

        if self.task_name == "email-classify":
            self.current_email = self.email_pool[0]
        elif self.task_name == "email-draft":
            self.current_email = self.email_pool[0]
        else:  # triage
            self.current_email = self.email_pool[0]

        return self._make_observation()

    def _make_observation(self) -> EmailObservation:
        return EmailObservation(
            email_id=self.current_email["id"],
            subject=self.current_email["subject"],
            body=self.current_email["body"],
            sender=self.current_email["sender"],
            timestamp=self.current_email["timestamp"],
            thread_length=self.current_email["thread_length"],
            attachments=self.current_email.get("attachments", []),
            task=self.task_name,
            step=self.step_count,
            inbox_size=len(self.email_pool),
            emails_remaining=len(self.email_pool) - self.current_idx
        )

    def step(self, action: EmailAction) -> tuple[EmailObservation, float, bool, dict]:
        self.step_count += 1
        raw_reward = 0.0
        done = False
        info = {}

        if self.task_name == "email-classify":
            from emailenv.graders.classify_grader import ClassifyGrader
            grader = ClassifyGrader()
            raw_reward, breakdown, reason = grader.grade(action, self.current_email)
            info = {"breakdown": breakdown, "reason": reason, "ground_truth": {
                "priority": self.current_email["ground_truth_priority"],
                "category": self.current_email["ground_truth_category"]
            }}
            done = self.current_idx >= len(self.email_pool) - 1 or self.step_count >= self.config["max_steps"]
            if not done and self.current_idx < len(self.email_pool) - 1:
                self.current_idx += 1
                self.current_email = self.email_pool[self.current_idx]

        elif self.task_name == "email-draft":
            from emailenv.graders.draft_grader import DraftGrader
            grader = DraftGrader()
            raw_reward, breakdown, reason = grader.grade(action, self.current_email)
            info = {"breakdown": breakdown, "reason": reason, "expected_tone": self.current_email.get("expected_response_tone")}
            done = self.current_idx >= len(self.email_pool) - 1 or self.step_count >= self.config["max_steps"]
            if not done and self.current_idx < len(self.email_pool) - 1:
                self.current_idx += 1
                self.current_email = self.email_pool[self.current_idx]

        else:  # triage
            from emailenv.graders.triage_grader import TriageGrader
            grader = TriageGrader()
            raw_reward, breakdown, reason = grader.grade_action(action, self.current_email)
            info = {"breakdown": breakdown, "reason": reason}
            done = self.current_idx >= len(self.email_pool) - 1 or self.step_count >= self.config["max_steps"]
            if not done and self.current_idx < len(self.email_pool) - 1:
                self.current_idx += 1
                self.current_email = self.email_pool[self.current_idx]

        # Safety net: clamp reward regardless of grader output
        reward = max(EPSILON, min(1.0 - EPSILON, raw_reward))
        reward = round(reward, 4)

        self.scores.append(reward)
        self.processed_count += 1

        obs = self._make_observation()
        return obs, reward, done, info

    def state(self) -> dict:
        return {
            "task": self.task_name,
            "step": self.step_count,
            "current_email_id": self.current_email["id"],
            "processed": self.processed_count,
            "scores": self.scores,
            "mean_score": sum(self.scores) / len(self.scores) if self.scores else 0.0
        }

    def close(self):
        pass
