import sys
sys.path.insert(0, '.')
from graders.classify_grader import ClassifyGrader
from graders.draft_grader import DraftGrader
from graders.triage_grader import TriageGrader
from types import SimpleNamespace

ALL_OK = True

email_classify = {"ground_truth_priority": "urgent", "ground_truth_category": "bug_report"}
email_draft = {"key_questions_to_address": ["refund", "timeline"], "expected_response_tone": "formal"}
email_triage = {**email_classify, **email_draft}

def check(label, score):
    global ALL_OK
    ok = 0.0 < score < 1.0
    if not ok:
        ALL_OK = False
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}: {score}")

print("=== ClassifyGrader ===")
g = ClassifyGrader()
check("missing fields",    g.grade(SimpleNamespace(priority=None,     category=None),          email_classify)[0])
check("totally wrong",     g.grade(SimpleNamespace(priority="low",    category="sales"),       email_classify)[0])
check("partial match",     g.grade(SimpleNamespace(priority="urgent", category="sales"),       email_classify)[0])
check("perfect",           g.grade(SimpleNamespace(priority="urgent", category="bug_report"),  email_classify)[0])

print("=== DraftGrader ===")
g = DraftGrader()
check("empty response",    g.grade(SimpleNamespace(response_text=None), email_draft)[0])
check("short bad",         g.grade(SimpleNamespace(response_text="ok"), email_draft)[0])
best_text = (
    "Dear client, sincerely thank you for your message. "
    "Our refund policy covers all purchases within 30 days. "
    "The timeline for processing is 5 business days. "
    "Best regards, Support Team. " * 4
)
check("perfect response",  g.grade(SimpleNamespace(response_text=best_text), email_draft)[0])

print("=== TriageGrader ===")
g = TriageGrader()
check("unknown action",    g.grade_action(SimpleNamespace(action_type="unknown",  priority=None,     category=None,          response_text=None), email_triage)[0])
check("skip action",       g.grade_action(SimpleNamespace(action_type="skip",     priority=None,     category=None,          response_text=None), email_triage)[0])
check("classify perfect",  g.grade_action(SimpleNamespace(action_type="classify", priority="urgent", category="bug_report",  response_text=None), email_triage)[0])
check("final empty",       g.compute_final_score([], 0, 0, 30))
check("final max",         g.compute_final_score([0.9] * 10, 10, 5, 30))

print()
print("=== RESULT:", "ALL PASSED" if ALL_OK else "SOME FAILED", "===")
sys.exit(0 if ALL_OK else 1)
