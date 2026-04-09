import httpx
import json
import sys

BASE_URL = "https://yashraj00-emailenv.hf.space"

tasks_actions = {
    "email-classify": {"action_type": "classify", "priority": "urgent", "category": "bug_report"},
    "email-draft":    {"action_type": "draft",    "response_text": "Dear user, sincerely thank you for reaching out. We will resolve your issue promptly. Best regards, Support Team"},
    "email-triage":   {"action_type": "skip"},
}

print(f"Testing live server: {BASE_URL}\n")
all_ok = True

for task, action in tasks_actions.items():
    print(f"--- {task} ---")
    
    # Reset
    r = httpx.post(f"{BASE_URL}/reset", json={"task": task}, timeout=30)
    print(f"  /reset -> {r.status_code} {r.json() if r.status_code == 200 else r.text[:200]}")
    if r.status_code != 200:
        all_ok = False
        continue

    # Step
    r2 = httpx.post(f"{BASE_URL}/step", json=action, timeout=30)
    print(f"  /step  -> {r2.status_code}")
    if r2.status_code != 200:
        print(f"  ERROR: {r2.text[:300]}")
        all_ok = False
        continue

    result = r2.json()
    reward = result.get("reward")
    done   = result.get("done")
    info   = result.get("info", {})
    ok = isinstance(reward, float) and 0.0 < reward < 1.0
    if not ok:
        all_ok = False

    print(f"  reward={reward}  done={done}  in_range={ok}")
    if "breakdown" in info:
        print(f"  breakdown={info['breakdown']}")
    if "error" in info:
        print(f"  error={info['error']}")

    # Close
    httpx.post(f"{BASE_URL}/close", timeout=10)
    print()

print("=== RESULT:", "ALL IN RANGE" if all_ok else "SOME OUT OF RANGE / ERRORS", "===")
sys.exit(0 if all_ok else 1)
