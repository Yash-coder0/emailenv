"""
Full episode test against live HF Space - simulates exactly what the validator does.
Runs ALL max_steps for each task and checks every single reward.
"""
import httpx
import sys

BASE_URL = "https://yashraj00-emailenv.hf.space"

TASK_CONFIG = {
    "email-classify": {"max_steps": 3,  "action": {"action_type": "classify", "priority": "urgent", "category": "bug_report"}},
    "email-draft":    {"max_steps": 5,  "action": {"action_type": "draft",    "response_text": "Dear user, sincerely thank you. Best regards, Support Team."}},
    "email-triage":   {"max_steps": 10, "action": {"action_type": "skip"}},
}

print(f"Full episode test against {BASE_URL}\n")
overall_ok = True

for task, cfg in TASK_CONFIG.items():
    print(f"=== {task} (max_steps={cfg['max_steps']}) ===")

    r = httpx.post(f"{BASE_URL}/reset", json={"task": task}, timeout=30)
    if r.status_code != 200:
        print(f"  FAIL /reset returned {r.status_code}: {r.text[:200]}")
        overall_ok = False
        continue

    rewards = []
    done = False
    step = 0
    task_ok = True

    while step < cfg["max_steps"] and not done:
        step += 1
        r2 = httpx.post(f"{BASE_URL}/step", json=cfg["action"], timeout=30)

        if r2.status_code != 200:
            print(f"  [FAIL] step={step} -> HTTP {r2.status_code}: {r2.text[:200]}")
            task_ok = False
            overall_ok = False
            break

        result = r2.json()
        reward = result.get("reward")
        done   = result.get("done", False)
        in_range = isinstance(reward, (int, float)) and 0.0 < float(reward) < 1.0

        if not in_range:
            task_ok = False
            overall_ok = False

        rewards.append(reward)
        print(f"  step={step}  reward={reward}  done={done}  in_range={in_range}")

    mean = sum(rewards) / len(rewards) if rewards else 0.0
    mean_ok = 0.0 < mean < 1.0
    if not mean_ok:
        overall_ok = False

    print(f"  -> mean={round(mean,4)}  mean_in_range={mean_ok}  task={'PASS' if task_ok else 'FAIL'}")
    httpx.post(f"{BASE_URL}/close", timeout=10)
    print()

print("=" * 50)
print("FINAL RESULT:", "ALL PASSED" if overall_ok else "SOME FAILED")
sys.exit(0 if overall_ok else 1)
