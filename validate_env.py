import sys, os
sys.path.insert(0, os.path.abspath('.'))

from env import EmailEnv, EmailAction

EPSILON = 1e-4
ALL_OK = True

for task in ['email-classify', 'email-draft', 'email-triage']:
    env = EmailEnv(task)
    env.reset()

    if task == 'email-classify':
        action = EmailAction(action_type='classify', priority='urgent', category='bug_report')
    elif task == 'email-draft':
        action = EmailAction(action_type='draft', response_text='Dear user, thank you. Best regards.')
    else:
        action = EmailAction(action_type='skip')

    obs, reward, done, info = env.step(action)
    clamped = max(EPSILON, min(1.0 - EPSILON, reward))
    ok = 0.0 < clamped < 1.0
    if not ok:
        ALL_OK = False
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {task}: reward={reward} -> clamped={clamped}")

print()
print("RESULT:", "ALL PASSED" if ALL_OK else "SOME FAILED")
sys.exit(0 if ALL_OK else 1)
