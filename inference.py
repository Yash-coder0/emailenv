import os
import json
import sys
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert email triage assistant. 
When given an email, respond with ONLY valid JSON matching the action schema.
For classify: {"action_type":"classify","priority":"urgent|high|normal|low","category":"bug_report|sales|hr|finance|support|spam|other"}
For draft: {"action_type":"draft","response_text":"your full response here"}
For triage: use classify, draft, archive, flag, or skip as appropriate."""


def get_action(obs: dict) -> dict:
    prompt = f"""Task: {obs['task']}
Email ID: {obs['email_id']}
Subject: {obs['subject']}
From: {obs['sender']}
Body: {obs['body']}
Step: {obs['step']}
Emails remaining: {obs.get('emails_remaining', 0)}

Respond with the appropriate JSON action only."""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except:
        return {"action_type": "skip"}


def run_episode(task_name: str):
    import httpx
    MAX_STEPS = {"email-classify": 3, "email-draft": 5, "email-triage": 30}
    max_steps = MAX_STEPS.get(task_name, 10)
    
    reset_resp = httpx.post(f"{ENV_BASE_URL}/reset", json={"task": task_name}, timeout=30)
    obs = reset_resp.json()
    
    print(f"[START] task={task_name} env=emailenv model={MODEL_NAME}", flush=True)
    
    rewards = []
    step = 0
    done = False
    success = False
    
    try:
        while step < max_steps and not done:
            step += 1
            action = get_action(obs)
            action_str = json.dumps(action, separators=(',', ':'))
            
            step_resp = httpx.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
            result = step_resp.json()
            
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            obs = result.get("observation", obs)
            info = result.get("info", {})
            error = info.get("error") or "null"
            
            rewards.append(reward)
            print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)
        
        success = done and (sum(rewards) / len(rewards) > 0.3 if rewards else False)
    except Exception as e:
        error_msg = str(e).replace('\n', ' ')
        print(f"[ERROR] Exception occurred on step {step}: {error_msg}", flush=True)
        if step == 0:
            step = 1
            rewards = [0.0]
            print(f"[STEP] step={step} action=null reward=0.00 done=true error={error_msg}", flush=True)
    finally:
        try:
            httpx.post(f"{ENV_BASE_URL}/close", timeout=10)
        except:
            pass
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    tasks = ["email-classify", "email-draft", "email-triage"]
    for task in tasks:
        run_episode(task)
