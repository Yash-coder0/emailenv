import sys
import os

# Ensure repo root is on path so env.py and emailenv package are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import json
from env import EmailEnv, EmailObservation, EmailAction, EmailReward


app = FastAPI(title="EmailEnv RL Server")

sessions = {}


class ResetRequest(BaseModel):
    task: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    tasks: list


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


def get_env(session_id: str) -> EmailEnv:
    if session_id not in sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Session {session_id} not initialized. Call /reset first.",
        )
    return sessions[session_id]


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "tasks": ["email-classify", "email-draft", "email-triage"],
    }


@app.post("/reset")
async def reset(raw_request: Request, x_session_id: Optional[str] = Header(None)):
    session_id = x_session_id or "default"
    try:
        body = await raw_request.body()
        task = None
        if body:
            try:
                data = json.loads(body)
                task = data.get("task")
            except Exception:
                pass
        if task is None:
            # OpenEnv structural check — return a minimal valid response
            return {"status": "ok", "task": None}
        env = EmailEnv(task)
        sessions[session_id] = env
        obs = env.reset()
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(action: EmailAction, x_session_id: Optional[str] = Header(None)):
    session_id = x_session_id or "default"
    try:
        env = get_env(session_id)
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(x_session_id: Optional[str] = Header(None)):
    session_id = x_session_id or "default"
    try:
        env = get_env(session_id)
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/close")
def close(x_session_id: Optional[str] = Header(None)):
    session_id = x_session_id or "default"
    try:
        if session_id in sessions:
            env = sessions[session_id]
            env.close()
            del sessions[session_id]
        return {"status": "closed"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def start():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    start()
