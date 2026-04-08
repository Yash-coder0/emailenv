from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
import json
from env import EmailEnv, EmailObservation, EmailAction, EmailReward


app = FastAPI(title="EmailEnv RL Server")

sessions = {}


class ResetRequest(BaseModel):
    task: str


class HealthResponse(BaseModel):
    status: str
    tasks: list


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict


def get_session_id(x_session_id: Optional[str] = Header(None)) -> str:
    return x_session_id or "default"


def get_env(session_id: str) -> EmailEnv:
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail=f"Session {session_id} not initialized. Call /reset first.")
    return sessions[session_id]


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "tasks": ["email-classify", "email-draft", "email-triage"]
    }


@app.post("/reset")
def reset(request: ResetRequest, x_session_id: Optional[str] = Header(None)):
    session_id = x_session_id or "default"
    try:
        env = EmailEnv(request.task)
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
            "info": info
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
