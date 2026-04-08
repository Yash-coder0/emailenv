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


# @app.post("/reset")
# async def reset(raw_request: Request, x_session_id: Optional[str] = Header(None)):
#     session_id = x_session_id or "default"
#     try:
#         body = await raw_request.body()
#         task = None
#         if body:
#             import json as _json
#             try:
#                 data = _json.loads(body)
#                 task = data.get("task")
#             except Exception:
#                 pass
#         if task is None:
#             # OpenEnv structural check — return a minimal valid response
#             return {"status": "ok", "task": None}
#         env = EmailEnv(task)
#         sessions[session_id] = env
#         obs = env.reset()
#         return obs.model_dump()
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

@app.post("/reset")
async def reset(request: Request, x_session_id: Optional[str] = Header(None)):
    session_id = x_session_id or "default"
    
    # The validator might send an empty body or {"task": "..."}
    # We parse it safely to avoid "Field Required" errors
    body = await request.json() if await request.body() else {}
    task = body.get("task")
    
    if task is None:
        # MANDATORY: The validator pings this to check if the server is alive
        return {"status": "ok", "message": "Server is ready for a task"}
        
    try:
        env = EmailEnv(task)
        sessions[session_id] = env
        obs = env.reset()
        # Ensure your EmailObservation has a .model_dump() method
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
