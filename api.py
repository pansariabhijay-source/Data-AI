"""
Axiom FastAPI Backend — Multi-Tier AI Platform API

Provides REST endpoints for:
- Dataset upload with auto-visualization
- Full pipeline execution (Free mode)
- Custom workflow execution (Enterprise mode)
- Independent agent execution (Enterprise mode)
- Per-agent output retrieval
- Visualization generation and serving
- Experiment tracking
- Real-time status polling
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import sys
import shutil
import threading
import traceback
import time
from pathlib import Path
from queue import Empty
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from core.config import load_settings
from core.logging_config import setup_logging, get_logger
from core.state import PipelineState
from core.utils import generate_run_id, get_timestamp, ensure_directory
from core.pipeline_worker import build_result_payload, run_pipeline_child
from database import get_db, User, UserSession, PromptHistory, UserPreference, SessionLocal

logger = get_logger("api")


def _warm_up_viz() -> None:
    """Prime heavy imports (matplotlib/seaborn + font cache) in a daemon thread.

    The first dataset upload renders charts; without warming, that very first
    request eats a ~5s one-time import/font-cache cost and *feels* like a hang.
    """
    def _prime() -> None:
        try:
            import pandas as pd
            from visualization.engine import generate_free_mode_viz
            generate_free_mode_viz(pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]}))
            logger.info("Visualization engine warmed up")
        except Exception:
            logger.warning("Viz warm-up failed (non-fatal)", exc_info=True)

    threading.Thread(target=_prime, daemon=True).start()


from contextlib import asynccontextmanager


def _active_run_ids() -> set[str]:
    """Run IDs that are currently in flight (must never be cleaned up)."""
    return {
        rid for rid, info in _active_runs.items()
        if isinstance(info, dict) and info.get("status") in ("starting", "running")
    }


def _run_artifact_cleanup() -> None:
    """Best-effort disk hygiene — enforce the artifact-retention policy."""
    try:
        from core.maintenance import cleanup_artifacts
        cleanup_artifacts(active_run_ids=_active_run_ids())
    except Exception:
        logger.warning("Artifact cleanup failed", exc_info=True)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Startup
    _warm_up_viz()
    # Reclaim disk from old runs at boot so a long-idle install self-heals.
    threading.Thread(target=_run_artifact_cleanup, daemon=True).start()
    yield
    # Shutdown (nothing to clean up — background runs are daemon threads)


app = FastAPI(title="Axiom — Autonomous AI Platform", version="2.0.0", lifespan=_lifespan)

# Lock CORS to explicitly allowed origins. Defaults to the local Next.js dev
# server; override via the ALLOWED_ORIGINS env var (comma-separated) in prod.
# A wildcard "*" with credentials is invalid per the CORS spec and insecure, so
# we never emit it.
_allowed_origins = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth Config ─────────────────────────────────────────────────────────────

import bcrypt

security = HTTPBearer()

# Session token lifetime, in days (configurable per deployment).
try:
    _SESSION_TTL_DAYS = int(os.environ.get("SESSION_TTL_DAYS", "7"))
except ValueError:
    _SESSION_TTL_DAYS = 7

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db=Depends(get_db)) -> User:
    token = credentials.credentials
    session = db.query(UserSession).filter(UserSession.token == token).first()
    if not session or session.is_expired:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return session.user

# ── Global pipeline state tracking ──────────────────────────────────────────

_active_runs: dict[str, dict] = {}  # run_id -> {state, status, logs, thread, agent_outputs, ...}

# Run state is mirrored to disk so status/results survive a server restart.
# Keys that hold live, non-serializable objects are never persisted.
_persist_lock = threading.Lock()
_ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_DIR", "artifacts"))
_UPLOADS_ROOT = Path("data") / "uploads"
_NON_PERSISTED_KEYS = {"pipeline_state"}

# ── Upload limits / tuning ──────────────────────────────────────────────────
# Hard cap on uploaded dataset size. The frontend uploads straight to the
# backend (bypassing the Next proxy), which streams to disk in chunks, so this
# can be generous. Override via MAX_UPLOAD_MB.
try:
    _MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "1024"))
except ValueError:
    _MAX_UPLOAD_MB = 1024
_MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MB streaming chunks

# Visualizations are rendered on at most this many rows. Charts are statistical
# summaries, so a representative sample looks identical to the full set while
# keeping render time bounded regardless of dataset size. Override via
# VIZ_SAMPLE_ROWS.
try:
    _VIZ_SAMPLE_ROWS = int(os.environ.get("VIZ_SAMPLE_ROWS", "50000"))
except ValueError:
    _VIZ_SAMPLE_ROWS = 50000

# Run the full pipeline in a child PROCESS (default) so its CPU-heavy work never
# holds the API's GIL — uploads/status stay responsive while a pipeline runs.
# Set PIPELINE_ISOLATION=0 to fall back to the in-thread runner. The "spawn"
# context is used for cross-platform consistency (and is the only option on
# Windows). Spawning the child re-imports core.pipeline_worker (not this module),
# so the child never starts a web server.
try:
    _PIPELINE_ISOLATION = int(os.environ.get("PIPELINE_ISOLATION", "1"))
except ValueError:
    _PIPELINE_ISOLATION = 1
_MP_CTX = multiprocessing.get_context("spawn")


def _run_state_path(run_id: str) -> Path:
    return _ARTIFACT_ROOT / run_id / "run_state.json"


def _persist_run(run_id: str) -> None:
    """Atomically snapshot a run's serializable state to disk."""
    run = _active_runs.get(run_id)
    if run is None:
        return
    try:
        snapshot = _sanitize_for_json(
            {k: v for k, v in run.items() if k not in _NON_PERSISTED_KEYS}
        )
        path = _run_state_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with _persist_lock:
            tmp.write_text(json.dumps(snapshot, default=str), encoding="utf-8")
            tmp.replace(path)
    except Exception:
        logger.warning("Failed to persist run state for %s", run_id, exc_info=True)


def _load_run(run_id: str) -> Optional[dict]:
    """Return a run dict, rehydrating from disk into the cache on a miss."""
    run = _active_runs.get(run_id)
    if run is not None:
        return run
    path = _run_state_path(run_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _active_runs[run_id] = data
        return data
    except Exception:
        logger.warning("Failed to load run state for %s", run_id, exc_info=True)
        return None


def _require_run(run_id: str, user: "User") -> dict:
    """Fetch a run (memory or disk) and enforce that it belongs to ``user``."""
    run = _load_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    owner = run.get("user_id")
    if owner is not None and owner != user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this run")
    return run


def _authorize_run_id(run_id: str, user: "User", db) -> None:
    """Ownership check for filesystem-only resources (markdown report, SHAP)."""
    run = _load_run(run_id)
    if run is not None:
        owner = run.get("user_id")
        if owner is not None and owner != user.id:
            raise HTTPException(status_code=403, detail="You do not have access to this run")
        return
    history = db.query(PromptHistory).filter(PromptHistory.run_id == run_id).first()
    if history is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if history.user_id != user.id:
        raise HTTPException(status_code=403, detail="You do not have access to this run")


def _safe_data_path(data_path: str) -> str:
    """Validate a user-supplied dataset path stays within the uploads directory.

    Prevents path-traversal / arbitrary file reads by rejecting anything that
    does not resolve to a file under ``data/uploads``.
    """
    uploads_root = _UPLOADS_ROOT.resolve()
    try:
        resolved = Path(data_path).resolve()
    except (OSError, ValueError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid data path")
    if resolved != uploads_root and uploads_root not in resolved.parents:
        raise HTTPException(status_code=400, detail="data_path must reference an uploaded dataset")
    if not resolved.exists():
        raise HTTPException(status_code=400, detail=f"Data file not found: {data_path}")
    return str(resolved)

# ── Request Models ──────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email: str
    username: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class AgentRunRequest(BaseModel):
    run_id: str
    config: Optional[dict] = None

class WorkflowRunRequest(BaseModel):
    agents: list[str]
    data_path: str
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    config: Optional[dict] = None

class VizGenerateRequest(BaseModel):
    type: str
    params: Optional[dict] = None

# ── Auth Endpoints ──────────────────────────────────────────────────────────

from datetime import datetime, timedelta, timezone

@app.post("/api/auth/signup")
async def signup(req: SignupRequest, db=Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(
        email=req.email,
        username=req.username,
        password_hash=get_password_hash(req.password)
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = UserSession.generate_token()
    session = UserSession(
        user_id=user.id,
        token=token,
        expires_at=datetime.now(timezone.utc) + timedelta(days=_SESSION_TTL_DAYS)
    )
    db.add(session)
    db.commit()
    
    return {"status": "success", "token": token, "user": user.to_dict()}

@app.post("/api/auth/login")
async def login(req: LoginRequest, db=Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    user.last_login = datetime.now(timezone.utc)
    
    token = UserSession.generate_token()
    session = UserSession(
        user_id=user.id,
        token=token,
        expires_at=datetime.now(timezone.utc) + timedelta(days=_SESSION_TTL_DAYS)
    )
    db.add(session)
    db.commit()
    
    return {"status": "success", "token": token, "user": user.to_dict()}

@app.get("/api/auth/me")
async def get_me(user: User = Depends(get_current_user)):
    return {"status": "success", "user": user.to_dict()}

@app.post("/api/auth/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security), db=Depends(get_db)):
    token = credentials.credentials
    session = db.query(UserSession).filter(UserSession.token == token).first()
    if session:
        db.delete(session)
        db.commit()
    return {"status": "success"}

@app.get("/api/auth/history")
async def get_prompt_history(user: User = Depends(get_current_user), db=Depends(get_db)):
    history = db.query(PromptHistory).filter(PromptHistory.user_id == user.id).order_by(PromptHistory.created_at.desc()).all()
    return {"status": "success", "history": [h.to_dict() for h in history]}


# ── Workspace preferences ────────────────────────────────────────────────────

class WorkspaceModeRequest(BaseModel):
    mode: str  # "free" | "enterprise"


def _get_or_create_pref(db, user_id: int) -> UserPreference:
    pref = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
    if pref is None:
        pref = UserPreference(user_id=user_id)
        db.add(pref)
        db.commit()
        db.refresh(pref)
    return pref


@app.get("/api/workspace")
async def get_workspace(user: User = Depends(get_current_user), db=Depends(get_db)):
    """Return the user's durable workspace preferences (selected mode, theme, onboarding)."""
    pref = _get_or_create_pref(db, user.id)
    return {"status": "success", "workspace": pref.to_dict()}


@app.post("/api/workspace/mode")
async def set_workspace_mode(
    req: WorkspaceModeRequest,
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Persist the user's selected workspace mode so it survives across sessions/devices."""
    if req.mode not in ("free", "enterprise"):
        raise HTTPException(status_code=400, detail="mode must be 'free' or 'enterprise'")
    pref = _get_or_create_pref(db, user.id)
    pref.selected_mode = req.mode
    pref.onboarding_completed = True
    db.commit()
    return {"status": "success", "workspace": pref.to_dict()}

# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health_check():
    """Lightweight liveness probe — never touches the DB or disk-heavy paths.

    The frontend can hit this to distinguish 'backend down' from 'request slow',
    and it doubles as a load-balancer/uptime healthcheck.
    """
    return {"status": "ok", "service": "axiom-api", "active_runs": len(_active_runs)}


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML."""
    html_path = Path(__file__).parent / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="Dashboard not found")


def _detect_and_read_csv(filepath: Path, **kwargs) -> "pd.DataFrame":
    """Read a CSV with automatic encoding detection.

    Tries utf-8 first, then falls back through common encodings.
    On success the file is re-saved as UTF-8 so every downstream
    consumer (agents, viz engine, etc.) can rely on UTF-8.
    """
    import pandas as pd

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "iso-8859-1"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, **kwargs)
            # Re-save as UTF-8 so all later reads just work
            if enc != "utf-8" and "nrows" not in kwargs and "usecols" not in kwargs:
                df.to_csv(filepath, index=False, encoding="utf-8")
                logger.info(f"Re-encoded {filepath.name} from {enc} → utf-8")
            return df
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise last_err or ValueError(f"Could not decode {filepath.name} with any known encoding")


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf with None for JSON compliance."""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(i) for i in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


async def _stream_upload_to_disk(file: UploadFile, dest: Path) -> int:
    """Stream an upload to ``dest`` in chunks, enforcing the size cap.

    Streaming avoids buffering the whole (potentially hundreds-of-MB) file in
    memory and lets us abort early once the cap is exceeded. Returns the number
    of bytes written.
    """
    size = 0
    too_big = False
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    too_big = True
                    break
                f.write(chunk)
    except OSError as e:
        # Disk full (ENOSPC) or other write failure — clean up the partial file
        # and return a clear, actionable error instead of a raw 500 traceback.
        dest.unlink(missing_ok=True)
        import errno
        if e.errno == errno.ENOSPC:
            raise HTTPException(
                status_code=507,
                detail="The server ran out of disk space while saving the upload. "
                       "Free up space and try again.",
            )
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
    if too_big:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {_MAX_UPLOAD_MB} MB upload limit",
        )
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    return size


def _profile_dataset(dest: Path) -> dict:
    """Read, profile, and visualize an uploaded dataset.

    Pure CPU/IO work — must be called via ``run_in_threadpool`` so it never
    blocks the asyncio event loop (a long render here would otherwise freeze
    every other request and look like the backend is unreachable).
    """
    import pandas as pd

    try:
        df_full = _detect_and_read_csv(dest)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The file is empty or has no parsable columns")
    except (pd.errors.ParserError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file as a table: {e}")

    if df_full.shape[1] == 0:
        raise HTTPException(status_code=400, detail="CSV contains no columns")
    if len(df_full) == 0:
        raise HTTPException(status_code=400, detail="CSV contains no data rows")

    columns = list(df_full.columns)
    dtypes = {col: str(dtype) for col, dtype in df_full.dtypes.items()}
    n_rows = len(df_full)
    preview = df_full.head(5).to_dict(orient="records")

    # Render charts on a bounded sample so upload latency stays flat regardless
    # of dataset size. Statistical charts are visually identical on a sample.
    if n_rows > _VIZ_SAMPLE_ROWS:
        df_viz = df_full.sample(n=_VIZ_SAMPLE_ROWS, random_state=0)
    else:
        df_viz = df_full

    visualizations: list = []
    try:
        from visualization.engine import generate_free_mode_viz
        visualizations = generate_free_mode_viz(df_viz)
    except Exception as viz_err:
        logger.warning(f"Auto-visualization failed: {viz_err}")

    return {
        "columns": columns,
        "dtypes": dtypes,
        "n_rows": n_rows,
        "preview": preview,
        "visualizations": visualizations,
    }


@app.post("/api/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """Upload a dataset file and auto-generate visualizations."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    allowed = {".csv", ".tsv", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Strip any directory components — never trust a client-supplied filename.
    safe_name = Path(file.filename).name
    # Namespace uploads per user so two users uploading "data.csv" never collide
    # or overwrite each other (a cross-user data leak). _safe_data_path still
    # confines downstream reads to the uploads root, so the subdir is allowed.
    upload_dir = ensure_directory(f"data/uploads/{user.id}")
    dest = upload_dir / safe_name

    # 1) Stream to disk (fast, I/O-bound, safe to await on the event loop).
    size = await _stream_upload_to_disk(file, dest)
    logger.info(f"File uploaded: {safe_name} ({size} bytes)")

    # 2) Profile + visualize off the event loop so concurrent requests (status
    #    polls, auth, other uploads) stay responsive during the heavy work.
    try:
        payload = await run_in_threadpool(_profile_dataset, dest)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Dataset profiling failed")
        raise HTTPException(status_code=500, detail=f"Could not parse dataset: {e}")

    payload.update({
        "status": "success",
        "filename": safe_name,
        "path": str(dest),
    })
    return _sanitize_for_json(payload)


@app.post("/api/init-run")
async def init_run(
    data_path: str = Form(...),
    target_column: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Initialize a run entry without starting any pipeline.
    Returns a run_id that can be used to invoke individual agents in the Enterprise Agent Console.
    """
    data_path = _safe_data_path(data_path)

    run_id = generate_run_id()

    _active_runs[run_id] = {
        "status": "initialized",
        "run_id": run_id,
        "data_path": data_path,
        "target_column": target_column,
        "mode": "enterprise",
        "started_at": get_timestamp(),
        "current_stage": None,
        "completed_stages": [],
        "logs": [{"time": get_timestamp(), "msg": "Run initialized — ready for independent agent execution"}],
        "result": None,
        "error": None,
        "agent_outputs": {},
        "visualizations": [],
        "pipeline_state": None,   # will hold the shared PipelineState across agents
        "user_id": user.id,
    }
    _persist_run(run_id)

    history = PromptHistory(
        user_id=user.id,
        run_id=run_id,
        data_path=data_path,
        target_column=target_column,
        mode="enterprise",
        status="running",
    )
    db.add(history)
    db.commit()

    return {"status": "initialized", "run_id": run_id}


@app.post("/api/run")
async def start_pipeline(
    data_path: str = Form(...),
    target_column: Optional[str] = Form(None),
    mode: Optional[str] = Form("free"),
    user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """Start a full pipeline run in the background."""
    data_path = _safe_data_path(data_path)

    run_id = generate_run_id()

    _active_runs[run_id] = {
        "status": "starting",
        "run_id": run_id,
        "data_path": data_path,
        "target_column": target_column,
        "mode": mode,
        "started_at": get_timestamp(),
        "current_stage": None,
        "completed_stages": [],
        "logs": [],
        "result": None,
        "error": None,
        "agent_outputs": {},
        "visualizations": [],
        "user_id": user.id,
    }
    _persist_run(run_id)

    # Record history
    history = PromptHistory(
        user_id=user.id,
        run_id=run_id,
        data_path=data_path,
        target_column=target_column,
        mode=mode,
        status="running"
    )
    db.add(history)
    db.commit()

    # Run pipeline in background thread
    thread = threading.Thread(
        target=_run_pipeline_background,
        args=(run_id, data_path, target_column, mode),
        daemon=True,
    )
    thread.start()

    return {"status": "started", "run_id": run_id}


@app.post("/api/workflow/run")
async def start_workflow(
    request: WorkflowRunRequest,
    user: User = Depends(get_current_user),
    db=Depends(get_db)
):
    """Start a custom agent workflow (Enterprise mode)."""
    data_path = _safe_data_path(request.data_path)

    run_id = generate_run_id()

    _active_runs[run_id] = {
        "status": "starting",
        "run_id": run_id,
        "data_path": data_path,
        "target_column": request.target_column,
        "mode": "enterprise",
        "started_at": get_timestamp(),
        "current_stage": None,
        "completed_stages": [],
        "logs": [],
        "result": None,
        "error": None,
        "agent_outputs": {},
        "visualizations": [],
        "workflow_agents": request.agents,
        "user_id": user.id,
    }
    _persist_run(run_id)

    # Record history
    history = PromptHistory(
        user_id=user.id,
        run_id=run_id,
        data_path=request.data_path,
        target_column=request.target_column,
        mode="enterprise",
        status="running"
    )
    db.add(history)
    db.commit()

    thread = threading.Thread(
        target=_run_workflow_background,
        args=(run_id, data_path, request.target_column, request.agents, request.problem_type),
        daemon=True,
    )
    thread.start()

    return {"status": "started", "run_id": run_id}


@app.post("/api/agent/{agent_name}/run")
async def run_agent_endpoint(
    agent_name: str,
    request: AgentRunRequest,
    user: User = Depends(get_current_user),
):
    """Run a single agent independently (Enterprise mode)."""
    _require_run(request.run_id, user)

    # Run agent in background
    thread = threading.Thread(
        target=_run_single_agent_background,
        args=(request.run_id, agent_name),
        daemon=True,
    )
    thread.start()

    return {"status": "started", "agent": agent_name, "run_id": request.run_id}


@app.get("/api/agent/{agent_name}/output/{run_id}")
async def get_agent_output(
    agent_name: str,
    run_id: str,
    user: User = Depends(get_current_user),
):
    """Get the output of a specific agent for a run."""
    run = _require_run(run_id, user)

    agent_outputs = run.get("agent_outputs", {})
    if agent_name not in agent_outputs:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} output not found")

    return agent_outputs[agent_name]


@app.get("/api/status/{run_id}")
async def get_status(run_id: str, user: User = Depends(get_current_user)):
    """Get the status of a pipeline run with agent outputs."""
    run = _require_run(run_id, user)
    return _sanitize_for_json({
        "run_id": run_id,
        "status": run["status"],
        "mode": run.get("mode", "free"),
        "current_stage": run.get("current_stage"),
        "completed_stages": run.get("completed_stages", []),
        "started_at": run.get("started_at"),
        "logs": run.get("logs", [])[-30:],  # Last 30 log entries
        "error": run.get("error"),
        "agent_outputs": {
            k: v for k, v in run.get("agent_outputs", {}).items()
        },
    })


@app.get("/api/results/{run_id}")
async def get_results(run_id: str, user: User = Depends(get_current_user)):
    """Get full results of a completed pipeline run."""
    run = _require_run(run_id, user)
    if run["status"] not in ("completed", "failed"):
        return {"status": run["status"], "message": "Pipeline still running"}

    result = run.get("result", {"status": "no results"})
    # Attach agent outputs
    result["agent_outputs"] = run.get("agent_outputs", {})
    return _sanitize_for_json(result)


@app.get("/api/report/{run_id}")
async def get_report(run_id: str, user: User = Depends(get_current_user), db=Depends(get_db)):
    """Get the markdown report for a completed run."""
    _authorize_run_id(run_id, user, db)
    report_path = Path("reports") / run_id / "pipeline_report.md"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return {"report": report_path.read_text(encoding="utf-8")}


@app.get("/api/report/{run_id}/pdf")
async def get_report_pdf(run_id: str, user: User = Depends(get_current_user)):
    """Render the full pipeline run as a styled PDF and stream it back."""
    run = _require_run(run_id, user)
    if run["status"] not in ("completed", "failed"):
        raise HTTPException(status_code=400, detail="Pipeline is still running")

    result = run.get("result") or {}
    visualizations = run.get("visualizations") or []

    # Optional inputs (loaded best-effort)
    shap_data: Optional[dict] = None
    shap_path = Path("artifacts") / run_id / "shap_importance.json"
    if shap_path.exists():
        try:
            shap_data = json.loads(shap_path.read_text(encoding="utf-8"))
        except Exception:
            shap_data = None

    md_report: Optional[str] = None
    md_path = Path("reports") / run_id / "pipeline_report.md"
    if md_path.exists():
        try:
            md_report = md_path.read_text(encoding="utf-8")
        except Exception:
            md_report = None

    run_info = {
        "run_id": run_id,
        "mode": run.get("mode", "free"),
        "started_at": run.get("started_at"),
        "completed_at": run.get("completed_at"),
        "duration_seconds": _duration_seconds(run.get("started_at"), run.get("completed_at")),
    }

    def _build() -> bytes:
        from visualization.pdf_report import build_pdf
        return build_pdf(
            run_info=run_info,
            result=result,
            visualizations=visualizations,
            shap_data=shap_data,
            markdown_report=md_report,
        )

    try:
        # PDF assembly is CPU-heavy — keep it off the event loop.
        pdf_bytes = await run_in_threadpool(_build)
    except Exception as e:
        logger.exception("PDF generation failed")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    from fastapi.responses import StreamingResponse
    filename = f"axiom-report-{run_id[:8]}.pdf"
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/report/{run_id}/notebook")
async def get_report_notebook(
    run_id: str,
    kind: str = Query("report"),
    user: User = Depends(get_current_user),
):
    """Export the run as a Jupyter notebook (.ipynb).

    kind="report" (default): report narrative + charts + code to load the champion.
    kind="reproduce": a standalone notebook that re-runs Axiom's actual pipeline
    agent-by-agent to reproduce the result from scratch.
    """
    run = _require_run(run_id, user)
    if run["status"] not in ("completed", "failed"):
        raise HTTPException(status_code=400, detail="Pipeline is still running")

    result = run.get("result") or {}
    visualizations = run.get("visualizations") or []

    if kind == "reproduce":
        data_path = run.get("data_path")
        target = run.get("target_column")

        def _build_repro() -> bytes:
            from visualization.notebook_report import build_reproduction_notebook
            return build_reproduction_notebook(
                run_info={"run_id": run_id, "mode": run.get("mode", "free")},
                result=result, data_path=data_path, target=target,
            )

        try:
            nb_bytes = await run_in_threadpool(_build_repro)
        except Exception as e:
            logger.exception("Reproduction notebook generation failed")
            raise HTTPException(status_code=500, detail=f"Notebook generation failed: {e}")

        from fastapi.responses import StreamingResponse
        filename = f"axiom-reproduce-{run_id[:8]}.ipynb"
        return StreamingResponse(
            iter([nb_bytes]),
            media_type="application/x-ipynb+json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    shap_data: Optional[dict] = None
    shap_path = Path("artifacts") / run_id / "shap_importance.json"
    if shap_path.exists():
        try:
            shap_data = json.loads(shap_path.read_text(encoding="utf-8"))
        except Exception:
            shap_data = None

    md_report: Optional[str] = None
    md_path = Path("reports") / run_id / "pipeline_report.md"
    if md_path.exists():
        try:
            md_report = md_path.read_text(encoding="utf-8")
        except Exception:
            md_report = None

    run_info = {"run_id": run_id, "mode": run.get("mode", "free")}

    def _build() -> bytes:
        from visualization.notebook_report import build_notebook
        return build_notebook(
            run_info=run_info,
            result=result,
            shap_data=shap_data,
            markdown_report=md_report,
            visualizations=visualizations,
        )

    try:
        nb_bytes = await run_in_threadpool(_build)
    except Exception as e:
        logger.exception("Notebook generation failed")
        raise HTTPException(status_code=500, detail=f"Notebook generation failed: {e}")

    from fastapi.responses import StreamingResponse
    filename = f"axiom-report-{run_id[:8]}.ipynb"
    return StreamingResponse(
        iter([nb_bytes]),
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/shap/{run_id}")
async def get_shap_data(run_id: str, user: User = Depends(get_current_user), db=Depends(get_db)):
    """Get SHAP feature importance data for a completed run."""
    _authorize_run_id(run_id, user, db)
    shap_path = Path("artifacts") / run_id / "shap_importance.json"
    if not shap_path.exists():
        raise HTTPException(status_code=404, detail="SHAP data not available for this run")
    try:
        data = json.loads(shap_path.read_text(encoding="utf-8"))
        return {"run_id": run_id, "shap_importance": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _duration_seconds(started_at: Optional[str], completed_at: Optional[str]) -> Optional[float]:
    """Compute wall-clock duration between two ISO-8601 UTC timestamps."""
    if not started_at or not completed_at:
        return None
    try:
        t0 = datetime.fromisoformat(started_at)
        t1 = datetime.fromisoformat(completed_at)
        return round((t1 - t0).total_seconds(), 3)
    except (ValueError, TypeError):
        return None


def _parse_metric_value(summary: Optional[str]) -> tuple[Optional[str], Optional[float]]:
    """Extract (best_model, best_metric_value) from a PromptHistory.result_summary JSON blob.

    result_summary is written as {"best_model": ..., "metric": "name=value", "error": ...}.
    """
    if not summary:
        return None, None
    try:
        data = json.loads(summary)
    except (json.JSONDecodeError, TypeError):
        return None, None
    best_model = data.get("best_model")
    metric_value: Optional[float] = None
    metric = data.get("metric")
    if isinstance(metric, str) and "=" in metric:
        try:
            metric_value = float(metric.rsplit("=", 1)[1])
        except (ValueError, IndexError):
            metric_value = None
    return best_model, metric_value


@app.get("/api/runs")
async def list_runs(user: User = Depends(get_current_user), db=Depends(get_db)):
    """List the authenticated user's pipeline runs.

    Sourced from the database so history survives server restarts, then overlaid
    with any live in-memory run state (stage progress, duration, agent counts).
    """
    runs: dict[str, dict] = {}

    # 1) Durable history from the database (persists across restarts).
    history_rows = (
        db.query(PromptHistory)
        .filter(PromptHistory.user_id == user.id)
        .order_by(PromptHistory.created_at.desc())
        .all()
    )
    for h in history_rows:
        if not h.run_id:
            continue
        best_model, best_metric_value = _parse_metric_value(h.result_summary)
        runs[h.run_id] = {
            "run_id": h.run_id,
            "status": "completed" if h.status == "completed" else ("failed" if h.status == "failed" else "running"),
            "started_at": h.created_at.isoformat() if h.created_at else None,
            "completed_at": None,
            "duration_seconds": None,
            "current_stage": None,
            "mode": h.mode or "free",
            "dataset": Path(h.data_path).name if h.data_path else None,
            "target": h.target_column,
            "best_model": best_model,
            "best_metric_value": best_metric_value,
            "completed_stages": [],
            "agent_count": 0,
        }

    # 2) Overlay richer live state for this user's in-flight / recent runs.
    for rid, info in _active_runs.items():
        if info.get("user_id") != user.id:
            continue
        result = info.get("result") if isinstance(info.get("result"), dict) else {}
        live = {
            "run_id": rid,
            "status": info.get("status", "running"),
            "started_at": info.get("started_at"),
            "completed_at": info.get("completed_at"),
            "duration_seconds": _duration_seconds(info.get("started_at"), info.get("completed_at")),
            "current_stage": info.get("current_stage"),
            "mode": info.get("mode", "free"),
            "dataset": (Path(info["data_path"]).name if info.get("data_path")
                        else runs.get(rid, {}).get("dataset")),
            "target": info.get("target_column") or runs.get(rid, {}).get("target"),
            "best_model": (result or {}).get("best_model") or runs.get(rid, {}).get("best_model"),
            "best_metric_value": (result or {}).get("best_metric_value") or runs.get(rid, {}).get("best_metric_value"),
            "completed_stages": info.get("completed_stages", []),
            "agent_count": len(info.get("agent_outputs", {})),
        }
        runs[rid] = live

    ordered = sorted(runs.values(), key=lambda r: r.get("started_at") or "", reverse=True)
    return {"runs": ordered}


@app.get("/api/visualizations/{run_id}")
async def get_visualizations(
    run_id: str,
    category: Optional[str] = Query(None),
    user: User = Depends(get_current_user),
):
    """Get visualizations for a run."""
    run = _require_run(run_id, user)
    vizs = run.get("visualizations", [])

    if category:
        vizs = [v for v in vizs if v.get("category") == category]

    return {"visualizations": vizs}


@app.post("/api/visualizations/{run_id}/generate")
async def generate_visualization_endpoint(
    run_id: str,
    request: VizGenerateRequest,
    user: User = Depends(get_current_user),
):
    """Generate a specific visualization on demand."""
    run = _require_run(run_id, user)
    data_path = run.get("data_path")
    if not data_path or not Path(data_path).exists():
        raise HTTPException(status_code=400, detail="Data file not found")

    valid_types = {
        "pairplot", "pca", "correlation", "missing_values",
        "distributions", "boxplots", "target_distribution",
    }
    if request.type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Unknown viz type: {request.type}")

    target = run.get("target_column")

    def _render() -> Optional[dict]:
        import pandas as pd
        from visualization.engine import (
            generate_pairplot, generate_pca_plot,
            generate_correlation_heatmap, generate_missing_values_heatmap,
            generate_distributions, generate_boxplots, generate_target_distribution,
        )
        df = _detect_and_read_csv(Path(data_path))
        if len(df) > _VIZ_SAMPLE_ROWS:
            df = df.sample(n=_VIZ_SAMPLE_ROWS, random_state=0)
        viz_map = {
            "pairplot": lambda: generate_pairplot(df, target),
            "pca": lambda: generate_pca_plot(df, target),
            "correlation": lambda: generate_correlation_heatmap(df),
            "missing_values": lambda: generate_missing_values_heatmap(df),
            "distributions": lambda: generate_distributions(df),
            "boxplots": lambda: generate_boxplots(df),
            "target_distribution": lambda: generate_target_distribution(df, target) if target else None,
        }
        return viz_map[request.type]()

    try:
        # Heavy render off the event loop so other requests stay responsive.
        result = await run_in_threadpool(_render)
    except Exception as e:
        logger.exception("On-demand visualization failed")
        raise HTTPException(status_code=500, detail=str(e))

    if not result:
        raise HTTPException(status_code=400, detail="Could not generate visualization")

    # Cache the visualization
    run.setdefault("visualizations", []).append(result)
    _persist_run(run_id)
    return _sanitize_for_json(result)


# ── Background pipeline execution ──────────────────────────────────────────


def _run_pipeline_background(
    run_id: str, data_path: str, target_column: Optional[str], mode: str = "free"
) -> None:
    """Dispatch a full pipeline run.

    Prefers an isolated child process (so training never starves the API); falls
    back to the in-thread runner if process isolation is disabled or the worker
    can't be spawned. Either way this is invoked from a daemon thread, so it may
    block.
    """
    if _PIPELINE_ISOLATION:
        proc = None
        try:
            event_q = _MP_CTX.Queue()
            proc = _MP_CTX.Process(
                target=run_pipeline_child,
                args=(event_q, run_id, data_path, target_column, mode),
                daemon=True,
            )
            proc.start()  # spawn failures surface here -> fall back, nothing ran yet
        except Exception:
            logger.exception("Failed to spawn pipeline worker; falling back to in-thread")
            proc = None
        if proc is not None:
            _consume_pipeline_events(run_id, event_q, proc)
            _run_artifact_cleanup()
            return

    _run_pipeline_inthread(run_id, data_path, target_column, mode)
    _run_artifact_cleanup()


def _consume_pipeline_events(run_id: str, event_q, proc) -> None:
    """Drain progress events from the worker process into the run's state.

    Blocking on ``event_q.get`` releases the GIL, so the API event loop stays
    responsive while the child does the CPU-heavy work. The parent owns all
    run-dict / disk / DB persistence (unchanged from the in-thread path).
    """
    run = _active_runs[run_id]
    run["status"] = "running"
    run["logs"].append({"time": get_timestamp(), "msg": "Pipeline starting (isolated process)..."})
    _persist_run(run_id)

    done_payload: Optional[dict] = None
    error_info: Optional[tuple] = None
    while True:
        try:
            evt = event_q.get(timeout=5)
        except Empty:
            if not proc.is_alive():
                break  # child died without a terminal event
            continue

        kind = evt[0]
        if kind == "start":
            _, agent, ts = evt
            run["current_stage"] = agent
            run["logs"].append({"time": ts, "msg": f"Starting {agent}..."})
            _persist_run(run_id)
        elif kind == "complete":
            _, agent, out, ts = evt
            run["agent_outputs"][agent] = out
            if out.get("status") == "completed":
                if agent not in run.setdefault("completed_stages", []):
                    run["completed_stages"].append(agent)
                dur = out.get("duration_seconds") or 0.0
                run["logs"].append({"time": ts, "msg": f"✓ {agent} completed ({dur:.1f}s)"})
            else:
                run["logs"].append({"time": ts, "msg": f"✗ {agent} failed: {out.get('error') or 'unknown'}"[:200]})
            _persist_run(run_id)
        elif kind == "done":
            done_payload = evt[1]
        elif kind == "error":
            error_info = (evt[1], evt[2] if len(evt) > 2 else "")
        elif kind == "__end__":
            break

    proc.join(timeout=15)
    if proc.is_alive():  # safety: don't leak a hung worker
        proc.terminate()

    _finalize_isolated_run(run_id, done_payload, error_info)


def _finalize_isolated_run(run_id: str, done_payload: Optional[dict],
                           error_info: Optional[tuple]) -> None:
    """Apply the worker's terminal result to run state + DB (mirrors in-thread)."""
    run = _active_runs[run_id]
    run["completed_at"] = get_timestamp()

    if done_payload is not None:
        failed = done_payload.get("failed", False)
        run["result"] = done_payload.get("result")
        for v in done_payload.get("visualizations", []) or []:
            run.setdefault("visualizations", []).append(v)
        if failed:
            run["status"] = "failed"
            run["error"] = done_payload.get("failure_reason") or "Pipeline failed during execution"
            run["logs"].append({"time": get_timestamp(), "msg": f"Pipeline failed: {run['error']}"})
        else:
            run["status"] = "completed"
            run["logs"].append({"time": get_timestamp(), "msg": "Pipeline completed successfully!"})
        db_summary = {
            "best_model": done_payload.get("best_model_name"),
            "metric": (f"{done_payload.get('best_metric_name')}={done_payload.get('best_metric_value')}"
                       if done_payload.get("best_metric_name") else None),
            "error": done_payload.get("failure_reason") if failed else None,
        }
        db_status = "failed" if failed else "completed"
    else:
        # Child raised, or died without a terminal event.
        msg = (error_info[0] if error_info else "Pipeline worker exited unexpectedly")
        if error_info and error_info[1]:
            logger.error("Isolated pipeline %s failed:\n%s", run_id, error_info[1])
        run["status"] = "failed"
        run["error"] = msg
        run["logs"].append({"time": get_timestamp(), "msg": f"Pipeline failed: {str(msg)[:300]}"})
        db_summary = {"error": str(msg)}
        db_status = "failed"

    with SessionLocal() as db:
        history = db.query(PromptHistory).filter(PromptHistory.run_id == run_id).first()
        if history:
            history.status = db_status
            history.result_summary = json.dumps(db_summary)
            db.commit()
    _persist_run(run_id)


def _run_pipeline_inthread(
    run_id: str, data_path: str, target_column: Optional[str], mode: str = "free"
) -> None:
    """Execute the full pipeline in this (background) thread — fallback path."""
    run = _active_runs[run_id]
    run["status"] = "running"
    run["logs"].append({"time": get_timestamp(), "msg": "Pipeline starting..."})
    _persist_run(run_id)

    try:
        settings = load_settings()
        setup_logging(log_dir=settings.pipeline.log_dir, level=settings.logging.level)

        from core.logging_config import set_correlation_id
        set_correlation_id(run_id)

        from core.agent_runner import run_full_pipeline

        state = PipelineState(
            run_id=run_id,
            raw_data_path=data_path,
            target_column=target_column,
            max_retries=settings.pipeline.max_retries,
            started_at=get_timestamp(),
        )

        def on_start(agent_name):
            run["current_stage"] = agent_name
            run["logs"].append({"time": get_timestamp(), "msg": f"Starting {agent_name}..."})
            _persist_run(run_id)

        def on_complete(agent_name, output):
            run["agent_outputs"][agent_name] = output.to_dict()
            if output.status == "completed":
                run["completed_stages"].append(agent_name)
                run["logs"].append({"time": get_timestamp(), "msg": f"✓ {agent_name} completed ({output.duration_seconds:.1f}s)"})
            else:
                run["logs"].append({"time": get_timestamp(), "msg": f"✗ {agent_name} failed: {output.error or 'unknown'}"[:200]})
            _persist_run(run_id)

        outputs = run_full_pipeline(state, settings, on_start, on_complete)

        state.completed_at = get_timestamp()

        # Check if the pipeline actually failed internally
        if state.failed:
            run["status"] = "failed"
            run["error"] = state.failure_reason or "Pipeline failed during execution"
            run["result"] = _build_result_payload(state)
            run["completed_at"] = get_timestamp()
            run["logs"].append({"time": get_timestamp(), "msg": f"Pipeline failed: {state.failure_reason or 'unknown error'}"})
        else:
            run["status"] = "completed"
            run["result"] = _build_result_payload(state)
            run["completed_at"] = get_timestamp()

            # Generate visualizations
            try:
                import pandas as pd
                from visualization.engine import generate_free_mode_viz, generate_enterprise_mode_viz, generate_model_comparison

                # Model comparison visualization
                model_viz = generate_model_comparison([
                    {"name": m.model_name, "status": m.status, "metrics": m.metrics, "is_best": m.is_best}
                    for m in state.model_results
                ])
                if model_viz:
                    run.setdefault("visualizations", []).append(model_viz)

                # Feature importance
                if state.feature_importances:
                    from visualization.engine import generate_feature_importance
                    fi_viz = generate_feature_importance(state.feature_importances)
                    if fi_viz:
                        run.setdefault("visualizations", []).append(fi_viz)

            except Exception as viz_err:
                logger.warning(f"Post-pipeline visualization failed: {viz_err}")

            run["logs"].append({"time": get_timestamp(), "msg": "Pipeline completed successfully!"})
        
        # Update database history
        with SessionLocal() as db:
            history = db.query(PromptHistory).filter(PromptHistory.run_id == run_id).first()
            if history:
                history.status = "failed" if state.failed else "completed"
                history.result_summary = json.dumps({
                    "best_model": state.best_model_name,
                    "metric": f"{state.best_metric_name}={state.best_metric_value}" if state.best_metric_name else None,
                    "error": state.failure_reason if state.failed else None,
                })
                db.commit()

        _persist_run(run_id)

    except Exception as e:
        run["status"] = "failed"
        run["error"] = str(e)
        run["completed_at"] = get_timestamp()
        run["logs"].append({"time": get_timestamp(), "msg": f"Pipeline failed: {str(e)[:300]}"})
        logger.exception(f"Pipeline run {run_id} failed")

        # Update database history
        with SessionLocal() as db:
            history = db.query(PromptHistory).filter(PromptHistory.run_id == run_id).first()
            if history:
                history.status = "failed"
                history.result_summary = json.dumps({"error": str(e)})
                db.commit()
        _persist_run(run_id)


def _run_workflow_background(
    run_id: str, data_path: str, target_column: Optional[str], agent_list: list[str],
    problem_type: Optional[str] = None,
) -> None:
    """Execute a custom agent workflow in a background thread."""
    run = _active_runs[run_id]
    run["status"] = "running"
    run["logs"].append({"time": get_timestamp(), "msg": f"Workflow starting with agents: {', '.join(agent_list)}"})
    _persist_run(run_id)

    try:
        settings = load_settings()
        setup_logging(log_dir=settings.pipeline.log_dir, level=settings.logging.level)

        from core.logging_config import set_correlation_id
        set_correlation_id(run_id)

        from core.agent_runner import run_workflow

        state = PipelineState(
            run_id=run_id,
            raw_data_path=data_path,
            target_column=target_column,
            problem_type=problem_type,
            max_retries=settings.pipeline.max_retries,
            started_at=get_timestamp(),
        )

        def on_start(agent_name):
            run["current_stage"] = agent_name
            run["logs"].append({"time": get_timestamp(), "msg": f"Starting {agent_name}..."})
            _persist_run(run_id)

        def on_complete(agent_name, output):
            run["agent_outputs"][agent_name] = output.to_dict()
            if output.status == "completed":
                run["completed_stages"].append(agent_name)
                run["logs"].append({"time": get_timestamp(), "msg": f"✓ {agent_name} completed ({output.duration_seconds:.1f}s)"})
            else:
                run["logs"].append({"time": get_timestamp(), "msg": f"✗ {agent_name} failed: {output.error or 'unknown'}"[:200]})
            _persist_run(run_id)

        outputs = run_workflow(agent_list, state, settings, on_start, on_complete)

        state.completed_at = get_timestamp()

        if state.failed:
            run["status"] = "failed"
            run["error"] = state.failure_reason or "Workflow failed during execution"
            run["result"] = _build_result_payload(state)
            run["completed_at"] = get_timestamp()
            run["logs"].append({"time": get_timestamp(), "msg": f"Workflow failed: {state.failure_reason or 'unknown error'}"})
        else:
            run["status"] = "completed"
            run["result"] = _build_result_payload(state)
            run["completed_at"] = get_timestamp()
            run["logs"].append({"time": get_timestamp(), "msg": "Workflow completed!"})
        
        # Update database history
        with SessionLocal() as db:
            history = db.query(PromptHistory).filter(PromptHistory.run_id == run_id).first()
            if history:
                history.status = "failed" if state.failed else "completed"
                history.result_summary = json.dumps({
                    "agents_run": agent_list,
                    "best_model": state.best_model_name,
                    "error": state.failure_reason if state.failed else None,
                })
                db.commit()

        _persist_run(run_id)

    except Exception as e:
        run["status"] = "failed"
        run["error"] = str(e)
        run["completed_at"] = get_timestamp()
        run["logs"].append({"time": get_timestamp(), "msg": f"Workflow failed: {str(e)[:300]}"})
        logger.exception(f"Workflow run {run_id} failed")

        # Update database history
        with SessionLocal() as db:
            history = db.query(PromptHistory).filter(PromptHistory.run_id == run_id).first()
            if history:
                history.status = "failed"
                history.result_summary = json.dumps({"error": str(e)})
                db.commit()
        _persist_run(run_id)

    _run_artifact_cleanup()


def _run_single_agent_background(run_id: str, agent_name: str) -> None:
    """Execute a single agent in a background thread, sharing state across calls."""
    run = _active_runs[run_id]
    data_path = run.get("data_path")
    target_column = run.get("target_column")

    # Mark as running immediately so the frontend can detect progress
    run["agent_outputs"][agent_name] = {
        "agent_id": agent_name,
        "status": "running",
        "started_at": get_timestamp(),
        "summary": {}, "metrics": {}, "logs": [],
        "artifacts": {}, "visualizations": [], "error": None,
    }
    run["current_stage"] = agent_name
    run["logs"].append({"time": get_timestamp(), "msg": f"Starting agent: {agent_name}"})
    _persist_run(run_id)

    try:
        settings = load_settings()
        from core.agent_runner import run_single_agent

        # Reuse existing PipelineState so each agent sees the previous agent's work.
        # A fresh state is only created the first time (no prior state).
        existing_state: Optional[PipelineState] = run.get("pipeline_state")
        if existing_state is not None:
            state = existing_state
        else:
            state = PipelineState(
                run_id=run_id,
                raw_data_path=data_path,
                target_column=target_column,
                max_retries=settings.pipeline.max_retries,
                started_at=get_timestamp(),
            )

        output = run_single_agent(agent_name, state, settings)

        # Persist updated state for the next agent call
        run["pipeline_state"] = state
        run["agent_outputs"][agent_name] = output.to_dict()

        # Accumulate agent's visualizations into the run's global visualizations
        if output.visualizations:
            run_vizs = run.setdefault("visualizations", [])
            for v in output.visualizations:
                if not any(rv.get("type") == v.get("type") for rv in run_vizs):
                    run_vizs.append(v)

        if output.status == "completed":
            if agent_name not in run.get("completed_stages", []):
                run.setdefault("completed_stages", []).append(agent_name)
            run["logs"].append({"time": get_timestamp(), "msg": f"✓ {agent_name} completed ({output.duration_seconds:.1f}s)"})
        else:
            run["logs"].append({"time": get_timestamp(), "msg": f"✗ {agent_name} failed: {output.error or 'unknown'}"[:200]})
        _persist_run(run_id)

    except Exception as e:
        run["agent_outputs"][agent_name] = {
            "agent_id": agent_name,
            "status": "failed",
            "error": str(e),
            "logs": [{"time": get_timestamp(), "msg": str(e)}],
            "summary": {},
            "metrics": {},
            "artifacts": {},
            "visualizations": [],
        }
        run["logs"].append({"time": get_timestamp(), "msg": f"✗ {agent_name} failed: {str(e)[:200]}"})
        logger.exception(f"Single agent {agent_name} failed in run {run_id}")
        _persist_run(run_id)


# The result-payload builder is shared with the out-of-process worker
# (core.pipeline_worker) so both execution paths produce identical payloads.
_build_result_payload = build_result_payload


@app.get("/api/export/{run_id}/excel")
async def export_run_excel(run_id: str, user: User = Depends(get_current_user)):
    """Export all agent outputs for a run as a multi-sheet Excel file."""
    run = _require_run(run_id, user)
    agent_outputs: dict = run.get("agent_outputs", {})

    if not agent_outputs:
        raise HTTPException(status_code=400, detail="No agent outputs available to export yet")

    def _build_excel() -> bytes:
        import io
        import pandas as pd

        buf = io.BytesIO()

        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            # ── Sheet 1: Run Info ──────────────────────────────────────────
            run_meta = [
                ("Run ID", run_id),
                ("Status", run.get("status", "unknown")),
                ("Mode", run.get("mode", "enterprise")),
                ("Data Path", run.get("data_path", "")),
                ("Target Column", run.get("target_column") or "None"),
                ("Started At", run.get("started_at", "")),
                ("Completed Stages", ", ".join(run.get("completed_stages", []))),
            ]
            pd.DataFrame(run_meta, columns=["Property", "Value"]).to_excel(
                writer, sheet_name="Run Info", index=False
            )

            # ── Sheet 2: Agent Summary (all agents, one row each) ──────────
            summary_rows = []
            for agent_name, output in agent_outputs.items():
                if output.get("status") == "running":
                    continue
                row: dict = {
                    "Agent": agent_name,
                    "Status": output.get("status", "—"),
                    "Duration (s)": round(output.get("duration_seconds") or 0, 3),
                    "Memory (MB)": round(output.get("memory_mb") or 0, 2),
                    "Error": output.get("error") or "",
                }
                for k, v in (output.get("metrics") or {}).items():
                    row[k] = round(v, 6) if isinstance(v, float) else v
                summary_rows.append(row)

            if summary_rows:
                pd.DataFrame(summary_rows).to_excel(
                    writer, sheet_name="Agent Summary", index=False
                )

            # ── Per-agent detail sheets ────────────────────────────────────
            for agent_name, output in agent_outputs.items():
                if output.get("status") not in ("completed", "failed"):
                    continue
                detail_rows = []
                for category, data in [
                    ("Summary", output.get("summary") or {}),
                    ("Metrics", output.get("metrics") or {}),
                ]:
                    for k, v in data.items():
                        if isinstance(v, (dict, list)):
                            v = json.dumps(v)
                        detail_rows.append({"Category": category, "Key": k, "Value": v})

                if detail_rows:
                    sheet_name = agent_name.replace("_", " ").title()[:31]
                    pd.DataFrame(detail_rows).to_excel(
                        writer, sheet_name=sheet_name, index=False
                    )

            # ── Sheet: Logs ────────────────────────────────────────────────
            logs = run.get("logs", [])
            if logs:
                pd.DataFrame(logs).to_excel(writer, sheet_name="Logs", index=False)

        buf.seek(0)
        return buf.read()

    try:
        from fastapi.responses import StreamingResponse
        # Workbook assembly is CPU-heavy — keep it off the event loop.
        xlsx_bytes = await run_in_threadpool(_build_excel)
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="openpyxl not installed. Run: pip install openpyxl",
        )
    except Exception as e:
        logger.exception("Excel export failed")
        raise HTTPException(status_code=500, detail=str(e))

    filename = f"axiom-{run_id[:8]}.xlsx"
    return StreamingResponse(
        iter([xlsx_bytes]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
