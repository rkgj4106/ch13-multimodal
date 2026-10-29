import json
import logging
import os
import re
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from engine import ROOT, plan, make_video

DATA=Path(os.getenv("DATA_DIR",str(ROOT/"data"))).resolve()
DATABASE=DATA/"jobs.sqlite3"
ACTIVE=("queued","planning","loading_model","generating","narrating","encoding")
log=logging.getLogger("uvicorn.error")


class VideoRequest(BaseModel):
    script:str=Field(min_length=20,max_length=10000)
    scene_count:int=Field(default=2,ge=1,le=6)
    frames:Literal[49,81]=49
    steps:int=Field(default=30,ge=20,le=50)
    seed:int=Field(default=42,ge=0,le=2147483641)
    voice:Literal["none","gtts"]="none"


def connect():
    db=sqlite3.connect(DATABASE,timeout=30);db.row_factory=sqlite3.Row;return db


def lookup(job_id):
    if not re.fullmatch(r"[a-f0-9]{32}",job_id):raise HTTPException(404,"Job not found")
    with connect() as db:row=db.execute("SELECT * FROM jobs WHERE id=?",(job_id,)).fetchone()
    if row is None:raise HTTPException(404,"Job not found")
    return dict(row)


def update(job_id,status,message,result=None):
    with connect() as db:
        db.execute("UPDATE jobs SET status=?,message=?,result=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                   (status,message,json.dumps(result) if result else None,job_id))
    log.info("Job %s [%s] %s",job_id,status,message)


def worker(job_id):
    try:
        body=VideoRequest.model_validate_json(lookup(job_id)["request"])
        update(job_id,"planning","Gemma is composing English motion prompts")
        board=plan(body.script,body.scene_count)
        result=make_video(board,DATA/job_id,body.voice,body.frames,body.steps,body.seed,
                          lambda status,msg:update(job_id,status,msg))
        update(job_id,"done","Video ready",result)
    except Exception as exc:
        log.exception("Motion video job failed: %s",job_id)
        update(job_id,"failed",str(exc)[:3000])


@asynccontextmanager
async def lifespan(app):
    DATA.mkdir(parents=True,exist_ok=True)
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, request TEXT NOT NULL, status TEXT NOT NULL,
          message TEXT NOT NULL, result TEXT,
          created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        db.execute("UPDATE jobs SET status='failed',message='Interrupted by a server restart. Submit again.' WHERE status NOT IN ('done','failed','queued')")
        queued=[row[0] for row in db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at")]
    app.state.executor=ThreadPoolExecutor(max_workers=1)
    for job_id in queued:app.state.executor.submit(worker,job_id)
    try:yield
    finally:app.state.executor.shutdown(wait=True)


app=FastAPI(title="Gemma + Wan Motion Video",lifespan=lifespan)


@app.get("/",response_class=HTMLResponse)
def home():return (ROOT/"index.html").read_text(encoding="utf-8")


@app.get("/healthz")
def health():return {"status":"ok"}


@app.get("/api/example")
def example():return {"script":(ROOT/"sample_script.txt").read_text(encoding="utf-8")}


@app.post("/api/jobs",status_code=202)
def create(body:VideoRequest):
    if len(body.script.strip())<20:raise HTTPException(400,"Enter at least 20 characters of script")
    job_id=uuid.uuid4().hex
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        count=db.execute("SELECT COUNT(*) FROM jobs WHERE status NOT IN ('done','failed')").fetchone()[0]
        if count>=3:raise HTTPException(429,"The queue is full; wait for an existing job to finish")
        db.execute("INSERT INTO jobs(id,request,status,message) VALUES(?,?,'queued','Waiting for the video GPU')",
                   (job_id,body.model_dump_json()))
    app.state.executor.submit(worker,job_id)
    return {"id":job_id,"status":"queued"}


@app.get("/api/jobs/{job_id}")
def status(job_id:str):
    row=lookup(job_id)
    result={k:row[k] for k in ["id","status","message","created_at","updated_at"]}
    if row["result"]:result["result"]=json.loads(row["result"])
    if row["status"]=="done":result["files"]={name:f"/api/jobs/{job_id}/files/{name}" for name in ["video.mp4","storyboard.json","subtitles.srt"]}
    return result


@app.get("/api/jobs/{job_id}/files/{filename}")
def download(job_id:str,filename:str):
    row=lookup(job_id)
    allowed={"video.mp4":"video/mp4","storyboard.json":"application/json","subtitles.srt":"application/x-subrip"}
    if filename not in allowed or row["status"]!="done":raise HTTPException(404,"File not ready")
    path=DATA/job_id/filename
    if not path.is_file():raise HTTPException(404,"File not found")
    return FileResponse(path,media_type=allowed[filename],filename=filename)
