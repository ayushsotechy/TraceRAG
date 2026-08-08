import shutil
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from tracerag import __version__
from tracerag.api.schemas import (
    AnswerResponse,
    CitationResponse,
    HealthResponse,
    IngestResponse,
    QuestionRequest,
)
from tracerag.config import get_settings
from tracerag.ingestion.loaders import UnsupportedDocumentError
from tracerag.service import RAGService

app = FastAPI(
    title="TraceRAG API",
    description="Evaluation-first retrieval-augmented generation with source citations.",
    version=__version__,
)

WEB_APP = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TraceRAG</title>
<style>:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#f0f6fc;--muted:#8b949e;--green:#3fb950;--blue:#58a6ff;--red:#f85149}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 Inter,system-ui,sans-serif}main{width:min(920px,calc(100% - 32px));margin:auto;padding:64px 0}.hero{margin-bottom:36px}.hero h1{font-size:48px;margin:0 0 4px}.hero p,.muted{color:var(--muted)}.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:24px;margin:18px 0}h2{margin:0 0 16px;font-size:21px}.drop{display:block;border:1px dashed #6e7681;border-radius:10px;padding:28px;text-align:center;cursor:pointer}.drop:hover{border-color:var(--blue)}input[type=file]{display:none}button{border:0;border-radius:8px;padding:11px 18px;background:#238636;color:white;font-weight:700;cursor:pointer}button:disabled{opacity:.45;cursor:not-allowed}.row{display:flex;gap:10px;margin-top:16px}.row input{flex:1;background:#0d1117;border:1px solid var(--line);border-radius:8px;color:var(--text);padding:12px}#files{margin:12px 0;color:var(--blue)}#status{min-height:24px;margin-top:12px}.error{color:var(--red)}.success{color:var(--green)}.answer{white-space:pre-wrap;margin-top:18px}.citation{border-left:3px solid var(--blue);padding-left:12px;margin:12px 0;color:#c9d1d9}@media(max-width:600px){main{padding:28px 0}.hero h1{font-size:38px}.row{flex-direction:column}}</style></head>
<body><main><section class="hero"><h1>TraceRAG</h1><p>Evidence-first answers from your documents</p></section><section class="card"><h2>1. Build a knowledge base</h2><label class="drop" for="documents"><strong>Choose PDF, Markdown, or text files</strong><br><span class="muted">Up to 200 MB per file</span></label><input id="documents" type="file" multiple accept=".pdf,.md,.txt"><div id="files"></div><button id="index" disabled>Upload &amp; build index</button><div id="status"></div></section><section class="card"><h2>2. Ask a question</h2><div class="row"><input id="question" placeholder="Ask about the indexed documents" disabled><button id="ask" disabled>Ask</button></div><div id="answer" class="answer"></div></section></main>
<script>const docs=document.querySelector('#documents'),index=document.querySelector('#index'),files=document.querySelector('#files'),status=document.querySelector('#status'),question=document.querySelector('#question'),ask=document.querySelector('#ask'),answer=document.querySelector('#answer');docs.addEventListener('change',()=>{const names=[...docs.files].map(f=>f.name);files.textContent=names.join(', ');index.disabled=!names.length;status.textContent=''});index.addEventListener('click',async()=>{index.disabled=true;status.className='';status.textContent='Uploading and indexing…';const body=new FormData();[...docs.files].forEach(f=>body.append('files',f));try{const r=await fetch('/v1/documents',{method:'POST',body});const data=await r.json();if(!r.ok)throw new Error(data.detail||'Upload failed');status.className='success';status.textContent=`Indexed ${data.chunks} chunks from ${data.files} file(s).`;question.disabled=false;ask.disabled=false}catch(e){status.className='error';status.textContent=e.message;index.disabled=false}});async function query(){const q=question.value.trim();if(!q)return;ask.disabled=true;answer.textContent='Searching for evidence…';try{const r=await fetch('/v1/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});const data=await r.json();if(!r.ok)throw new Error(data.detail||'Query failed');answer.textContent=data.answer;for(const c of data.citations){const el=document.createElement('div');el.className='citation';el.textContent=`[${c.source_id}] ${c.filename} · page ${c.page}: ${c.excerpt}`;answer.appendChild(el)}}catch(e){answer.textContent=e.message}finally{ask.disabled=false}}ask.addEventListener('click',query);question.addEventListener('keydown',e=>{if(e.key==='Enter')query()});</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def web_app() -> str:
    return WEB_APP


@lru_cache
def get_service() -> RAGService:
    return RAGService(get_settings())


ServiceDependency = Annotated[RAGService, Depends(get_service)]


@app.get("/health", response_model=HealthResponse)
def health(service: ServiceDependency) -> HealthResponse:
    return HealthResponse(
        status="ok",
        documents=service.document_count,
        chunks=service.chunk_count,
        retrieval_mode=service.retrieval_mode,
    )


@app.post("/v1/documents", response_model=IngestResponse)
def ingest_documents(
    files: Annotated[list[UploadFile], File()],
    service: ServiceDependency,
) -> IngestResponse:
    upload_dir = service.settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        for upload in files:
            safe_name = Path(upload.filename or "document").name
            path = upload_dir / safe_name
            with path.open("wb") as destination:
                shutil.copyfileobj(upload.file, destination)
            paths.append(path)
        chunks = service.ingest(paths)
        return IngestResponse(files=len(paths), chunks=chunks)
    except (UnsupportedDocumentError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/query", response_model=AnswerResponse)
def query(
    request: QuestionRequest,
    service: ServiceDependency,
) -> AnswerResponse:
    result = service.ask(request.question)
    return AnswerResponse(
        answer=result.text,
        citations=[CitationResponse(**asdict(citation)) for citation in result.citations],
        confidence=result.confidence,
        abstained=result.abstained,
    )
