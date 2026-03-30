"""
FastAPI router for dataset management (upload, list, delete).
Provides a central gateway for files that are kemudian profiled by the Context-Intelligence-Agent.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.models import DatasetMeta, DatasetsResponse
from app.utils.storage import storage

logger = logging.getLogger(__name__)
router = APIRouter(tags=["datasets"])

# ── Paths ─────────────────────────────────────────────────────────────────────

UPLOAD_DIR = Path("uploads")
METADATA_FILE = UPLOAD_DIR / "datasets.json"

def ensure_upload_dir():
    if not UPLOAD_DIR.exists():
        UPLOAD_DIR.mkdir(parents=True)
    if not METADATA_FILE.exists():
        with open(METADATA_FILE, "w") as f:
            json.dump([], f)

# ── Metadata helpers ──────────────────────────────────────────────────────────

def load_metadata() -> List[dict]:
    ensure_upload_dir()
    try:
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_metadata(metadata: List[dict]):
    ensure_upload_dir()
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DatasetMeta, summary="Upload a dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> DatasetMeta:
    """
    1. Saves file to local or Azure Blob Storage.
    2. Calls Context-Intelligence-Agent to profile the file.
    3. Returns enriched metadata (context_id, stats, etc.).
    """
    ensure_upload_dir()
    
    file_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix.lower()
    save_name = f"{file_id}_{file.filename}"
    
    # Check supported formats
    fmt = "csv"
    if ext == ".parquet":
        fmt = "parquet"
    elif ext == ".json":
        fmt = "json"
    elif ext not in [".csv"]:
         raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    # 1. Save to Storage
    try:
        content = await file.read()
        file_uri = await storage.save_file(content, save_name)
    except Exception as exc:
        logger.error("Failed to save file: %s", exc)
        raise HTTPException(status_code=500, detail="Could not save file")

    # 2. Call Context Agent to Profile
    context_agent_url = f"{settings.context_agent_url}/profile"
    logger.info(f"Calling Context Agent at: {context_agent_url} for file: {file_uri}")
    
    source_type = "azure_blob" if storage.use_azure else "local_file"
    payload = {
        "source": {
            "type": source_type,
            "path": file_uri,
            "format": fmt
        }
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(context_agent_url, json=payload)
            if resp.status_code != 200:
                logger.error(f"Context Agent failed with {resp.status_code}: {resp.text}")
                await storage.delete_file(save_name)
                raise HTTPException(
                    status_code=502, 
                    detail=f"Context profiling failed: {resp.text[:200]}"
                )

            context_obj = resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error calling Context Agent")
        await storage.delete_file(save_name)
        raise HTTPException(
            status_code=502, 
            detail=f"Context Agent unreachable: {str(exc)}"
        )
    # 3. Create metadata entry
    meta = DatasetMeta(
        context_id=context_obj.get("source_id", file_id),
        filename=file.filename,
        file_type=fmt,
        row_count=context_obj.get("row_count", 0),
        columns=[c["name"] for c in context_obj.get("columns", [])],
        preview=[],  # Populate below
        size_bytes=len(content),
        uploaded_at=datetime.utcnow()
    )
    
    if "columns" in context_obj and context_obj["columns"]:
        cols = context_obj["columns"]
        max_samples = 0
        for col in cols:
            samples = col.get("sample_values", [])
            if isinstance(samples, list) and len(samples) > max_samples:
                max_samples = len(samples)
        
        preview_rows = []
        for i in range(min(5, max_samples)):
            row = {}
            for col in cols:
                name = col.get("name", "unknown")
                samples = col.get("sample_values", [])
                if isinstance(samples, list) and i < len(samples):
                    row[name] = samples[i]
                else:
                    row[name] = None
            preview_rows.append(row)
        meta.preview = preview_rows

    # Store in local registry
    metadata = load_metadata()
    try:
        entry = meta.model_dump()
        if isinstance(entry.get("uploaded_at"), datetime):
            entry["uploaded_at"] = entry["uploaded_at"].isoformat()
        
        entry["local_path"] = file_uri
        entry["storage_name"] = save_name
        metadata.insert(0, entry)
        save_metadata(metadata)
    except Exception as exc:
        logger.error("Failed to serialize or save metadata: %s", exc)
    
    return meta


@router.get("/datasets", response_model=DatasetsResponse, summary="List all datasets")
async def list_datasets() -> DatasetsResponse:
    metadata = load_metadata()
    datasets = []
    for m in metadata:
        try:
            datasets.append(DatasetMeta(**m))
        except Exception:
            continue
    return DatasetsResponse(datasets=datasets)


@router.delete("/datasets/{context_id}", summary="Delete a dataset")
async def delete_dataset(context_id: str):
    metadata = load_metadata()
    item = next((m for m in metadata if m["context_id"] == context_id), None)
    
    if not item:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    storage_name = item.get("storage_name") or item.get("filename")
    try:
        await storage.delete_file(storage_name)
    except Exception as exc:
        logger.warning("Failed to delete file %s: %s", storage_name, exc)

    updated = [m for m in metadata if m["context_id"] != context_id]
    save_metadata(updated)
    
    return {"status": "ok", "deleted": context_id}
