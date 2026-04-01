"""
Pydantic models shared across the Orchestrator service.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    SKIPPED   = "skipped"   # dependency failed


class AgentName(str, Enum):
    CONTEXT = "context"
    SQL     = "sql"
    VIZ     = "viz"
    ML      = "ml"
    NLP     = "nlp"
    REPORT  = "report"


# ── Task graph ─────────────────────────────────────────────────────────────────

class TaskNode(BaseModel):
    """A single node in the directed task graph."""
    task_id:    str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent:      AgentName
    description: str
    payload:    Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)  # task_ids


class TaskGraph(BaseModel):
    """LLM-produced execution plan."""
    intent:     str
    tasks:      List[TaskNode]
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Runtime task state ─────────────────────────────────────────────────────────

class TaskResult(BaseModel):
    task_id:    str
    agent:      AgentName
    status:     TaskStatus = TaskStatus.PENDING
    result:     Optional[Any] = None
    error:      Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at:   Optional[datetime] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds() * 1000
        return None


# ── Multimodal file attachment ─────────────────────────────────────────────────

class FileAttachment(BaseModel):
    """A file attached to an analysis request (uploaded to blob storage)."""
    url:       str  = Field(..., description="Blob storage URL or base64 data URI")
    filename:  str  = Field(..., description="Original filename (e.g. report.pdf, photo.png)")
    mime_type: str  = Field("application/octet-stream", description="MIME type of the file")


# ── API request / response ─────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    query:      str = Field(..., min_length=3, description="Natural-language user query")
    context_id: str = Field(..., description="UUID from Context Agent's /context endpoint")
    session_id: Optional[str] = Field(None, description="Optional prior session to resume")
    files:      Optional[List[FileAttachment]] = Field(
        None,
        description="Optional file attachments (images, PDFs, CSVs) uploaded to blob storage. "
                    "These are forwarded to every agent so they can be processed as needed.",
    )


class PlanRequest(BaseModel):
    query:      str
    context_id: str


class AnalyzeResponse(BaseModel):
    task_id:    str
    session_id: str
    intent:     str
    graph:      TaskGraph
    results:    List[TaskResult]
    partial:    bool = False          # True if any task timed-out/failed
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentHealthInfo(BaseModel):
    agent:     str
    url:       str
    healthy:   bool
    latency_ms: Optional[float] = None
    error:     Optional[str] = None


class AgentsStatusResponse(BaseModel):
    agents:    List[AgentHealthInfo]
    checked_at: datetime = Field(default_factory=datetime.utcnow)


# ── Dataset management ────────────────────────────────────────────────────────

class DatasetMeta(BaseModel):
    context_id: str
    filename: str
    file_type: str
    row_count: int
    columns: List[str]
    preview: List[Dict[str, Any]]
    size_bytes: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class DatasetsResponse(BaseModel):
    datasets: List[DatasetMeta]


# ── SSE event envelope ────────────────────────────────────────────────────────

class SSEEvent(BaseModel):
    event: str                # "task_start" | "task_complete" | "result" | "error"
    data:  Dict[str, Any]
