"""Persist shared project snapshots on the local showroom server."""
import json
import os
import secrets
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field, StringConstraints, model_validator

from bench.quotes import Name, QuoteRoom, QuoteSku


router = APIRouter()
Coordinate = Annotated[float, Field(allow_inf_nan=False)]
Point = tuple[Coordinate, Coordinate]
ImageData = Annotated[str, StringConstraints(
    pattern=r"^data:image/(png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}$")]


class Workspace(BaseModel):
    name: Name
    photo: ImageData
    room_id: str | None
    quad: Annotated[list[Point], Field(max_length=4)]
    boundary: list[Point]
    boundary_closed: bool
    auto_plane: bool
    exclusions: list[list[Point]]
    ai_exclusions: list[list[Point]]
    material_id: str | None
    custom_faces: list[ImageData]
    settings: dict[str, str | bool]
    surfaces: list['Surface'] = Field(default_factory=list, max_length=12)
    active_surface_id: str | None = None


class Surface(BaseModel):
    id: Name
    name: Name
    kind: Literal['floor', 'wall']
    quad: Annotated[list[Point], Field(max_length=4)]
    boundary: list[Point]
    boundary_closed: bool
    auto_plane: bool
    exclusions: list[list[Point]]
    ai_exclusions: list[list[Point]]
    material_id: str | None
    custom_faces: list[ImageData]
    settings: dict[str, str | bool]
    selection_points: list[tuple[Coordinate, Coordinate, Literal[0, 1]]]


class SavedRoom(QuoteRoom):
    workspace: Workspace


class ProjectQuote(BaseModel):
    rooms: list[SavedRoom]
    skus: dict[Name, QuoteSku]

    @model_validator(mode="after")
    def check_product_codes(self):
        if any(room.sku_name not in self.skus for room in self.rooms):
            raise ValueError("Every room must reference an existing product code")
        return self


class Project(BaseModel):
    name: Name
    schema_version: Literal[1]
    workspace: Workspace
    quote: ProjectQuote


class ProjectUpdate(Project):
    revision: Annotated[int, Field(ge=1, strict=True)]


def connection():
    root = Path(os.environ.get("SHOWROOM_DATA_DIR", "data"))
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "projects.sqlite3")
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY, snapshot TEXT NOT NULL,
        revision INTEGER NOT NULL, saved_at TEXT NOT NULL
    )""")
    return db


def record(row):
    origin = os.environ.get("SHOWROOM_PUBLIC_ORIGIN", "").rstrip("/")
    return {**json.loads(row["snapshot"]), "id": row["id"],
            "revision": row["revision"], "saved_at": row["saved_at"],
            "share_url": f'{origin}/?project={row["id"]}' if origin else None}


@router.post("/projects", status_code=201)
def create_project(body: Project, response: Response):
    project_id = secrets.token_urlsafe(32)
    snapshot = body.model_dump_json(exclude_unset=True)
    saved_at = datetime.now(UTC).isoformat()
    with closing(connection()) as db, db:
        db.execute("INSERT INTO projects VALUES (?, ?, ?, ?)",
                   (project_id, snapshot, 1, saved_at))
        row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    response.headers["Cache-Control"] = "no-store"
    return record(row)


@router.get("/projects/{project_id}")
def get_project(project_id: str, response: Response):
    with closing(connection()) as db:
        row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Project not found. Check the complete project link.")
    response.headers["Cache-Control"] = "no-store"
    return record(row)


@router.put("/projects/{project_id}")
def update_project(project_id: str, body: ProjectUpdate, response: Response):
    snapshot = body.model_dump_json(exclude={"revision"}, exclude_unset=True)
    saved_at = datetime.now(UTC).isoformat()
    with closing(connection()) as db, db:
        changed = db.execute(
            "UPDATE projects SET snapshot = ?, revision = revision + 1, "
            "saved_at = ? WHERE id = ? AND revision = ?",
            (snapshot, saved_at, project_id, body.revision))
        row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Project not found. Check the complete project link.")
        if changed.rowcount == 0:
            raise HTTPException(409, "This project changed on another device. "
                                "Reload the saved version before saving again.")
    response.headers["Cache-Control"] = "no-store"
    return record(row)
