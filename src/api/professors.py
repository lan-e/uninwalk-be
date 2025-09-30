from fastapi import APIRouter, HTTPException

import src.db.professors_repo as db


router = APIRouter()


@router.get("")
def list_professors():
    professors = db.list_professors()
    return professors


@router.get("/{id}")
def get_professor(id: str):
    professor = db.get_professor_by_id(id)

    if not professor:
        raise HTTPException(404, "Professor not found")

    return professor
