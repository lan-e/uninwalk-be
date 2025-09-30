from fastapi import APIRouter, HTTPException

import src.db.professors_repo as db


router = APIRouter()


@router.get("")
def list_professors():
    professors = db.list_professors()
    return professors
