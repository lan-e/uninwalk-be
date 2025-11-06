from fastapi import APIRouter, HTTPException

import src.db.rooms_repo as db


router = APIRouter()


@router.get("")
def list_rooms():
    rooms = db.list_rooms()
    return rooms


@router.get("/{id}")
def get_rooom(id: str):
    room = db.get_room_by_id(id)

    if not room:
        raise HTTPException(404, "Room not found")

    return room
