from fastapi import APIRouter

import src.db.unin_data_repo as db


router = APIRouter()


@router.get("")
def list_data():
    data = db.list_data()
    return data
