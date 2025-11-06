from fastapi import APIRouter

from .professors import router as professors
from .rooms import router as rooms
from .unin_data import router as unin_data


router = APIRouter()

router.include_router(professors, prefix="/professors", tags=["Professors"])
router.include_router(rooms, prefix="/rooms", tags=["Rooms"])
router.include_router(unin_data, prefix="/unin-data", tags=["UNIN data"])

__all__ = ["router"]
