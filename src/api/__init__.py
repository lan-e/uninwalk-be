from fastapi import APIRouter

from .professors import router as professors


router = APIRouter()

router.include_router(professors, prefix="/professors", tags=["Professors"])

__all__ = ["router"]
