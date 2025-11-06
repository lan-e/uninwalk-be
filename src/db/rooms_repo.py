from pathlib import Path

from src.db.utility import load_json_file


BASE_DIR = Path(__file__).resolve().parent.parent
ROOMS_FILE_PATH = BASE_DIR / "data" / "rooms.json"


def get_room_by_id(id: str) -> dict:
    rooms = load_json_file(ROOMS_FILE_PATH)
    for room in rooms:
        if room.get("id") == id:
            return room
    return None


def list_rooms() -> list[dict]:
    rooms = load_json_file(ROOMS_FILE_PATH)
    return rooms if rooms else []
