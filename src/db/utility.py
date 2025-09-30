import json


def load_json_file(file_path: str) -> dict[str, any] | list[dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_file = json.load(f)
        return json_file
    except Exception as e:
        raise Exception(f"Unable to read JSON file, check file path. Details: {e}")
