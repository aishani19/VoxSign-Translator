import json
import os
from typing import List


def load_actions(default: str = "a,b") -> List[str]:
    """
    Action loading priority:
    1) labels.json -> {"actions": ["hello", "thanks", ...]}
    2) VOXSIGN_ACTIONS env var (comma separated)
    3) fallback default
    """
    labels_path = os.path.join("labels.json")
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            actions = payload.get("actions", [])
            cleaned = [str(x).strip() for x in actions if str(x).strip()]
            if cleaned:
                return cleaned
        except Exception:
            pass

    actions_env = os.environ.get("VOXSIGN_ACTIONS", default)
    return [item.strip() for item in actions_env.split(",") if item.strip()]
