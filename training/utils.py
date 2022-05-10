from datetime import datetime


def get_ts_now_as_str() -> str:
    """Get an ISO 8601-compliant timestamp for use in naming."""
    return datetime.now().isoformat().replace(":", "-")
