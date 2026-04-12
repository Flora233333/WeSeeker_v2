from __future__ import annotations

from datetime import datetime, timezone


MODIFIED_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_FILETIME_EPOCH_OFFSET_SECONDS = 11644473600
_FILETIME_TICKS_PER_SECOND = 10_000_000


def format_modified_datetime(value: datetime) -> str:
    return value.strftime(MODIFIED_TIME_FORMAT)


def format_modified_timestamp(timestamp: float) -> str:
    return format_modified_datetime(datetime.fromtimestamp(timestamp))


def format_modified_value(value: object) -> str:
    if value in (None, ""):
        return ""

    if isinstance(value, datetime):
        return format_modified_datetime(value)

    if isinstance(value, (int, float)):
        return _format_filetime(int(value))

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""
        if stripped.isdigit():
            return _format_filetime(int(stripped))

        try:
            return format_modified_datetime(
                datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            )
        except ValueError:
            return stripped

    return str(value)


def _format_filetime(filetime: int) -> str:
    if filetime <= 0:
        return ""

    unix_timestamp = (filetime / _FILETIME_TICKS_PER_SECOND) - _FILETIME_EPOCH_OFFSET_SECONDS
    dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc).astimezone()
    return format_modified_datetime(dt)
