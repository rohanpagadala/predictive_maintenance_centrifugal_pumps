from __future__ import annotations

import hmac
import logging

from fastapi import Header, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from app import config

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    if not config.API_KEY_REQUIRED:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, config.API_KEY):
        logger.warning("Rejected request with missing/invalid API key.")
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
