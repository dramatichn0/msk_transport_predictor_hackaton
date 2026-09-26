"""Safe HTTP error responses, without echoing raw telemetry or invalid JSON values."""
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Exclude raw inputs and exception objects from JSON validation error bodies."""
    detail = [{key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})
