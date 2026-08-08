"""Consistent API error envelopes with correlation IDs."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

ERROR_CODES = {
    400: "bad_request",
    404: "not_found",
    409: "conflict",
    422: "validation_error",
    429: "rate_limited",
    503: "service_unavailable",
}


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)


async def http_exception_handler(request: Request, error: HTTPException) -> JSONResponse:
    message = error.detail if isinstance(error.detail, str) else "Request failed"
    headers = dict(error.headers or {})
    if request_id := _request_id(request):
        headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=error.status_code,
        headers=headers,
        content=_error_payload(
            request,
            code=ERROR_CODES.get(error.status_code, "request_error"),
            message=message,
            detail=error.detail,
        ),
    )


async def validation_exception_handler(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        headers=_request_id_headers(request),
        content=_error_payload(
            request,
            code="validation_error",
            message="Request validation failed",
            detail="Request validation failed",
            violations=jsonable_encoder(error.errors()),
        ),
    )


async def unexpected_exception_handler(request: Request, error: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        headers=_request_id_headers(request),
        content=_error_payload(
            request,
            code="internal_error",
            message="An internal server error occurred",
            detail="An internal server error occurred",
        ),
    )


def _error_payload(
    request: Request,
    *,
    code: str,
    message: str,
    detail: Any,
    violations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "detail": detail,
        "error": {"code": code, "message": message},
        "request_id": _request_id(request),
    }
    if violations is not None:
        payload["violations"] = violations
    return payload


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _request_id_headers(request: Request) -> dict[str, str]:
    request_id = _request_id(request)
    return {"X-Request-ID": request_id} if request_id else {}
