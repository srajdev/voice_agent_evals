"""
GET /reports/:id — retrieve a previously generated evaluation report.
GET /reports     — list all reports (paginated).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

# Import the in-memory store shared with the evaluate route
from api.routes.evaluate import _reports

router = APIRouter()


@router.get("/reports")
async def list_reports(limit: int = 20, offset: int = 0):
    """List all stored evaluation reports (most recent first)."""
    all_ids = list(reversed(list(_reports.keys())))
    page_ids = all_ids[offset : offset + limit]
    return {
        "total": len(_reports),
        "limit": limit,
        "offset": offset,
        "reports": [
            {
                "report_id": rid,
                "trace_id": _reports[rid].get("trace_id"),
                "evaluated_at": _reports[rid].get("evaluated_at"),
                "overall_score": _reports[rid].get("summary", {}).get("overall_score"),
                "overall_label": _reports[rid].get("summary", {}).get("overall_label"),
            }
            for rid in page_ids
        ],
    }


@router.get("/reports/{report_id}")
async def get_report(report_id: str):
    """Retrieve a full evaluation report by ID."""
    report = _reports.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
    return JSONResponse(content=report)
