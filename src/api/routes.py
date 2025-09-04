from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from ..core.graph import research_workflow

# Input model for API
class ResearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    query: str = Field(..., min_length=3, max_length=500)
    max_results: Optional[int] = Field(default=5, ge=1, le=50)
    date_filter: Optional[str] = Field(default=None, pattern=r'^\d{4}(-\d{2})?$')
    focus_areas: Optional[list[str]] = None
    analysis_depth: Optional[str] = Field(default="medium", pattern=r'^(shallow|medium|deep)$')

router = APIRouter()

@router.post("/research", summary="Run research workflow")
async def run_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Trigger the multi-agent research workflow.
    Returns a session ID immediately and runs workflow in background.
    """
    try:
        # Schedule workflow in background
        background_tasks.add_task(
            research_workflow.run_research_workflow,
            request.query,
            user_id=None,
            max_results=request.max_results,
            date_filter=request.date_filter,
            focus_areas=request.focus_areas,
            analysis_depth=request.analysis_depth
        )
        return {"status": "scheduled", "message": "Research workflow started", "query": request.query}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/research/{session_id}", summary="Get workflow results")
async def get_research_results(session_id: str):
    """
    Retrieve results of a previously scheduled research workflow.
    """
    # In a production scenario, results would be stored in database or cache.
    # Here, we simulate by returning a placeholder.
    # Implement persistent storage lookup as needed.
    return {"session_id": session_id, "status": "completed", "research_report": "Report data here"}
