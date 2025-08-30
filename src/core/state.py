"""
State management for the multi-agent research workflow
Defines the shared state structure that flows between agents
"""

from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from dataclasses import dataclass
from datetime import datetime

# LangGraph state management
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

@dataclass
class ResearchQuery:
    """
    Structured research query with parameters
    
    Concept: Query Object Pattern
    - Encapsulates all search parameters
    - Makes it easy to pass complex queries between agents
    - Enables query validation and preprocessing
    """
    query: str
    max_results: int = 20
    date_filter: Optional[str] = None
    focus_areas: Optional[List[str]] = None
    analysis_depth: str = "medium"
    research_type: str = "general"  # general, literature_review, comparative_analysis
    

class WorkflowState(TypedDict):
    """
    LangGraph state definition using TypedDict
    
    Concept: Shared State Pattern
    - All agents read from and write to this shared state
    - TypedDict provides type safety for state fields
    - LangGraph manages state persistence and updates
    """
    
    # Input and configuration
    research_query: ResearchQuery
    session_id: str
    user_id: Optional[str]
    
    # Agent outputs
    retrieved_papers: List[Dict[str, Any]]
    analysis_results: List[Dict[str, Any]]
    
    # Workflow metadata
    current_step: str
    workflow_status: str  # "running", "completed", "failed", "human_review"
    error_messages: List[str]
    
    # Human-in-the-loop
    human_feedback: Optional[Dict[str, Any]]
    requires_human_review: bool
    
    # Messages for conversational interface
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Timestamps and tracking
    created_at: str
    updated_at: str
    
    # Final outputs
    research_report: Optional[Dict[str, Any]]
    
def create_initial_state(
    query: str,
    session_id: str,
    user_id: Optional[str] = None,
    **kwargs
) -> WorkflowState:
    """
    Create initial workflow state
    
    Concept: State Initialization
    - Sets up clean starting state for workflow
    - Provides sensible defaults
    - Enables customization via kwargs
    """
    
    research_query = ResearchQuery(
        query=query,
        max_results=kwargs.get('max_results', 20),
        date_filter=kwargs.get('date_filter'),
        focus_areas=kwargs.get('focus_areas'),
        analysis_depth=kwargs.get('analysis_depth', 'medium'),
        research_type=kwargs.get('research_type', 'general')
    )
    
    current_time = datetime.now().isoformat()
    
    return WorkflowState(
        research_query=research_query,
        session_id=session_id,
        user_id=user_id,
        retrieved_papers=[],
        analysis_results=[],
        current_step="initialized",
        workflow_status="running",
        error_messages=[],
        human_feedback=None,
        requires_human_review=False,
        messages=[],
        created_at=current_time,
        updated_at=current_time,
        research_report=None
    )