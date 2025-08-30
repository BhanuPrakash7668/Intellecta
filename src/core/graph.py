"""
LangGraph workflow definition for multi-agent research pipeline
Orchestrates Literature Retriever, Analysis Agent, and Report Generator
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage

# Local imports  
from .state import WorkflowState, create_initial_state
from ..agents.literature_retriever import literature_retriever
from ..agents.analysis_agent import analysis_agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchWorkflow:
    """
    Multi-agent research workflow using LangGraph
    
    Concept: Workflow Orchestration
    - Defines the flow between different agents
    - Manages state transitions and error handling
    - Provides checkpointing for reliability
    - Enables human-in-the-loop interventions
    """
    
    def __init__(self, checkpointer_path: str = "research_workflow.db"):
        """
        Initialize the research workflow
        
        Concept: Workflow Setup
        - Creates state graph with all nodes and edges
        - Sets up checkpointer for state persistence
        - Configures error handling and recovery
        """
        self.checkpointer = SqliteSaver.from_conn_string(checkpointer_path)
        self.graph = self._create_workflow_graph()
        logger.info("✅ Research workflow initialized")
        
    def _create_workflow_graph(self) -> StateGraph:
        """
        Create the LangGraph workflow structure
        
        Concept: Graph Definition
        - Defines all workflow nodes (agent functions)
        - Sets up conditional routing logic
        - Configures state management and checkpointing
        """
        
        # Create state graph
        workflow = StateGraph(WorkflowState)
        
        # Add workflow nodes
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("retrieve_literature", self._retrieve_literature)
        workflow.add_node("analyze_papers", self._analyze_papers)
        workflow.add_node("generate_report", self._generate_report)
        workflow.add_node("human_review", self._human_review)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define workflow edges and routing
        workflow.set_entry_point("validate_input")
        
        # Conditional routing after input validation
        workflow.add_conditional_edges(
            "validate_input",
            self._should_continue_after_validation,
            {
                "continue": "retrieve_literature",
                "error": "handle_error"
            }
        )
        
        # After literature retrieval
        workflow.add_conditional_edges(
            "retrieve_literature", 
            self._should_continue_after_retrieval,
            {
                "continue": "analyze_papers",
                "no_results": "handle_error",
                "error": "handle_error"
            }
        )
        
        # After analysis
        workflow.add_conditional_edges(
            "analyze_papers",
            self._should_continue_after_analysis,
            {
                "continue": "generate_report",
                "human_review": "human_review",
                "error": "handle_error"
            }
        )
        
        # After report generation
        workflow.add_conditional_edges(
            "generate_report",
            self._should_finalize,
            {
                "complete": END,
                "human_review": "human_review",
                "error": "handle_error"
            }
        )
        
        # Human review can go back to analysis or end
        workflow.add_conditional_edges(
            "human_review",
            self._handle_human_feedback,
            {
                "reanalyze": "analyze_papers", 
                "regenerate": "generate_report",
                "complete": END
            }
        )
        
        # Error handling always ends the workflow
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    
    async def run_research_workflow(
        self, 
        query: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the complete research workflow
        
        Concept: Workflow Execution
        - Creates initial state
        - Runs the workflow graph
        - Returns final results
        - Handles errors gracefully
        """
        
        # Generate session ID for this workflow run
        session_id = str(uuid.uuid4())
        
        logger.info(f"🚀 Starting research workflow for query: '{query}' (session: {session_id})")
        
        try:
            # Create initial state
            initial_state = create_initial_state(
                query=query,
                session_id=session_id, 
                user_id=user_id,
                **kwargs
            )
            
            # Run the workflow
            config = {"configurable": {"thread_id": session_id}}
            final_state = await self.graph.ainvoke(initial_state, config=config)
            
            logger.info(f"✅ Workflow completed for session: {session_id}")
            
            return {
                "session_id": session_id,
                "status": final_state["workflow_status"],
                "research_report": final_state.get("research_report"),
                "retrieved_papers": final_state["retrieved_papers"],
                "analysis_results": final_state["analysis_results"],
                "error_messages": final_state["error_messages"]
            }
            
        except Exception as e:
            logger.error(f"❌ Workflow failed for session {session_id}: {e}")
            return {
                "session_id": session_id,
                "status": "failed",
                "error": str(e),
                "research_report": None
            }
    
    # Node Functions (Agent Orchestration)
    
    async def _validate_input(self, state: WorkflowState) -> WorkflowState:
        """
        Validate input parameters and research query
        
        Concept: Input Validation Node
        - Ensures query is valid and well-formed
        - Sets up initial workflow state
        - Adds validation messages to state
        """
        logger.info("🔍 Validating research input...")
        
        state["current_step"] = "validating_input"
        state["updated_at"] = datetime.now().isoformat()
        
        query = state["research_query"].query
        
        # Basic validation
        if not query or len(query.strip()) < 3:
            state["error_messages"].append("Query must be at least 3 characters long")
            state["workflow_status"] = "failed"
            return state
        
        if len(query) > 500:
            state["error_messages"].append("Query too long (max 500 characters)")
            state["workflow_status"] = "failed" 
            return state
        
        # Add validation message
        state["messages"].append(
            AIMessage(content=f"✅ Research query validated: '{query[:100]}...' if len(query) > 100 else query")
        )
        
        logger.info("✅ Input validation passed")
        return state
    
    async def _retrieve_literature(self, state: WorkflowState) -> WorkflowState:
        """
        Execute literature retrieval using Literature Retriever Agent
        
        Concept: Agent Integration Node
        - Calls the Literature Retriever Agent
        - Updates state with retrieved papers
        - Handles retrieval errors gracefully
        """
        logger.info("📚 Retrieving literature...")
        
        state["current_step"] = "retrieving_literature"
        state["updated_at"] = datetime.now().isoformat()
        
        try:
            query_obj = state["research_query"]
            
            # Call Literature Retriever Agent
            papers = await literature_retriever._async_search(
                query=query_obj.query,
                max_results=query_obj.max_results,
                date_filter=query_obj.date_filter
            )
            
            state["retrieved_papers"] = papers
            
            # Add status message
            state["messages"].append(
                AIMessage(content=f"📄 Retrieved {len(papers)} research papers from ArXiv and Semantic Scholar")
            )
            
            logger.info(f"✅ Retrieved {len(papers)} papers")
            
        except Exception as e:
            error_msg = f"Literature retrieval failed: {str(e)}"
            state["error_messages"].append(error_msg)
            state["workflow_status"] = "failed"
            logger.error(f"❌ {error_msg}")
        
        return state
    
    async def _analyze_papers(self, state: WorkflowState) -> WorkflowState:
        """
        Execute paper analysis using Analysis Agent
        
        Concept: Analysis Integration Node
        - Calls the Analysis Agent with retrieved papers
        - Updates state with analysis results
        - Determines if human review is needed
        """
        logger.info("🔬 Analyzing retrieved papers...")
        
        state["current_step"] = "analyzing_papers"
        state["updated_at"] = datetime.now().isoformat()
        
        try:
            papers = state["retrieved_papers"]
            query_obj = state["research_query"]
            
            if not papers:
                state["error_messages"].append("No papers to analyze")
                state["workflow_status"] = "failed"
                return state
            
            # Call Analysis Agent
# In src/core/graph.py

# ... inside _analyze_papers
            analysis_results = await analysis_agent._async_analyze(
                papers=papers,
                focus_areas=query_obj.focus_areas,
                analysis_depth=query_obj.analysis_depth
            )
            
            state["analysis_results"] = analysis_results
            
            # Determine if human review is needed
            # (Based on analysis complexity, conflicting results, etc.)
            state["requires_human_review"] = self._should_request_human_review(analysis_results)
            
            # Add status message
            themes_count = sum(len(result.get('themes', [])) for result in analysis_results)
            state["messages"].append(
                AIMessage(content=f"🧠 Analysis complete: {len(analysis_results)} papers analyzed, {themes_count} themes identified")
            )
            
            logger.info(f"✅ Analysis complete for {len(analysis_results)} papers")
            
        except Exception as e:
            error_msg = f"Paper analysis failed: {str(e)}"
            state["error_messages"].append(error_msg)
            state["workflow_status"] = "failed"
            logger.error(f"❌ {error_msg}")
        
        return state
    
    async def _generate_report(self, state: WorkflowState) -> WorkflowState:
        """
        Generate final research report
        
        Concept: Report Synthesis Node
        - Combines literature retrieval and analysis results
        - Generates structured research report
        - Prepares final output for user
        """
        logger.info("📝 Generating research report...")
        
        state["current_step"] = "generating_report"
        state["updated_at"] = datetime.now().isoformat()
        
        try:
            papers = state["retrieved_papers"]
            analyses = state["analysis_results"]
            query = state["research_query"].query
            
            # Generate comprehensive report
            report = await self._synthesize_research_report(query, papers, analyses)
            
            state["research_report"] = report
            state["workflow_status"] = "completed"
            
            # Add completion message
            state["messages"].append(
                AIMessage(content="📋 Research report generated successfully!")
            )
            
            logger.info("✅ Research report generated")
            
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            state["error_messages"].append(error_msg)
            state["workflow_status"] = "failed"
            logger.error(f"❌ {error_msg}")
        
        return state
    
    async def _human_review(self, state: WorkflowState) -> WorkflowState:
        """
        Handle human review and feedback
        
        Concept: Human-in-the-Loop Node
        - Pauses workflow for human input
        - Processes human feedback
        - Determines next steps based on feedback
        """
        logger.info("👤 Requesting human review...")
        
        state["current_step"] = "human_review"
        state["workflow_status"] = "human_review"
        state["updated_at"] = datetime.now().isoformat()
        
        # Add human review message
        state["messages"].append(
            AIMessage(content="🤔 Analysis complete. Human review requested for quality assurance.")
        )
        
        # In a real implementation, this would pause and wait for human input
        # For now, we'll simulate automatic approval
        state["human_feedback"] = {
            "action": "approve",
            "comments": "Automated approval for demo purposes",
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("✅ Human review completed")
        return state
    
    async def _handle_error(self, state: WorkflowState) -> WorkflowState:
        """
        Handle workflow errors and cleanup
        
        Concept: Error Handling Node
        - Processes workflow errors
        - Provides meaningful error messages
        - Ensures clean workflow termination
        """
        logger.info("❌ Handling workflow error...")
        
        state["current_step"] = "error_handling"
        state["workflow_status"] = "failed"
        state["updated_at"] = datetime.now().isoformat()
        
        # Consolidate error messages
        if state["error_messages"]:
            error_summary = "; ".join(state["error_messages"])
            state["messages"].append(
                AIMessage(content=f"❌ Workflow failed: {error_summary}")
            )
        else:
            state["messages"].append(
                AIMessage(content="❌ Workflow failed due to unknown error")
            )
        
        logger.error("❌ Workflow terminated due to errors")
        return state
    
    # Conditional Edge Functions
    
    def _should_continue_after_validation(self, state: WorkflowState) -> str:
        """Determine next step after input validation"""
        return "error" if state["error_messages"] else "continue"
    
    def _should_continue_after_retrieval(self, state: WorkflowState) -> str:
        """Determine next step after literature retrieval"""
        if state["error_messages"]:
            return "error"
        elif not state["retrieved_papers"]:
            return "no_results" 
        else:
            return "continue"
    
    def _should_continue_after_analysis(self, state: WorkflowState) -> str:
        """Determine next step after paper analysis"""
        if state["error_messages"]:
            return "error"
        elif state["requires_human_review"]:
            return "human_review"
        else:
            return "continue"
    
    def _should_finalize(self, state: WorkflowState) -> str:
        """Determine if workflow should finalize"""
        if state["error_messages"]:
            return "error"
        elif state["requires_human_review"] and not state["human_feedback"]:
            return "human_review"
        else:
            return "complete"
    
    def _handle_human_feedback(self, state: WorkflowState) -> str:
        """Process human feedback and determine next action"""
        feedback = state["human_feedback"]
        if not feedback:
            return "complete"
        
        action = feedback.get("action", "complete")
        if action == "reanalyze":
            return "reanalyze"
        elif action == "regenerate":
            return "regenerate"
        else:
            return "complete"
    
    # Helper Functions
    
    def _should_request_human_review(self, analysis_results: List[Dict[str, Any]]) -> bool:
        """
        Determine if human review is needed based on analysis results
        
        Concept: Smart Human-in-the-Loop Triggering
        - Analyzes result quality and confidence
        - Triggers human review for complex/ambiguous cases
        - Reduces manual intervention for clear-cut cases
        """
        # Check for low relevance scores
        avg_relevance = sum(result.get('relevance_score', 0) for result in analysis_results) / len(analysis_results) if analysis_results else 0
        
        if avg_relevance < 0.15:  # Low overall relevance
            return True
        
        # Check for conflicting themes across papers
        all_themes = []
        for result in analysis_results:
            all_themes.extend(result.get('themes', []))
        
        if len(set(all_themes)) > len(all_themes) * 0.8:  # High theme diversity might indicate scattered results
            return True
        
        # Check for analysis errors
        error_count = sum(1 for result in analysis_results if 'error' in result)
        if error_count > len(analysis_results) * 0.2:  # More than 20% errors
            return True
        
        return False
    
    async def _synthesize_research_report(
        self, 
        query: str, 
        papers: List[Dict[str, Any]], 
        analyses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize final research report from all gathered data
        
        Concept: Report Generation
        - Combines all workflow outputs into coherent report
        - Provides executive summary and detailed findings
        - Structures information for easy consumption
        """
        
        # Extract key insights
        all_themes = []
        all_concepts = []
        total_citations = 0
        
        for analysis in analyses:
            all_themes.extend(analysis.get('themes', []))
            all_concepts.extend(analysis.get('key_concepts', []))
            total_citations += analysis.get('citations', 0)
        
        # Count theme frequency
        theme_counts = {}
        for theme in all_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Generate report structure
        report = {
            'query': query,
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_papers': len(papers),
                'total_citations': total_citations,
                'top_themes': [theme for theme, count in top_themes],
                'analysis_depth': 'comprehensive' if len(papers) > 10 else 'focused'
            },
            'papers': papers,
            'analyses': analyses,
            'key_findings': {
                'dominant_themes': dict(top_themes),
                'research_gap_analysis': 'Generated based on theme analysis',
                'methodology_trends': 'Extracted from paper methodologies'
            },
            'recommendations': [
                'Further research areas based on gap analysis',
                'Key papers for deeper investigation',
                'Emerging themes to watch'
            ]
        }
        
        return report

# Create workflow instance
research_workflow = ResearchWorkflow()