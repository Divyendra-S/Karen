"""Graph builder module for LangGraph construction with functional composition."""

from typing import Dict, Any, Callable, Optional, List
from langgraph.graph import StateGraph, END
from datetime import datetime
from uuid import uuid4
from ..models.graph_state import GraphState, create_initial_graph_state


def create_jd_graph() -> StateGraph:
    """Create and configure the job description generation graph.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Import node functions (will be implemented in Phase 3.2)
    from .nodes.greeting import greeting_node
    from .nodes.question import question_router_node, question_generator_node
    from .nodes.collector import user_input_collector_node
    from .nodes.updater import state_updater_node
    from .nodes.checker import completeness_checker_node
    from .nodes.generator import jd_generator_node, output_node
    from .edges.conditionals import (
        should_continue_collecting,
        should_generate_jd,
        route_to_next_question
    )
    
    # Create graph with state schema
    graph = StateGraph(GraphState)
    
    # Add all nodes
    graph.add_node("greeting", greeting_node)
    graph.add_node("question_router", question_router_node) 
    graph.add_node("question_generator", question_generator_node)
    graph.add_node("user_input_collector", user_input_collector_node)
    graph.add_node("state_updater", state_updater_node)
    graph.add_node("completeness_checker", completeness_checker_node)
    graph.add_node("jd_generator", jd_generator_node)
    graph.add_node("output", output_node)
    
    # Set entry point
    graph.set_entry_point("greeting")
    
    # Add edges with conditional logic
    graph.add_edge("greeting", "question_router")
    
    # Conditional edge from question_router
    graph.add_conditional_edges(
        "question_router",
        route_to_next_question,
        {
            "question_generator": "question_generator",
            "completeness_checker": "completeness_checker",
            "jd_generator": "jd_generator"
        }
    )
    
    graph.add_edge("question_generator", "user_input_collector")
    graph.add_edge("user_input_collector", "state_updater")
    
    # Conditional edge from state_updater
    graph.add_conditional_edges(
        "state_updater", 
        should_continue_collecting,
        {
            "continue": "question_router",
            "complete": "completeness_checker"
        }
    )
    
    # Conditional edge from completeness_checker  
    graph.add_conditional_edges(
        "completeness_checker",
        should_generate_jd,
        {
            "generate": "jd_generator",
            "continue": "question_router"
        }
    )
    
    graph.add_edge("jd_generator", "output")
    graph.add_edge("output", END)
    
    return graph.compile()


def get_graph_visualization() -> str:
    """Get a text representation of the graph structure.
    
    Returns:
        ASCII art representation of the graph flow
    """
    return """
    [START]
        ↓
    [GREETING] 
        ↓
    [QUESTION_ROUTER] ←─────────┐
        ↓                      │
    [QUESTION_GENERATOR]       │
        ↓                      │
    [USER_INPUT_COLLECTOR]     │
        ↓                      │
    [STATE_UPDATER] ───────────┘
        ↓
    [COMPLETENESS_CHECKER]
        ↓
    [JD_GENERATOR]
        ↓
    [OUTPUT]
        ↓
    [END]
    """


# Graph configuration constants
GRAPH_CONFIG = {
    "max_retry_attempts": 3,
    "timeout_seconds": 300,
    "enable_checkpointing": True,
    "thread_id": "jd_generation_session"
}


def create_graph_config(
    max_retries: int = 3,
    timeout: int = 300,
    enable_checkpointing: bool = True
) -> Dict[str, Any]:
    """Create graph configuration dictionary.
    
    Args:
        max_retries: Maximum number of retry attempts
        timeout: Timeout in seconds for graph execution
        enable_checkpointing: Whether to enable state checkpointing
        
    Returns:
        Configuration dictionary for graph execution
    """
    return {
        "max_retry_attempts": max_retries,
        "timeout_seconds": timeout,
        "enable_checkpointing": enable_checkpointing,
        "thread_id": f"jd_session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    }


def validate_graph_structure(graph: StateGraph) -> tuple[bool, List[str]]:
    """Validate that graph structure is correct.
    
    Args:
        graph: The compiled StateGraph to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    try:
        # Check that graph has required nodes
        required_nodes = {
            "greeting", "question_router", "question_generator",
            "user_input_collector", "state_updater", "completeness_checker", 
            "jd_generator", "output"
        }
        
        graph_nodes = set(graph.nodes.keys())
        missing_nodes = required_nodes - graph_nodes
        
        if missing_nodes:
            errors.append(f"Missing required nodes: {missing_nodes}")
        
        # Check entry point is set
        if not hasattr(graph, 'entry_point') or graph.entry_point != "greeting":
            errors.append("Entry point should be 'greeting'")
            
        return len(errors) == 0, errors
        
    except Exception as e:
        return False, [f"Graph validation error: {str(e)}"]


# Node function type signature
NodeFunction = Callable[[GraphState], Dict[str, Any]]
EdgeFunction = Callable[[GraphState], str]