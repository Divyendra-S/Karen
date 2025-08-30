"""LangGraph state models using TypedDict approach with functional validation."""

from typing import TypedDict, List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field, ValidationError
from .job_requirements import JobRequirements
from .conversation import ConversationState, Message, ConversationPhase


class GraphState(TypedDict):
    """LangGraph state schema using TypedDict for compatibility."""
    
    # Message history for LangGraph
    messages: List[Dict[str, Any]]
    
    # Job data dictionary (will be converted to JobRequirements)
    job_data: Dict[str, Any]
    
    # Current field being collected
    current_field: Optional[str]
    
    # Validation errors from last operation
    validation_errors: List[str]
    
    # Conversation completion status
    is_complete: bool
    
    # Generated job description
    generated_jd: Optional[str]
    
    # Current conversation phase
    conversation_phase: str
    
    # Retry count for current operation
    retry_count: int
    
    # Session metadata
    session_metadata: Dict[str, Any]


class GraphStateValidator(BaseModel):
    """Pydantic model for validating GraphState data."""
    
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Message history for the conversation"
    )
    
    job_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job requirements data being collected"
    )
    
    current_field: Optional[str] = Field(
        None,
        description="Field currently being collected"
    )
    
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Current validation errors"
    )
    
    is_complete: bool = Field(
        False,
        description="Whether job data collection is complete"
    )
    
    generated_jd: Optional[str] = Field(
        None,
        description="Generated job description text"
    )
    
    conversation_phase: str = Field(
        ConversationPhase.GREETING.value,
        description="Current conversation phase"
    )
    
    retry_count: int = Field(
        0,
        ge=0,
        le=3,
        description="Number of retries for current operation"
    )
    
    session_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Session tracking metadata"
    )


def create_initial_graph_state() -> GraphState:
    """Create initial graph state with default values."""
    return GraphState(
        messages=[],
        job_data={},
        current_field=None,
        validation_errors=[],
        is_complete=False,
        generated_jd=None,
        conversation_phase=ConversationPhase.GREETING.value,
        retry_count=0,
        session_metadata={}
    )


def validate_graph_state(state: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Validate graph state using Pydantic validator."""
    try:
        GraphStateValidator(**state)
        return True, []
    except ValidationError as e:
        errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
        return False, errors


def update_graph_state(
    current_state: GraphState, 
    updates: Dict[str, Any]
) -> GraphState:
    """Functionally update graph state with new values."""
    # Create new state with updates
    new_state = {**current_state, **updates}
    
    # Validate the new state
    is_valid, errors = validate_graph_state(new_state)
    if not is_valid:
        raise ValueError(f"Invalid state update: {errors}")
    
    return GraphState(new_state)


def add_message_to_state(state: GraphState, message: Dict[str, Any]) -> GraphState:
    """Add a message to the graph state."""
    new_messages = state["messages"] + [message]
    return update_graph_state(state, {"messages": new_messages})


def update_job_field(
    state: GraphState, 
    field_name: str, 
    field_value: Any
) -> GraphState:
    """Update a specific job field in the state."""
    new_job_data = {**state["job_data"], field_name: field_value}
    
    # Validate the updated job data
    try:
        JobRequirements(**new_job_data)
        validation_errors = []
    except ValidationError as e:
        validation_errors = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
    
    return update_graph_state(state, {
        "job_data": new_job_data,
        "validation_errors": validation_errors
    })


def mark_field_complete(state: GraphState, field_name: str) -> GraphState:
    """Mark a field as complete and advance to next field."""
    # Get next field to collect
    field_priority = [
        "job_title", "department", "employment_type", "location",
        "experience", "responsibilities", "skills", "education",
        "salary", "benefits", "additional_requirements"
    ]
    
    # Find next uncollected field
    collected_fields = set(state["job_data"].keys())
    next_field = None
    
    for field in field_priority:
        if field not in collected_fields and field != field_name:
            next_field = field
            break
    
    # Check if we have all required fields
    required_fields = {"job_title", "responsibilities", "skills"}
    is_complete = required_fields.issubset(collected_fields | {field_name})
    
    # Determine next phase
    if is_complete:
        next_phase = ConversationPhase.REVIEWING_DATA.value
    elif next_field in ["job_title", "department", "employment_type"]:
        next_phase = ConversationPhase.COLLECTING_BASIC_INFO.value
    elif next_field == "experience":
        next_phase = ConversationPhase.COLLECTING_EXPERIENCE.value
    elif next_field == "skills":
        next_phase = ConversationPhase.COLLECTING_SKILLS.value
    elif next_field == "responsibilities":
        next_phase = ConversationPhase.COLLECTING_RESPONSIBILITIES.value
    else:
        next_phase = ConversationPhase.COLLECTING_REQUIREMENTS.value
    
    return update_graph_state(state, {
        "current_field": next_field,
        "is_complete": is_complete,
        "conversation_phase": next_phase,
        "retry_count": 0,
        "validation_errors": []
    })


def convert_to_job_requirements(state: GraphState) -> Optional[JobRequirements]:
    """Convert graph state job data to JobRequirements model."""
    try:
        return JobRequirements(**state["job_data"])
    except ValidationError:
        return None


def convert_conversation_to_graph(conv_state: ConversationState) -> GraphState:
    """Convert ConversationState to GraphState for LangGraph compatibility."""
    # Convert messages to dict format
    messages = [
        {
            "role": msg.role.value,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "message_type": msg.message_type.value
        }
        for msg in conv_state.conversation_history
    ]
    
    return GraphState(
        messages=messages,
        job_data={},  # Will be populated as conversation progresses
        current_field=conv_state.current_field,
        validation_errors=conv_state.validation_errors,
        is_complete=conv_state.is_complete,
        generated_jd=None,
        conversation_phase=conv_state.current_phase.value,
        retry_count=conv_state.retry_count,
        session_metadata={
            "session_id": str(conv_state.metadata.session_id),
            "started_at": conv_state.metadata.started_at.isoformat(),
            "voice_enabled": conv_state.metadata.voice_enabled
        }
    )


def convert_graph_to_conversation(graph_state: GraphState) -> ConversationState:
    """Convert GraphState back to ConversationState."""
    from datetime import datetime
    from uuid import UUID
    
    # Convert messages back to Message objects
    messages = []
    for msg_dict in graph_state["messages"]:
        message = Message(
            role=msg_dict["role"],
            content=msg_dict["content"],
            message_type=msg_dict.get("message_type", "response"),
            timestamp=datetime.fromisoformat(msg_dict.get("timestamp", datetime.utcnow().isoformat()))
        )
        messages.append(message)
    
    # Determine completed fields from job_data
    completed_fields = frozenset(graph_state["job_data"].keys())
    
    # Create conversation metadata
    session_meta = graph_state["session_metadata"]
    metadata = ConversationMetadata(
        session_id=UUID(session_meta.get("session_id", str(uuid4()))),
        started_at=datetime.fromisoformat(session_meta.get("started_at", datetime.utcnow().isoformat())),
        voice_enabled=session_meta.get("voice_enabled", False),
        total_messages=len(messages)
    )
    
    return ConversationState(
        current_phase=ConversationPhase(graph_state["conversation_phase"]),
        completed_fields=completed_fields,
        current_field=graph_state["current_field"],
        conversation_history=messages,
        metadata=metadata,
        validation_errors=graph_state["validation_errors"],
        retry_count=graph_state["retry_count"]
    )


# Type aliases for function signatures
StateUpdateFunction = Callable[[GraphState], GraphState]
FieldValidationFunction = Callable[[Any], tuple[bool, List[str]]]
MessageGeneratorFunction = Callable[[GraphState], Dict[str, Any]]