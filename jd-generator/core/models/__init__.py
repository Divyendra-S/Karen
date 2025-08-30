"""Core models package for JD Generator."""

# Job requirements models
from .job_requirements import (
    JobRequirements,
    EmploymentType,
    LocationRequirement,
    LocationType,
    ExperienceRequirement,
    ExperienceLevel,
    SkillsRequirement,
    Skill,
    SkillType,
    EducationRequirement,
    EducationLevel,
    SalaryRange,
    create_default_job_requirements,
    validate_job_requirements,
    merge_job_requirements
)

# Conversation models
from .conversation import (
    ConversationState,
    Message,
    MessageRole,
    ConversationPhase,
    MessageType,
    ConversationMetadata,
    create_initial_conversation_state,
    create_greeting_message,
    is_conversation_stale,
    get_next_pending_field
)

# Graph state models
from .graph_state import (
    GraphState,
    GraphStateValidator,
    create_initial_graph_state,
    validate_graph_state,
    update_graph_state,
    add_message_to_state,
    save_raw_response,
    save_processed_data,
    mark_field_complete,
    convert_to_job_requirements,
    convert_conversation_to_graph,
    convert_graph_to_conversation
)

__all__ = [
    # Job requirements
    "JobRequirements",
    "EmploymentType", 
    "LocationRequirement",
    "LocationType",
    "ExperienceRequirement",
    "ExperienceLevel",
    "SkillsRequirement",
    "Skill",
    "SkillType",
    "EducationRequirement", 
    "EducationLevel",
    "SalaryRange",
    "create_default_job_requirements",
    "validate_job_requirements",
    "merge_job_requirements",
    
    # Conversation
    "ConversationState",
    "Message",
    "MessageRole",
    "ConversationPhase",
    "MessageType", 
    "ConversationMetadata",
    "create_initial_conversation_state",
    "create_greeting_message",
    "is_conversation_stale",
    "get_next_pending_field",
    
    # Graph state
    "GraphState",
    "GraphStateValidator",
    "create_initial_graph_state",
    "validate_graph_state",
    "update_graph_state",
    "add_message_to_state",
    "save_raw_response",
    "save_processed_data",
    "mark_field_complete",
    "convert_to_job_requirements",
    "convert_conversation_to_graph",
    "convert_graph_to_conversation"
]