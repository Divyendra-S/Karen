"""Conversation models for managing chat state with functional patterns."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, FrozenSet
from pydantic import BaseModel, Field, field_validator
from uuid import UUID, uuid4


class MessageRole(str, Enum):
    """Message role types for conversation tracking."""

    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


class ConversationPhase(str, Enum):
    """Conversation phase states for flow management."""

    GREETING = "greeting"
    COLLECTING_BASIC_INFO = "collecting_basic_info"
    COLLECTING_EXPERIENCE = "collecting_experience"
    COLLECTING_SKILLS = "collecting_skills"
    COLLECTING_RESPONSIBILITIES = "collecting_responsibilities"
    COLLECTING_REQUIREMENTS = "collecting_requirements"
    REVIEWING_DATA = "reviewing_data"
    GENERATING_JD = "generating_jd"
    COMPLETED = "completed"
    ERROR = "error"


class MessageType(str, Enum):
    """Types of messages for proper handling."""

    QUESTION = "question"
    RESPONSE = "response"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    ERROR = "error"
    INFO = "info"


class Message(BaseModel):
    """Individual message in conversation with immutable design."""

    id: UUID = Field(default_factory=uuid4, description="Unique message identifier")

    role: MessageRole = Field(..., description="Role of the message sender")

    content: str = Field(
        ..., min_length=1, max_length=5000, description="Message content"
    )

    message_type: MessageType = Field(
        MessageType.RESPONSE, description="Type of message for handling logic"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the message was created"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional message metadata"
    )

    related_field: Optional[str] = Field(
        None, description="JD field this message relates to"
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate and clean message content."""
        return v.strip()

    def with_content(self, new_content: str) -> "Message":
        """Return new message with updated content (functional update)."""
        return self.model_copy(update={"content": new_content})

    def with_metadata(self, **kwargs) -> "Message":
        """Return new message with updated metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return self.model_copy(update={"metadata": new_metadata})


class ConversationMetadata(BaseModel):
    """Metadata for conversation tracking and analytics."""

    session_id: UUID = Field(
        default_factory=uuid4, description="Unique session identifier"
    )

    started_at: datetime = Field(
        default_factory=datetime.utcnow, description="When conversation started"
    )

    last_activity: datetime = Field(
        default_factory=datetime.utcnow, description="Last user activity timestamp"
    )

    total_messages: int = Field(
        0, ge=0, description="Total number of messages in conversation"
    )

    user_language: str = Field("en", description="Detected user language code")

    voice_enabled: bool = Field(False, description="Whether voice input is enabled")

    completion_time_seconds: Optional[int] = Field(
        None, ge=0, description="Time taken to complete JD generation"
    )

    user_satisfaction_score: Optional[int] = Field(
        None, ge=1, le=5, description="User satisfaction rating (1-5)"
    )

    @property
    def duration_minutes(self) -> float:
        """Calculate conversation duration in minutes."""
        delta = self.last_activity - self.started_at
        return delta.total_seconds() / 60

    def update_activity(self) -> "ConversationMetadata":
        """Return new metadata with updated activity timestamp."""
        return self.model_copy(
            update={
                "last_activity": datetime.utcnow(),
                "total_messages": self.total_messages + 1,
            }
        )


class ConversationState(BaseModel):
    """Immutable conversation state management."""

    current_phase: ConversationPhase = Field(
        ConversationPhase.GREETING, description="Current phase of the conversation"
    )

    completed_fields: FrozenSet[str] = Field(
        default_factory=frozenset, description="Set of completed JD fields"
    )

    pending_fields: List[str] = Field(
        default_factory=lambda: [
            "job_title",
            "department",
            "employment_type",
            "location",
            "experience",
            "responsibilities",
            "skills",
            "education",
        ],
        description="List of fields still to be collected",
    )

    current_field: Optional[str] = Field(
        None, description="Field currently being collected"
    )

    conversation_history: List[Message] = Field(
        default_factory=list, description="Complete conversation message history"
    )

    metadata: ConversationMetadata = Field(
        default_factory=ConversationMetadata,
        description="Conversation metadata and tracking",
    )

    validation_errors: List[str] = Field(
        default_factory=list, description="Current validation errors"
    )

    retry_count: int = Field(
        0, ge=0, le=3, description="Number of retries for current field"
    )

    @property
    def progress_percentage(self) -> float:
        """Calculate conversation progress percentage."""
        total_fields = len(self.completed_fields) + len(self.pending_fields)
        if total_fields == 0:
            return 0.0
        return len(self.completed_fields) / total_fields * 100

    @property
    def is_complete(self) -> bool:
        """Check if conversation is complete."""
        required_fields = {"job_title", "responsibilities", "skills"}
        return required_fields.issubset(self.completed_fields)

    @property
    def last_message(self) -> Optional[Message]:
        """Get the last message in conversation."""
        return self.conversation_history[-1] if self.conversation_history else None

    @property
    def user_messages(self) -> List[Message]:
        """Get all user messages."""
        return [
            msg for msg in self.conversation_history if msg.role == MessageRole.USER
        ]

    @property
    def assistant_messages(self) -> List[Message]:
        """Get all assistant messages."""
        return [
            msg
            for msg in self.conversation_history
            if msg.role == MessageRole.ASSISTANT
        ]

    def add_message(self, message: Message) -> "ConversationState":
        """Return new state with added message (functional update)."""
        new_history = self.conversation_history + [message]
        updated_metadata = self.metadata.update_activity()

        return self.model_copy(
            update={"conversation_history": new_history, "metadata": updated_metadata}
        )

    def complete_field(self, field_name: str) -> "ConversationState":
        """Return new state with field marked as completed."""
        new_completed = self.completed_fields | {field_name}
        new_pending = [f for f in self.pending_fields if f != field_name]

        return self.model_copy(
            update={
                "completed_fields": new_completed,
                "pending_fields": new_pending,
                "current_field": None,
                "retry_count": 0,
            }
        )

    def set_current_field(self, field_name: str) -> "ConversationState":
        """Return new state with current field set."""
        return self.model_copy(update={"current_field": field_name, "retry_count": 0})

    def advance_phase(self, new_phase: ConversationPhase) -> "ConversationState":
        """Return new state with advanced phase."""
        return self.model_copy(update={"current_phase": new_phase})

    def add_validation_error(self, error: str) -> "ConversationState":
        """Return new state with added validation error."""
        new_errors = self.validation_errors + [error]
        return self.model_copy(
            update={
                "validation_errors": new_errors,
                "retry_count": self.retry_count + 1,
            }
        )

    def clear_validation_errors(self) -> "ConversationState":
        """Return new state with cleared validation errors."""
        return self.model_copy(update={"validation_errors": []})

    model_config = {
        "json_schema_extra": {
            "example": {
                "current_phase": "collecting_basic_info",
                "completed_fields": ["job_title", "department"],
                "pending_fields": ["experience", "skills"],
                "current_field": "experience",
                "conversation_history": [
                    {
                        "role": "assistant",
                        "content": "What is the job title?",
                        "message_type": "question",
                    },
                    {
                        "role": "user",
                        "content": "Software Engineer",
                        "message_type": "response",
                    },
                ],
            }
        }
    }


def create_initial_conversation_state() -> ConversationState:
    """Create initial conversation state with default values."""
    return ConversationState()


def create_greeting_message() -> Message:
    """Create initial greeting message."""
    return Message(
        role=MessageRole.ASSISTANT,
        content="Hi! I'm here to help you create a comprehensive job description. I'll ask you a series of questions to gather all the necessary information. Let's start with the basics!",
        message_type=MessageType.INFO,
    )


def is_conversation_stale(state: ConversationState, max_idle_minutes: int = 30) -> bool:
    """Check if conversation has been idle too long."""
    idle_time = datetime.utcnow() - state.metadata.last_activity
    return idle_time.total_seconds() / 60 > max_idle_minutes


def get_next_pending_field(state: ConversationState) -> Optional[str]:
    """Get the next field to collect based on priority."""
    if not state.pending_fields:
        return None

    # Define priority order for field collection
    priority_order = [
        "job_title",
        "department",
        "employment_type",
        "location",
        "experience",
        "responsibilities",
        "skills",
        "education",
        "salary",
        "benefits",
        "additional_requirements",
    ]

    # Return highest priority pending field
    for field in priority_order:
        if field in state.pending_fields:
            return field

    # Fallback to first pending field
    return state.pending_fields[0]
