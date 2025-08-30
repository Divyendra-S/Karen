# Claude AI Assistant Guidelines - JD Generator Project

## 🎯 Project Context
This is an AI-powered Job Description Generator using LangGraph, Streamlit, and voice capabilities. The project follows a functional programming approach with strong typing and validation.

## 🐍 Python Best Practices

### Code Style
- **Python Version**: 3.10+ (use modern Python features)
- **Type Hints**: Always use type hints for function arguments and return values
- **Docstrings**: Use Google-style docstrings for all functions and classes
- **Line Length**: Maximum 88 characters (Black formatter default)
- **Imports**: Group imports in order: standard library, third-party, local

### Naming Conventions
```python
# Classes: PascalCase
class JobRequirement:
    pass

# Functions/Variables: snake_case
def process_user_input(text: str) -> str:
    user_response = text.strip()
    return user_response

# Constants: UPPER_SNAKE_CASE
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT = 30

# Private functions/methods: prefix with underscore
def _validate_internal(data: dict) -> bool:
    pass
```

## 🎯 Functional Programming Approach

### Pure Functions
```python
# ✅ GOOD: Pure function - no side effects
def calculate_completeness_score(completed_fields: set, required_fields: set) -> float:
    if not required_fields:
        return 0.0
    return len(completed_fields & required_fields) / len(required_fields)

# ❌ BAD: Function with side effects
def update_and_calculate(data: dict, fields: set) -> float:
    data['updated'] = True  # Side effect!
    return len(fields) / 10
```

### Immutability
```python
from typing import FrozenSet, Tuple, NamedTuple
from dataclasses import dataclass, replace

# Use immutable data structures
@dataclass(frozen=True)
class JobConfig:
    title: str
    department: str
    
    def with_title(self, new_title: str) -> 'JobConfig':
        return replace(self, title=new_title)

# Use NamedTuple for simple immutable structures
class ResponseData(NamedTuple):
    message: str
    success: bool
    data: dict
```

### Function Composition
```python
from functools import partial, reduce
from typing import Callable, List

# Compose functions
def compose(*functions: Callable) -> Callable:
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

# Use partial application
validate_email = partial(re.match, r'^[\w\.-]+@[\w\.-]+\.\w+$')

# Pipeline pattern
def process_pipeline(data: str) -> str:
    pipeline = compose(
        str.strip,
        str.lower,
        lambda x: x.replace(' ', '_')
    )
    return pipeline(data)
```

### Avoid Mutations
```python
# ✅ GOOD: Return new data
def add_field(job_data: dict, field: str, value: Any) -> dict:
    return {**job_data, field: value}

# ❌ BAD: Mutate in place
def add_field_bad(job_data: dict, field: str, value: Any) -> None:
    job_data[field] = value  # Mutation!
```

## 📦 Pydantic Best Practices

### Model Definition
```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum

class EmploymentType(str, Enum):
    """Use string enums for better serialization"""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"

class JobRequirements(BaseModel):
    """Always use BaseModel with clear field definitions"""
    
    # Use Field for validation and documentation
    job_title: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Job position title"
    )
    
    # Optional with defaults
    department: Optional[str] = Field(
        None,
        description="Department or team name"
    )
    
    # Enums for constrained choices
    employment_type: EmploymentType = Field(
        EmploymentType.FULL_TIME,
        description="Type of employment"
    )
    
    # Lists with validation
    responsibilities: List[str] = Field(
        default_factory=list,
        min_items=1,
        max_items=20,
        description="Key job responsibilities"
    )
    
    # Nested models
    salary: Optional['SalaryRange'] = None
    
    # Computed fields
    @property
    def is_remote(self) -> bool:
        return self.location and "remote" in self.location.lower()
    
    # Field validators
    @field_validator('job_title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        if any(char in v for char in ['@', '#', '$']):
            raise ValueError('Job title cannot contain special characters')
        return v.strip().title()
    
    # Model validator
    @model_validator(mode='after')
    def validate_model(self) -> 'JobRequirements':
        if self.employment_type == EmploymentType.INTERNSHIP:
            if self.experience_years and self.experience_years > 1:
                raise ValueError('Internships cannot require more than 1 year experience')
        return self
    
    # Configuration
    model_config = {
        "json_schema_extra": {
            "example": {
                "job_title": "Software Engineer",
                "department": "Engineering",
                "employment_type": "full_time"
            }
        }
    }
```

### Validation Patterns
```python
from pydantic import ValidationError, validator
from typing import Union

# Custom validators
class SmartValidator(BaseModel):
    email: str
    phone: Optional[str] = None
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.replace('-', '').replace('+', '').isdigit():
            raise ValueError('Phone must contain only digits')
        return v

# Error handling
def safe_parse(data: dict) -> Union[JobRequirements, dict]:
    try:
        return JobRequirements(**data)
    except ValidationError as e:
        return {"errors": e.errors()}
```

### Settings Management
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Centralized settings management"""
    
    # API Configuration
    openai_api_key: str = Field(..., env='OPENAI_API_KEY')
    model_name: str = Field('gpt-4', env='MODEL_NAME')
    
    # App Configuration
    debug: bool = Field(False, env='DEBUG')
    max_retries: int = Field(3, env='MAX_RETRIES')
    
    class Config:
        env_file = '.env'
        case_sensitive = False
        
@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton"""
    return Settings()
```

## 🏗️ LangGraph Integration Patterns

### State Management
```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph

class GraphState(TypedDict):
    """Use TypedDict for graph state"""
    messages: List[dict]
    job_data: dict
    current_field: Optional[str]
    is_complete: bool
    
def create_graph() -> StateGraph:
    """Functional graph builder"""
    graph = StateGraph(GraphState)
    
    # Add nodes functionally
    graph.add_node("greeting", greeting_node)
    graph.add_node("collect", collector_node)
    
    # Add edges
    graph.add_edge("greeting", "collect")
    
    return graph.compile()
```

### Node Functions
```python
from typing import Dict, Any

def greeting_node(state: GraphState) -> Dict[str, Any]:
    """Pure function node - returns state updates"""
    return {
        "messages": state["messages"] + [
            {"role": "assistant", "content": "Welcome!"}
        ]
    }

# Use closures for parameterized nodes
def create_question_node(question_bank: List[str]) -> Callable:
    def question_node(state: GraphState) -> Dict[str, Any]:
        next_question = select_question(question_bank, state)
        return {"current_field": next_question}
    return question_node
```

## 🧪 Testing Patterns

### Unit Testing
```python
import pytest
from unittest.mock import Mock, patch

# Fixtures for reusable test data
@pytest.fixture
def sample_job_data():
    return {
        "job_title": "Software Engineer",
        "department": "Engineering"
    }

# Parametrized tests
@pytest.mark.parametrize("input_text,expected", [
    ("software engineer", "Software Engineer"),
    ("DATA ANALYST", "Data Analyst"),
])
def test_title_formatting(input_text, expected):
    result = format_job_title(input_text)
    assert result == expected

# Mock external dependencies
@patch('openai.ChatCompletion.create')
def test_llm_call(mock_openai):
    mock_openai.return_value = Mock(choices=[...])
    result = generate_question("test")
    assert result is not None
```

## 🔧 Utility Functions

### Error Handling
```python
from typing import Optional, Union, TypeVar
from functools import wraps
import logging

T = TypeVar('T')

def safe_execute(default: T = None) -> Callable:
    """Decorator for safe execution with default return"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Union[T, Any]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}")
                return default
        return wrapper
    return decorator

@safe_execute(default=[])
def get_responsibilities(job_data: dict) -> List[str]:
    return job_data["responsibilities"]
```

### Data Processing
```python
from itertools import groupby
from operator import itemgetter

def group_by_category(items: List[dict]) -> dict:
    """Functional grouping"""
    sorted_items = sorted(items, key=itemgetter('category'))
    return {
        k: list(v) 
        for k, v in groupby(sorted_items, key=itemgetter('category'))
    }

def filter_complete_fields(fields: dict) -> dict:
    """Filter with dict comprehension"""
    return {k: v for k, v in fields.items() if v is not None}
```

## 📝 Documentation Standards

### Function Documentation
```python
def process_user_input(
    text: str,
    context: Optional[dict] = None,
    validate: bool = True
) -> Tuple[str, bool]:
    """Process and validate user input.
    
    Args:
        text: Raw user input text
        context: Optional context dictionary for validation
        validate: Whether to perform validation
        
    Returns:
        Tuple of (processed_text, is_valid)
        
    Raises:
        ValueError: If text is empty
        ValidationError: If validation fails
        
    Example:
        >>> process_user_input("  Software Engineer  ")
        ("Software Engineer", True)
    """
    if not text:
        raise ValueError("Text cannot be empty")
    
    processed = text.strip()
    is_valid = _validate_input(processed) if validate else True
    
    return processed, is_valid
```

## 🚀 Performance Considerations

### Async Operations
```python
import asyncio
from typing import List

async def fetch_multiple(urls: List[str]) -> List[dict]:
    """Concurrent fetching"""
    tasks = [fetch_single(url) for url in urls]
    return await asyncio.gather(*tasks)

# Cache expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(param: str) -> dict:
    """Cached computation"""
    return perform_calculation(param)
```

## 📋 Checklist for New Features

When implementing new features:
- [ ] Use type hints for all functions
- [ ] Write pure functions when possible
- [ ] Create Pydantic models for data validation
- [ ] Add comprehensive docstrings
- [ ] Include error handling
- [ ] Write unit tests
- [ ] Avoid mutations and side effects
- [ ] Use functional patterns (map, filter, reduce)
- [ ] Implement proper logging
- [ ] Follow naming conventions

## 🔄 Code Review Checklist

Before committing:
- [ ] Run `black .` for formatting
- [ ] Run `ruff check .` for linting
- [ ] Run `mypy .` for type checking
- [ ] Run `pytest` for tests
- [ ] Ensure all functions have docstrings
- [ ] Check for any hardcoded values
- [ ] Verify error handling is in place
- [ ] Confirm no sensitive data in code

## 🎯 Commands to Run

```bash
# Format code
black .
ruff check . --fix

# Type checking
mypy .

# Run tests
pytest -v

# Check test coverage
pytest --cov=core --cov=app

# Run the application
streamlit run app/main.py
```