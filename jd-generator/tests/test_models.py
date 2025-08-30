"""Unit tests for Pydantic models with functional testing patterns."""

import pytest
from datetime import datetime
from decimal import Decimal
from core.models.job_requirements import (
    JobRequirements, EmploymentType, LocationRequirement, LocationType,
    ExperienceRequirement, ExperienceLevel, SkillsRequirement, Skill, SkillType,
    EducationRequirement, EducationLevel, SalaryRange,
    create_default_job_requirements, validate_job_requirements
)
from core.models.conversation import (
    ConversationState, Message, MessageRole, ConversationPhase,
    ConversationMetadata, create_initial_conversation_state
)
from core.models.graph_state import (
    GraphState, create_initial_graph_state, validate_graph_state,
    convert_to_job_requirements
)


class TestJobRequirements:
    """Test job requirements models."""
    
    def test_create_default_job_requirements(self):
        """Test creating default job requirements."""
        job_req = create_default_job_requirements()
        assert job_req.job_title == "Tbd"
        assert job_req.location.location_type == LocationType.REMOTE
        assert job_req.experience.level == ExperienceLevel.MID
        assert job_req.responsibilities == ["TBD"]
    
    def test_job_title_validation(self):
        """Test job title validation."""
        with pytest.raises(ValueError, match="cannot contain special characters"):
            JobRequirements(
                job_title="Engineer@Company",
                location=LocationRequirement(location_type=LocationType.REMOTE),
                experience=ExperienceRequirement(level=ExperienceLevel.MID)
            )
    
    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        job_req = JobRequirements(
            job_title="Software Engineer",
            location=LocationRequirement(location_type=LocationType.REMOTE),
            experience=ExperienceRequirement(level=ExperienceLevel.SENIOR),
            responsibilities=["Develop software"],
            skills=SkillsRequirement(
                technical_skills=[
                    Skill(name="Python", skill_type=SkillType.TECHNICAL)
                ]
            )
        )
        assert job_req.completion_percentage == 1.0
        assert job_req.is_complete


class TestLocationRequirement:
    """Test location requirement model."""
    
    def test_on_site_requires_city(self):
        """Test that on-site positions require a city."""
        with pytest.raises(ValueError, match="On-site positions must specify a city"):
            LocationRequirement(location_type=LocationType.ON_SITE)
    
    def test_travel_percentage_validation(self):
        """Test travel percentage validation."""
        with pytest.raises(ValueError, match="Travel percentage specified but travel not required"):
            LocationRequirement(
                location_type=LocationType.REMOTE,
                travel_percentage=25,
                travel_required=False
            )


class TestExperienceRequirement:
    """Test experience requirement model."""
    
    def test_years_validation(self):
        """Test years validation logic."""
        with pytest.raises(ValueError, match="Maximum years cannot be less than minimum"):
            ExperienceRequirement(
                level=ExperienceLevel.MID,
                years_min=5,
                years_max=3
            )
    
    def test_level_years_consistency(self):
        """Test experience level and years consistency."""
        with pytest.raises(ValueError, match="entry level typically requires"):
            ExperienceRequirement(
                level=ExperienceLevel.ENTRY,
                years_min=5  # Entry level max is 2 years
            )


class TestSalaryRange:
    """Test salary range model."""
    
    def test_salary_validation(self):
        """Test salary range validation."""
        with pytest.raises(ValueError, match="Minimum salary cannot exceed maximum"):
            SalaryRange(
                min_salary=Decimal("100000"),
                max_salary=Decimal("80000")
            )
    
    def test_computed_properties(self):
        """Test computed salary properties."""
        salary = SalaryRange(
            min_salary=Decimal("80000"),
            max_salary=Decimal("120000")
        )
        assert salary.midpoint == Decimal("100000")
        assert salary.range_width == Decimal("40000")
        assert "80,000 - 120,000" in salary.formatted_range


class TestSkillsRequirement:
    """Test skills requirement model."""
    
    def test_skill_properties(self):
        """Test skill property methods."""
        skills = SkillsRequirement(
            technical_skills=[
                Skill(name="Python", skill_type=SkillType.TECHNICAL, is_required=True),
                Skill(name="Docker", skill_type=SkillType.TECHNICAL, is_required=False)
            ],
            programming_languages=["Python", "JavaScript"]
        )
        
        required_skills = skills.required_technical_skills
        assert "Python" in required_skills
        assert "Docker" not in required_skills
        
        all_skills = skills.all_skill_names
        assert "Python" in all_skills
        assert "JavaScript" in all_skills
        assert isinstance(all_skills, frozenset)


class TestConversationModels:
    """Test conversation state models."""
    
    def test_initial_conversation_state(self):
        """Test initial conversation state creation."""
        state = create_initial_conversation_state()
        assert state.current_phase == ConversationPhase.GREETING
        assert len(state.completed_fields) == 0
        assert len(state.pending_fields) > 0
        assert not state.is_complete
    
    def test_message_creation(self):
        """Test message model creation."""
        message = Message(
            role=MessageRole.USER,
            content="Software Engineer"
        )
        assert message.role == MessageRole.USER
        assert message.content == "Software Engineer"
        assert isinstance(message.timestamp, datetime)
    
    def test_conversation_state_updates(self):
        """Test functional state updates."""
        state = create_initial_conversation_state()
        
        # Add message
        message = Message(role=MessageRole.USER, content="Test")
        new_state = state.add_message(message)
        
        assert len(new_state.conversation_history) == 1
        assert new_state.metadata.total_messages == 1
        
        # Original state unchanged (immutability)
        assert len(state.conversation_history) == 0
    
    def test_field_completion(self):
        """Test field completion logic."""
        state = create_initial_conversation_state()
        
        # Complete a field
        new_state = state.complete_field("job_title")
        
        assert "job_title" in new_state.completed_fields
        assert "job_title" not in new_state.pending_fields
        assert new_state.current_field is None
        assert new_state.retry_count == 0


class TestGraphState:
    """Test graph state models."""
    
    def test_initial_graph_state(self):
        """Test initial graph state creation."""
        state = create_initial_graph_state()
        assert state["messages"] == []
        assert state["job_data"] == {}
        assert not state["is_complete"]
        assert state["conversation_phase"] == ConversationPhase.GREETING.value
    
    def test_graph_state_validation(self):
        """Test graph state validation."""
        valid_state = create_initial_graph_state()
        is_valid, errors = validate_graph_state(valid_state)
        assert is_valid
        assert len(errors) == 0
    
    def test_invalid_graph_state(self):
        """Test invalid graph state detection."""
        invalid_state = {
            "messages": "not a list",  # Should be list
            "job_data": {},
            "current_field": None,
            "validation_errors": [],
            "is_complete": False,
            "generated_jd": None,
            "conversation_phase": "invalid_phase",
            "retry_count": -1,  # Should be >= 0
            "session_metadata": {}
        }
        
        is_valid, errors = validate_graph_state(invalid_state)
        assert not is_valid
        assert len(errors) > 0


class TestFunctionalValidation:
    """Test functional validation patterns."""
    
    def test_validate_job_requirements_success(self):
        """Test successful job requirements validation."""
        data = {
            "job_title": "Software Engineer",
            "location": {"location_type": "remote"},
            "experience": {"level": "mid"},
            "responsibilities": ["Develop software"],
            "skills": {
                "technical_skills": [
                    {"name": "Python", "skill_type": "technical"}
                ]
            }
        }
        
        job_req, errors = validate_job_requirements(data)
        assert job_req is not None
        assert len(errors) == 0
        assert job_req.is_complete
    
    def test_validate_job_requirements_failure(self):
        """Test job requirements validation with errors."""
        data = {
            "job_title": "E@",  # Invalid characters
            "location": {"location_type": "invalid_type"}
        }
        
        job_req, errors = validate_job_requirements(data)
        assert job_req is None
        assert len(errors) > 0
    
    def test_immutable_updates(self):
        """Test that model updates return new instances."""
        original = create_default_job_requirements()
        updated = original.with_field("job_title", "New Title")
        
        # Original unchanged
        assert original.job_title == "Tbd"
        # New instance has update
        assert updated.job_title == "New Title"


# Parametrized tests for enum values
@pytest.mark.parametrize("employment_type", [
    EmploymentType.FULL_TIME,
    EmploymentType.PART_TIME,
    EmploymentType.CONTRACT,
    EmploymentType.INTERNSHIP
])
def test_employment_types(employment_type):
    """Test all employment type values are valid."""
    # Use appropriate experience level for internships
    experience_level = ExperienceLevel.ENTRY if employment_type == EmploymentType.INTERNSHIP else ExperienceLevel.MID
    
    job_req = JobRequirements(
        job_title="Test Job",
        employment_type=employment_type,
        location=LocationRequirement(location_type=LocationType.REMOTE),
        experience=ExperienceRequirement(level=experience_level)
    )
    assert job_req.employment_type == employment_type


@pytest.mark.parametrize("location_type", [
    LocationType.ON_SITE,
    LocationType.REMOTE, 
    LocationType.HYBRID
])
def test_location_types(location_type):
    """Test all location type values are valid."""
    # For on-site, include city
    location_data = {"location_type": location_type}
    if location_type == LocationType.ON_SITE:
        location_data["city"] = "Test City"
    
    location = LocationRequirement(**location_data)
    assert location.location_type == location_type