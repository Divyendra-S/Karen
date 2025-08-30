"""Job requirements models using Pydantic with functional validation patterns."""

from enum import Enum
from typing import List, Optional, Set, FrozenSet
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator, model_validator


class EmploymentType(str, Enum):
    """Employment type enumeration with string values for better serialization."""
    
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"
    FREELANCE = "freelance"


class LocationType(str, Enum):
    """Location requirement types."""
    
    ON_SITE = "on_site"
    REMOTE = "remote"
    HYBRID = "hybrid"


class ExperienceLevel(str, Enum):
    """Experience level classifications."""
    
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


class EducationLevel(str, Enum):
    """Education level requirements."""
    
    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"
    PROFESSIONAL = "professional"
    NONE_REQUIRED = "none_required"


class SkillType(str, Enum):
    """Types of skills for categorization."""
    
    TECHNICAL = "technical"
    SOFT = "soft"
    LANGUAGE = "language"
    CERTIFICATION = "certification"


class LocationRequirement(BaseModel):
    """Location and remote work requirements."""
    
    location_type: LocationType = Field(
        ...,
        description="Type of work location arrangement"
    )
    
    city: Optional[str] = Field(
        None,
        description="City where the job is located"
    )
    
    state: Optional[str] = Field(
        None,
        description="State/province where the job is located"
    )
    
    country: str = Field(
        "United States",
        description="Country where the job is located"
    )
    
    timezone_requirements: Optional[str] = Field(
        None,
        description="Required timezone overlap for remote work"
    )
    
    travel_required: bool = Field(
        False,
        description="Whether travel is required for this position"
    )
    
    travel_percentage: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Percentage of time spent traveling"
    )
    
    @field_validator('city', 'state')
    @classmethod
    def validate_location_text(cls, v: Optional[str]) -> Optional[str]:
        """Validate and clean location text."""
        if v is None:
            return v
        return v.strip().title()
    
    @model_validator(mode='after')
    def validate_location_consistency(self) -> 'LocationRequirement':
        """Validate location requirements are consistent."""
        if self.location_type == LocationType.ON_SITE and not self.city:
            raise ValueError("On-site positions must specify a city")
        
        if self.travel_percentage and not self.travel_required:
            raise ValueError("Travel percentage specified but travel not required")
            
        return self


class ExperienceRequirement(BaseModel):
    """Experience requirements with functional validation."""
    
    level: ExperienceLevel = Field(
        ...,
        description="Required experience level"
    )
    
    years_min: int = Field(
        0,
        ge=0,
        le=50,
        description="Minimum years of experience required"
    )
    
    years_max: Optional[int] = Field(
        None,
        ge=0,
        le=50,
        description="Maximum years of experience (for targeting)"
    )
    
    industry_experience: Optional[str] = Field(
        None,
        description="Specific industry experience requirements"
    )
    
    leadership_required: bool = Field(
        False,
        description="Whether leadership experience is required"
    )
    
    management_experience: bool = Field(
        False,
        description="Whether people management experience is required"
    )
    
    @model_validator(mode='after')
    def validate_experience_consistency(self) -> 'ExperienceRequirement':
        """Validate experience requirements are logical."""
        if self.years_max and self.years_max < self.years_min:
            raise ValueError("Maximum years cannot be less than minimum years")
        
        # Validate level consistency with years
        level_year_mapping = {
            ExperienceLevel.ENTRY: (0, 2),
            ExperienceLevel.JUNIOR: (1, 3),
            ExperienceLevel.MID: (3, 7),
            ExperienceLevel.SENIOR: (5, 12),
            ExperienceLevel.LEAD: (7, 15),
            ExperienceLevel.PRINCIPAL: (10, 25)
        }
        
        min_expected, max_expected = level_year_mapping.get(self.level, (0, 50))
        
        if self.years_min > 0 and self.years_min > max_expected:
            raise ValueError(f"{self.level.value} level typically requires {min_expected}-{max_expected} years")
        
        return self


class Skill(BaseModel):
    """Individual skill with metadata."""
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Skill name"
    )
    
    skill_type: SkillType = Field(
        ...,
        description="Type/category of skill"
    )
    
    is_required: bool = Field(
        True,
        description="Whether this skill is required or preferred"
    )
    
    proficiency_level: Optional[str] = Field(
        None,
        description="Required proficiency level (beginner, intermediate, advanced)"
    )
    
    @field_validator('name')
    @classmethod
    def validate_skill_name(cls, v: str) -> str:
        """Validate and clean skill name."""
        return v.strip()


class SkillsRequirement(BaseModel):
    """Skills requirements with immutable collections."""
    
    technical_skills: List[Skill] = Field(
        default_factory=list,
        description="Technical skills required for the position"
    )
    
    soft_skills: List[Skill] = Field(
        default_factory=list,
        description="Soft skills required for the position"
    )
    
    certifications: List[str] = Field(
        default_factory=list,
        description="Professional certifications required"
    )
    
    programming_languages: List[str] = Field(
        default_factory=list,
        description="Programming languages required"
    )
    
    frameworks_tools: List[str] = Field(
        default_factory=list,
        description="Frameworks and tools experience required"
    )
    
    @property
    def required_technical_skills(self) -> List[str]:
        """Get list of required technical skill names."""
        return [
            skill.name for skill in self.technical_skills 
            if skill.is_required and skill.skill_type == SkillType.TECHNICAL
        ]
    
    @property
    def all_skill_names(self) -> FrozenSet[str]:
        """Get immutable set of all skill names."""
        all_skills = []
        all_skills.extend([skill.name for skill in self.technical_skills])
        all_skills.extend([skill.name for skill in self.soft_skills])
        all_skills.extend(self.programming_languages)
        all_skills.extend(self.frameworks_tools)
        all_skills.extend(self.certifications)
        return frozenset(all_skills)


class EducationRequirement(BaseModel):
    """Education requirements with enum-based validation."""
    
    level: EducationLevel = Field(
        EducationLevel.BACHELOR,
        description="Required education level"
    )
    
    field_of_study: Optional[str] = Field(
        None,
        description="Preferred field of study"
    )
    
    is_required: bool = Field(
        True,
        description="Whether education requirement is mandatory"
    )
    
    alternatives_accepted: List[str] = Field(
        default_factory=list,
        description="Alternative qualifications accepted (e.g., equivalent experience)"
    )
    
    @field_validator('field_of_study')
    @classmethod
    def validate_field_of_study(cls, v: Optional[str]) -> Optional[str]:
        """Validate and clean field of study."""
        if v is None:
            return v
        return v.strip().title()


class SalaryRange(BaseModel):
    """Salary range with computed properties and validation."""
    
    min_salary: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Minimum salary amount"
    )
    
    max_salary: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Maximum salary amount"
    )
    
    currency: str = Field(
        "USD",
        description="Currency code (ISO 4217)"
    )
    
    frequency: str = Field(
        "annual",
        description="Salary frequency (annual, monthly, hourly)"
    )
    
    is_negotiable: bool = Field(
        True,
        description="Whether salary is negotiable"
    )
    
    equity_offered: bool = Field(
        False,
        description="Whether equity/stock options are offered"
    )
    
    bonus_structure: Optional[str] = Field(
        None,
        description="Description of bonus or commission structure"
    )
    
    @property
    def midpoint(self) -> Optional[Decimal]:
        """Calculate salary midpoint."""
        if self.min_salary is None or self.max_salary is None:
            return None
        return (self.min_salary + self.max_salary) / 2
    
    @property
    def range_width(self) -> Optional[Decimal]:
        """Calculate salary range width."""
        if self.min_salary is None or self.max_salary is None:
            return None
        return self.max_salary - self.min_salary
    
    @property
    def formatted_range(self) -> str:
        """Format salary range for display."""
        if self.min_salary is None and self.max_salary is None:
            return "Salary not specified"
        
        if self.min_salary and self.max_salary:
            return f"{self.currency} {self.min_salary:,.0f} - {self.max_salary:,.0f} ({self.frequency})"
        elif self.min_salary:
            return f"{self.currency} {self.min_salary:,.0f}+ ({self.frequency})"
        elif self.max_salary:
            return f"Up to {self.currency} {self.max_salary:,.0f} ({self.frequency})"
        
        return "Competitive salary"
    
    @model_validator(mode='after')
    def validate_salary_range(self) -> 'SalaryRange':
        """Validate salary range is logical."""
        if (self.min_salary is not None and 
            self.max_salary is not None and 
            self.min_salary > self.max_salary):
            raise ValueError("Minimum salary cannot exceed maximum salary")
        
        return self


class JobRequirements(BaseModel):
    """Main job requirements model with immutable data and comprehensive validation."""
    
    job_title: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Job position title"
    )
    
    department: Optional[str] = Field(
        None,
        max_length=100,
        description="Department or team name"
    )
    
    employment_type: EmploymentType = Field(
        EmploymentType.FULL_TIME,
        description="Type of employment arrangement"
    )
    
    location: LocationRequirement = Field(
        ...,
        description="Location and remote work requirements"
    )
    
    experience: ExperienceRequirement = Field(
        ...,
        description="Experience requirements for the position"
    )
    
    responsibilities: List[str] = Field(
        default_factory=list,
        min_length=1,
        max_length=20,
        description="Key job responsibilities and duties"
    )
    
    skills: SkillsRequirement = Field(
        default_factory=SkillsRequirement,
        description="Skills and competency requirements"
    )
    
    education: EducationRequirement = Field(
        default_factory=EducationRequirement,
        description="Education and qualification requirements"
    )
    
    salary: Optional[SalaryRange] = Field(
        None,
        description="Salary range and compensation details"
    )
    
    benefits: List[str] = Field(
        default_factory=list,
        description="Benefits and perks offered"
    )
    
    additional_requirements: Optional[str] = Field(
        None,
        max_length=1000,
        description="Additional requirements or notes"
    )
    
    @field_validator('job_title')
    @classmethod
    def validate_job_title(cls, v: str) -> str:
        """Validate and clean job title."""
        if any(char in v for char in ['@', '#', '$', '%', '&']):
            raise ValueError('Job title cannot contain special characters')
        return v.strip().title()
    
    @field_validator('responsibilities')
    @classmethod
    def validate_responsibilities(cls, v: List[str]) -> List[str]:
        """Validate and clean responsibilities list."""
        return [resp.strip() for resp in v if resp.strip()]
    
    @field_validator('benefits')
    @classmethod
    def validate_benefits(cls, v: List[str]) -> List[str]:
        """Validate and clean benefits list."""
        return [benefit.strip() for benefit in v if benefit.strip()]
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage of required fields."""
        required_fields = {
            'job_title': bool(self.job_title),
            'location': True,  # Always present due to default
            'experience': True,  # Always present due to default
            'responsibilities': bool(self.responsibilities),
            'skills': bool(self.skills.technical_skills or self.skills.soft_skills)
        }
        
        completed = sum(required_fields.values())
        return completed / len(required_fields)
    
    @property
    def missing_required_fields(self) -> FrozenSet[str]:
        """Get set of missing required fields."""
        missing = set()
        
        if not self.job_title:
            missing.add('job_title')
        if not self.responsibilities:
            missing.add('responsibilities')
        if not (self.skills.technical_skills or self.skills.soft_skills):
            missing.add('skills')
            
        return frozenset(missing)
    
    @property
    def is_complete(self) -> bool:
        """Check if all required fields are complete."""
        return len(self.missing_required_fields) == 0
    
    @model_validator(mode='after')
    def validate_job_consistency(self) -> 'JobRequirements':
        """Validate overall job requirements consistency."""
        # Validate internship constraints
        if self.employment_type == EmploymentType.INTERNSHIP:
            if (self.experience.years_min > 1 or 
                self.experience.level not in [ExperienceLevel.ENTRY, ExperienceLevel.JUNIOR]):
                raise ValueError('Internships should not require significant experience')
        
        # Validate senior roles have appropriate requirements (only if years_min > 0)
        if (self.experience.level in [ExperienceLevel.SENIOR, ExperienceLevel.LEAD, ExperienceLevel.PRINCIPAL] and
            self.experience.years_min > 0 and self.experience.years_min < 3):
            raise ValueError(f'{self.experience.level.value} roles typically require 3+ years experience')
        
        return self
    
    def with_field(self, field_name: str, value) -> 'JobRequirements':
        """Return new instance with updated field (functional update pattern)."""
        return self.model_copy(update={field_name: value})
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "job_title": "Senior Software Engineer",
                "department": "Engineering",
                "employment_type": "full_time",
                "location": {
                    "location_type": "hybrid",
                    "city": "San Francisco",
                    "state": "California",
                    "country": "United States"
                },
                "experience": {
                    "level": "senior",
                    "years_min": 5,
                    "years_max": 10,
                    "leadership_required": True
                },
                "responsibilities": [
                    "Design and develop scalable software solutions",
                    "Lead technical architecture decisions",
                    "Mentor junior developers"
                ],
                "skills": {
                    "technical_skills": [
                        {"name": "Python", "skill_type": "technical", "is_required": True},
                        {"name": "React", "skill_type": "technical", "is_required": True}
                    ],
                    "programming_languages": ["Python", "JavaScript"],
                    "frameworks_tools": ["React", "Django", "PostgreSQL"]
                }
            }
        }
    }


def create_default_job_requirements() -> JobRequirements:
    """Create a default JobRequirements instance with minimal required fields."""
    return JobRequirements(
        job_title="TBD",  # Placeholder that meets minimum length
        location=LocationRequirement(location_type=LocationType.REMOTE),
        experience=ExperienceRequirement(level=ExperienceLevel.MID),
        responsibilities=["TBD"],  # Placeholder that meets minimum length
        skills=SkillsRequirement(),
        education=EducationRequirement()
    )


def validate_job_requirements(data: dict) -> tuple[Optional[JobRequirements], List[str]]:
    """Functional validation that returns either valid model or errors."""
    try:
        job_req = JobRequirements(**data)
        return job_req, []
    except Exception as e:
        errors = [str(e)]
        return None, errors


def merge_job_requirements(base: JobRequirements, updates: dict) -> JobRequirements:
    """Functionally merge updates into job requirements."""
    return base.model_copy(update=updates)