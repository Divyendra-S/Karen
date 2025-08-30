"""User input collector node with validation and processing."""

from typing import Dict, Any, Optional, List
from ...models.graph_state import GraphState


def get_last_user_message_content(state: GraphState) -> Optional[str]:
    """Helper to get last user message content."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def user_input_collector_node(state: GraphState) -> Dict[str, Any]:
    """Collect and process user input with validation.
    
    Pure function that processes user input for the current field,
    performs basic validation, and prepares for state update.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with processed user input
    """
    # Get the last user message (should be the input we're processing)
    user_input = get_last_user_message_content(state)
    current_field = state.get("current_field")
    
    if not user_input or not current_field:
        return add_validation_error_to_state(
            state, 
            "No user input or current field to process"
        )
    
    # Process input based on field type
    processed_value, validation_errors = process_user_input(
        user_input, 
        current_field, 
        state
    )
    
    if validation_errors:
        # Add validation errors to state
        updated_state = state
        for error in validation_errors:
            updated_state = add_validation_error_to_state(updated_state, error)
        return updated_state
    
    # Store processed value for state updater
    return {
        **state,
        "temp_processed_value": processed_value,
        "validation_errors": []
    }


def process_user_input(
    user_input: str, 
    field_name: str, 
    state: GraphState
) -> tuple[Any, List[str]]:
    """Process user input for specific field with validation.
    
    Args:
        user_input: Raw user input text
        field_name: Field being collected
        state: Current graph state for context
        
    Returns:
        Tuple of (processed_value, validation_errors)
    """
    # Clean input
    cleaned_input = user_input.strip()
    
    if not cleaned_input:
        return None, ["Please provide a response."]
    
    # Check for skip indicators
    skip_indicators = ["skip", "not sure", "don't know", "pass", "later"]
    if any(indicator in cleaned_input.lower() for indicator in skip_indicators):
        return None, []  # No error, just skip this field
    
    # Field-specific processing
    if field_name == "background_intro":
        return process_background_intro(cleaned_input)
    
    elif field_name == "job_title":
        return process_job_title(cleaned_input)
    
    elif field_name == "department":
        return process_department(cleaned_input)
    
    elif field_name == "employment_type":
        return process_employment_type(cleaned_input)
    
    elif field_name == "location":
        return process_location(cleaned_input, state)
    
    elif field_name == "experience":
        return process_experience(cleaned_input)
    
    elif field_name == "responsibilities":
        return process_responsibilities(cleaned_input)
    
    elif field_name == "skills":
        return process_skills(cleaned_input)
    
    elif field_name == "education":
        return process_education(cleaned_input)
    
    elif field_name == "salary":
        return process_salary(cleaned_input)
    
    elif field_name == "benefits":
        return process_benefits(cleaned_input)
    
    elif field_name == "additional_requirements":
        return process_additional_requirements(cleaned_input)
    
    else:
        return cleaned_input, []


def process_background_intro(input_text: str) -> tuple[str, List[str]]:
    """Process background introduction input."""
    errors = []
    
    if len(input_text) < 10:
        errors.append("Please tell me a bit more about yourself and your career interests.")
    
    return input_text.strip(), errors


def process_job_title(input_text: str) -> tuple[str, List[str]]:
    """Process job title input with validation."""
    errors = []
    
    if len(input_text) < 3:
        errors.append("Job title should be at least 3 characters long.")
    
    if len(input_text) > 100:
        errors.append("Job title should be less than 100 characters.")
    
    if any(char in input_text for char in ['@', '#', '$', '%', '&']):
        errors.append("Job title cannot contain special characters (@, #, $, %, &).")
    
    # Clean and format
    cleaned_title = input_text.strip().title()
    
    return cleaned_title, errors


def process_department(input_text: str) -> tuple[str, List[str]]:
    """Process department input with validation."""
    errors = []
    
    if len(input_text) > 100:
        errors.append("Department name should be less than 100 characters.")
    
    cleaned_dept = input_text.strip().title()
    return cleaned_dept, errors


def process_employment_type(input_text: str) -> tuple[str, List[str]]:
    """Process employment type input with validation."""
    input_lower = input_text.lower()
    
    # Map common inputs to standard types
    type_mapping = {
        "full time": "full_time",
        "full-time": "full_time", 
        "fulltime": "full_time",
        "permanent": "full_time",
        "part time": "part_time",
        "part-time": "part_time",
        "parttime": "part_time",
        "contract": "contract",
        "contractor": "contract",
        "consulting": "contract",
        "intern": "internship",
        "internship": "internship",
        "temp": "temporary",
        "temporary": "temporary",
        "freelance": "freelance",
        "freelancer": "freelance"
    }
    
    for key, value in type_mapping.items():
        if key in input_lower:
            return value, []
    
    # If no match found, return as-is with suggestion
    return input_text, [
        "I didn't recognize that employment type. Common types are: "
        "full-time, part-time, contract, internship, temporary, freelance."
    ]


def process_location(input_text: str, state: GraphState) -> tuple[Dict[str, Any], List[str]]:
    """Process location input with validation."""
    input_lower = input_text.lower()
    errors = []
    
    # Detect location type
    if any(word in input_lower for word in ["remote", "work from home", "wfh"]):
        location_type = "remote"
        city = None
        state_val = None
    elif any(word in input_lower for word in ["hybrid", "flexible", "mix"]):
        location_type = "hybrid"
        city, state_val = extract_city_state(input_text)
    else:
        location_type = "on_site"
        city, state_val = extract_city_state(input_text)
        
        if not city:
            errors.append("On-site positions need a city location. Please specify the city.")
    
    location_data = {
        "location_type": location_type,
        "city": city,
        "state": state_val,
        "country": "United States"  # Default, can be made configurable
    }
    
    return location_data, errors


def process_experience(input_text: str) -> tuple[Dict[str, Any], List[str]]:
    """Process experience input with validation."""
    import re
    
    errors = []
    
    # Extract years of experience
    years_match = re.search(r'(\d+)(?:\+)?\s*(?:years?|yrs?)', input_text.lower())
    years_min = int(years_match.group(1)) if years_match else 0
    
    # Determine experience level
    input_lower = input_text.lower()
    if any(word in input_lower for word in ["entry", "junior", "new grad", "0-2"]):
        level = "entry"
    elif any(word in input_lower for word in ["senior", "sr", "lead", "5+"]):
        level = "senior"
    elif any(word in input_lower for word in ["principal", "staff", "architect", "10+"]):
        level = "principal"
    else:
        level = "mid"
    
    # Check for leadership mentions
    leadership_required = any(
        word in input_lower 
        for word in ["leadership", "management", "team lead", "supervise"]
    )
    
    experience_data = {
        "level": level,
        "years_min": years_min,
        "leadership_required": leadership_required,
        "industry_experience": extract_industry_experience(input_text)
    }
    
    return experience_data, errors


def process_responsibilities(input_text: str) -> tuple[List[str], List[str]]:
    """Process responsibilities input with validation."""
    errors = []
    
    if len(input_text) < 20:
        errors.append("Please provide more detailed responsibilities (at least 20 characters).")
    
    # Split into individual responsibilities
    # Handle various separators: newlines, bullets, numbers, semicolons
    import re
    
    # Split by common separators
    separators = ['\n', '•', '-', '*', ';']
    responsibilities = [input_text]
    
    for sep in separators:
        new_resp = []
        for resp in responsibilities:
            new_resp.extend(resp.split(sep))
        responsibilities = new_resp
    
    # Clean and filter
    cleaned_responsibilities = []
    for resp in responsibilities:
        # Remove leading numbers/bullets
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', resp.strip())
        cleaned = cleaned.strip('•-* ')
        
        if len(cleaned) > 5:  # Minimum meaningful length
            cleaned_responsibilities.append(cleaned)
    
    if not cleaned_responsibilities:
        errors.append("Please provide at least one clear responsibility.")
    
    return cleaned_responsibilities, errors


def process_skills(input_text: str) -> tuple[Dict[str, Any], List[str]]:
    """Process skills input with validation."""
    errors = []
    
    if len(input_text) < 10:
        errors.append("Please provide more detailed skill requirements.")
    
    # Extract different types of skills
    technical_skills = extract_technical_skills(input_text)
    soft_skills = extract_soft_skills(input_text) 
    programming_languages = extract_programming_languages(input_text)
    frameworks_tools = extract_frameworks_tools(input_text)
    
    skills_data = {
        "technical_skills": [
            {"name": skill, "skill_type": "technical", "is_required": True}
            for skill in technical_skills
        ],
        "soft_skills": [
            {"name": skill, "skill_type": "soft", "is_required": True}
            for skill in soft_skills
        ],
        "programming_languages": programming_languages,
        "frameworks_tools": frameworks_tools
    }
    
    # Validate we have some skills
    total_skills = len(technical_skills) + len(soft_skills) + len(programming_languages)
    if total_skills == 0:
        errors.append("Please specify at least a few required skills.")
    
    return skills_data, errors


def process_education(input_text: str) -> tuple[Dict[str, Any], List[str]]:
    """Process education input with validation."""
    input_lower = input_text.lower()
    errors = []
    
    # Map education level
    if any(word in input_lower for word in ["phd", "doctorate", "doctoral"]):
        level = "doctorate"
    elif any(word in input_lower for word in ["master", "mba", "ms", "ma"]):
        level = "master"
    elif any(word in input_lower for word in ["bachelor", "ba", "bs", "degree"]):
        level = "bachelor"
    elif any(word in input_lower for word in ["associate", "aa", "as"]):
        level = "associate"
    elif any(word in input_lower for word in ["high school", "diploma", "ged"]):
        level = "high_school"
    elif any(word in input_lower for word in ["none", "not required", "no requirement"]):
        level = "none_required"
    else:
        level = "bachelor"  # Default assumption
    
    # Extract field of study
    field_of_study = extract_field_of_study(input_text)
    
    education_data = {
        "level": level,
        "field_of_study": field_of_study,
        "is_required": "not required" not in input_lower
    }
    
    return education_data, errors


def process_salary(input_text: str) -> tuple[Optional[Dict[str, Any]], List[str]]:
    """Process salary input with validation."""
    import re
    
    # Check if user wants to skip
    if any(word in input_text.lower() for word in ["skip", "not specify", "competitive", "negotiable"]):
        return None, []
    
    errors = []
    
    # Extract salary numbers
    salary_pattern = r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
    matches = re.findall(salary_pattern, input_text.replace(',', ''))
    
    if not matches:
        return None, ["Please specify a salary amount or say 'skip' to leave blank."]
    
    # Convert to numbers
    amounts = [float(match.replace(',', '')) for match in matches]
    
    if len(amounts) == 1:
        # Single amount - treat as minimum or base
        min_salary = amounts[0]
        max_salary = None
    else:
        # Range provided
        min_salary = min(amounts)
        max_salary = max(amounts)
    
    # Detect frequency
    frequency = "annual"  # Default
    if any(word in input_text.lower() for word in ["hour", "hourly", "/hr"]):
        frequency = "hourly"
    elif any(word in input_text.lower() for word in ["month", "monthly", "/mo"]):
        frequency = "monthly"
    
    salary_data = {
        "min_salary": min_salary,
        "max_salary": max_salary,
        "currency": "USD",
        "frequency": frequency,
        "is_negotiable": "negotiable" in input_text.lower()
    }
    
    return salary_data, errors


def process_benefits(input_text: str) -> tuple[List[str], List[str]]:
    """Process benefits input with validation."""
    # Check if user wants to skip
    if any(word in input_text.lower() for word in ["skip", "none", "not applicable"]):
        return [], []
    
    errors = []
    
    # Split benefits into list
    import re
    separators = ['\n', '•', '-', '*', ';', ',']
    benefits = [input_text]
    
    for sep in separators:
        new_benefits = []
        for benefit in benefits:
            new_benefits.extend(benefit.split(sep))
        benefits = new_benefits
    
    # Clean benefits
    cleaned_benefits = []
    for benefit in benefits:
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', benefit.strip())
        cleaned = cleaned.strip('•-* ')
        
        if len(cleaned) > 3:
            cleaned_benefits.append(cleaned)
    
    return cleaned_benefits, errors


def process_additional_requirements(input_text: str) -> tuple[Optional[str], List[str]]:
    """Process additional requirements input."""
    # Check if user wants to skip
    if any(word in input_text.lower() for word in ["skip", "none", "no additional"]):
        return None, []
    
    errors = []
    
    if len(input_text) > 1000:
        errors.append("Additional requirements should be less than 1000 characters.")
    
    return input_text.strip(), errors


# Helper functions for extraction

def extract_city_state(text: str) -> tuple[Optional[str], Optional[str]]:
    """Extract city and state from location text."""
    import re
    
    # Pattern for "City, State" format
    city_state_pattern = r'([A-Za-z\s]+),\s*([A-Za-z]{2,})'
    match = re.search(city_state_pattern, text)
    
    if match:
        return match.group(1).strip().title(), match.group(2).strip().title()
    
    # If no comma, treat whole text as city
    city = text.strip().title()
    return city if city else None, None


def extract_industry_experience(text: str) -> Optional[str]:
    """Extract industry experience mentions from text."""
    industries = [
        "fintech", "healthcare", "education", "e-commerce", "gaming",
        "automotive", "aerospace", "manufacturing", "retail", "consulting"
    ]
    
    text_lower = text.lower()
    found_industries = [industry for industry in industries if industry in text_lower]
    
    return ", ".join(found_industries) if found_industries else None


def extract_technical_skills(text: str) -> List[str]:
    """Extract technical skills from text."""
    # Common technical skills to look for
    technical_keywords = [
        "python", "javascript", "java", "react", "angular", "vue",
        "node.js", "django", "flask", "sql", "postgresql", "mysql",
        "mongodb", "aws", "azure", "docker", "kubernetes", "git",
        "ci/cd", "testing", "agile", "scrum"
    ]
    
    text_lower = text.lower()
    found_skills = []
    
    for skill in technical_keywords:
        if skill in text_lower:
            found_skills.append(skill.title())
    
    # Also extract by common patterns
    import re
    # Look for capitalized tech terms
    tech_pattern = r'\b[A-Z][a-z]*[A-Z][A-Za-z]*\b'
    tech_matches = re.findall(tech_pattern, text)
    found_skills.extend(tech_matches)
    
    return list(set(found_skills))  # Remove duplicates


def extract_soft_skills(text: str) -> List[str]:
    """Extract soft skills from text."""
    soft_skill_keywords = [
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "time management", "creativity", "adaptability",
        "attention to detail", "analytical", "collaborative"
    ]
    
    text_lower = text.lower()
    found_skills = []
    
    for skill in soft_skill_keywords:
        if skill in text_lower:
            found_skills.append(skill.title())
    
    return found_skills


def extract_programming_languages(text: str) -> List[str]:
    """Extract programming languages from text."""
    languages = [
        "python", "javascript", "java", "c++", "c#", "go", "rust",
        "typescript", "php", "ruby", "swift", "kotlin", "scala",
        "r", "matlab", "sql"
    ]
    
    text_lower = text.lower()
    found_languages = []
    
    for lang in languages:
        if lang in text_lower:
            found_languages.append(lang.title())
    
    return found_languages


def extract_frameworks_tools(text: str) -> List[str]:
    """Extract frameworks and tools from text."""
    frameworks_tools = [
        "react", "angular", "vue", "django", "flask", "express",
        "spring", "laravel", "rails", "docker", "kubernetes",
        "aws", "azure", "gcp", "jenkins", "gitlab", "jira"
    ]
    
    text_lower = text.lower()
    found_tools = []
    
    for tool in frameworks_tools:
        if tool in text_lower:
            found_tools.append(tool.title())
    
    return found_tools


def extract_field_of_study(text: str) -> Optional[str]:
    """Extract field of study from education text."""
    common_fields = [
        "computer science", "engineering", "business", "marketing",
        "finance", "economics", "psychology", "mathematics",
        "statistics", "communications", "design"
    ]
    
    text_lower = text.lower()
    for field in common_fields:
        if field in text_lower:
            return field.title()
    
    # Look for "in [field]" pattern
    import re
    field_pattern = r'in\s+([A-Za-z\s]+)'
    match = re.search(field_pattern, text_lower)
    if match:
        return match.group(1).strip().title()
    
    return None


def add_validation_error_to_state(state: GraphState, error: str) -> GraphState:
    """Add validation error to state (helper function)."""
    new_errors = state.get("validation_errors", []) + [error]
    new_retry_count = state.get("retry_count", 0) + 1
    
    return {
        **state,
        "validation_errors": new_errors,
        "retry_count": new_retry_count
    }