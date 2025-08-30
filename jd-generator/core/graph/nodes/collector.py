"""User input collector node with two-layer storage approach."""

from typing import Dict, Any, Optional, Tuple
from ...models.graph_state import GraphState, save_raw_response, save_processed_data
import streamlit as st
from app.config import settings
import logging

logger = logging.getLogger(__name__)


def get_last_user_message_content(state: GraphState) -> Optional[str]:
    """Helper to get last user message content."""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return None


def user_input_collector_node(state: GraphState, context: dict = None) -> Dict[str, Any]:
    """Collect user input with two-layer storage: raw + processed.
    
    CRITICAL: Always save user's exact input first, then interpret separately.
    PROACTIVE: Extract ANY relevant information from user input, not just current field.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with both raw and processed user input
    """
    user_input = get_last_user_message_content(state)
    current_field = state.get("current_field")
    
    if not user_input:
        return {
            **state,
            "clarification_needed": state.get("clarification_needed", []) + ["missing_input"]
        }
    
    logger.info(f"Processing: '{user_input}' for field: {current_field}")
    
    # STEP 1: PROACTIVE EXTRACTION - Extract ALL relevant info from ANY input
    extracted_fields = extract_all_relevant_fields(user_input, state, context)
    updated_state = state
    
    # Save all extracted fields after validation
    for field, value in extracted_fields.items():
        if value is not None:
            # Validate before saving
            is_valid, validated_value = validate_extracted_field(field, value)
            if is_valid:
                updated_state = save_raw_response(updated_state, field, user_input)
                updated_state = save_processed_data(updated_state, field, validated_value)
                logger.info(f"🎯 Proactively extracted {field}: '{validated_value}' from user input")
            else:
                logger.warning(f"❌ Invalid extraction for {field}: {validated_value}")
    
    # CRITICAL: Skip Step 2 if current_field was already extracted to prevent loops
    if current_field and current_field in extracted_fields:
        logger.info(f"⏭️ Skipping Step 2 for {current_field} - already extracted proactively")
        return updated_state
    
    # STEP 2: Handle current field if we're asking about one specifically
    if current_field:
        # Step 2a: Save raw input for current field
        updated_state = save_raw_response(updated_state, current_field, user_input)
        
        # Step 2b: Try to extract/interpret current field value with context
        extracted_value, needs_clarification = extract_field_value_with_llm(
            user_input, 
            current_field, 
            updated_state,
            context
        )
        
        if needs_clarification:
            # Mark field for clarification, don't save processed data yet
            current_retry = updated_state.get("retry_count", 0)
            new_retry_count = min(current_retry + 1, 3)  # Cap at 3
            return {
                **updated_state,
                "clarification_needed": updated_state.get("clarification_needed", []) + [current_field],
                "retry_count": new_retry_count
            }
        
        # Step 2c: Validate and save extracted/cleaned value if successful
        if extracted_value is not None:
            is_valid, validated_value = validate_extracted_field(current_field, extracted_value)
            if is_valid:
                final_state = save_processed_data(updated_state, current_field, validated_value)
                logger.info(f"Saved - Raw: '{user_input}' → Processed: '{validated_value}'")
                return final_state
            else:
                # Validation failed, ask for clarification
                logger.warning(f"❌ Validation failed for {current_field}: {validated_value}")
                return {
                    **updated_state,
                    "clarification_needed": updated_state.get("clarification_needed", []) + [current_field],
                    "retry_count": min(updated_state.get("retry_count", 0) + 1, 3)
                }
        
        # If extraction returned None but no clarification needed, skip field
        logger.info(f"Field {current_field} skipped by user")
    
    return updated_state


def extract_all_relevant_fields(
    user_input: str, 
    state: GraphState, 
    context: dict = None
) -> Dict[str, Any]:
    """Proactively extract ALL relevant field information from any user input.
    
    This function analyzes user input and extracts any professional information
    that can be used to populate fields, regardless of what we're currently asking about.
    
    Args:
        user_input: User's message text
        state: Current graph state  
        context: Optional conversation context
        
    Returns:
        Dictionary mapping field names to extracted values
    """
    try:
        # Build conversation context for better extraction
        conversation_history = ""
        if context:
            conversation_history = format_conversation_history(context.get("recent_messages", []))
        
        # Get context about what we're currently asking for
        current_field = state.get("current_field", "")
        recently_asked = state.get("messages", [])[-2:] if state.get("messages") else []
        last_ai_question = ""
        for msg in reversed(recently_asked):
            if msg.get("role") == "assistant":
                last_ai_question = msg.get("content", "")
                break
        
        # Build extraction prompt avoiding f-string issues
        field_context = current_field.replace('_', ' ') if current_field else 'general info'
        
        extraction_prompt = """CONVERSATION CONTEXT:
Last AI Question: "{}"
Currently collecting: {}

USER INPUT: "{}"

CRITICAL EXTRACTION RULES:
1. ONLY extract information that is EXPLICITLY and CLEARLY stated
2. DO NOT extract names from greetings like "hello", "hi", "yes", "ok"
3. DO NOT extract job titles from vague responses like "yes", "sure", "exactly"
4. DO NOT guess or infer anything not directly stated
5. For names: ONLY extract when user says "I am [Name]" or "My name is [Name]"
6. For job titles: ONLY extract when user states their actual role
7. For experience: ONLY extract when numbers are explicitly mentioned with "years"

FIELD DEFINITIONS:
- name: Their actual first/last name (not "yes", "hello", etc.)
- job_title: Their current actual job role (not descriptions or responses)
- experience: Years of work experience (must include numbers)
- skills: Specific technologies/tools mentioned
- responsibilities: Specific job duties they mention
- location: City, state, or "Remote" if mentioned
- department: Specific team/department name
- education: Degree type and field mentioned

VALID EXTRACTION EXAMPLES:
"I am Sarah" → {{"name": "Sarah"}}
"I'm a software engineer" → {{"job_title": "Software Engineer"}}
"I have 3 years experience" → {{"experience": "3 years"}}
"I work with Python and React" → {{"skills": "Python, React"}}

INVALID EXTRACTIONS (return empty):
"yes" → {{}} (not a name)
"hello" → {{}} (not professional info)  
"exactly" → {{}} (not specific info)
"sure" → {{}} (confirmation, not data)
"ok" → {{}} (not information)

Extract from user input above. Return ONLY JSON with explicitly stated information or {{}} if no clear professional data.""".format(last_ai_question, field_context, user_input)

        if hasattr(st, 'session_state') and 'groq_client' in st.session_state:
            groq_client = st.session_state.groq_client
            
            logger.info(f"🔄 Calling Groq API for proactive extraction from: '{user_input}'")
            completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": extraction_prompt}],
                model=settings.llm_model,
                max_tokens=200,
                temperature=0.1
            )
            
            llm_response = completion.choices[0].message.content.strip()
            logger.info(f"✅ LLM Extraction response: '{llm_response}'")
            
            # Parse JSON response - handle mixed responses
            import json
            try:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{[^}]*\}', llm_response)
                if json_match:
                    json_str = json_match.group(0)
                    extracted_data = json.loads(json_str)
                    if isinstance(extracted_data, dict):
                        return extracted_data
                
                # If no JSON found, try parsing the whole response
                extracted_data = json.loads(llm_response)
                if isinstance(extracted_data, dict):
                    return extracted_data
                    
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from LLM response: {llm_response}")
                # Return empty dict for simple greetings and non-informative inputs
                return {}
                
        # Fallback extraction using basic patterns
        return extract_basic_patterns(user_input)
        
    except Exception as e:
        logger.error(f"Error in proactive extraction: {e}")
        return extract_basic_patterns(user_input)


def extract_basic_patterns(user_input: str) -> Dict[str, Any]:
    """Conservative pattern extraction as fallback - only extract obvious patterns."""
    import re
    extracted = {}
    input_lower = user_input.lower().strip()
    
    # Skip extraction for non-informative inputs
    non_informative = ["yes", "ok", "sure", "exactly", "hi", "hello", "hey", "good", "nice"]
    if input_lower in non_informative:
        return {}
    
    # Extract name - very conservative
    name_patterns = [
        r"i am ([A-Z][a-z]+)",  # Capitalized names only
        r"my name is ([A-Z][a-z]+)",
        r"call me ([A-Z][a-z]+)"
    ]
    for pattern in name_patterns:
        match = re.search(pattern, user_input)  # Use original case
        if match:
            name = match.group(1)
            # Validate it's actually a name
            if len(name) > 2 and name.lower() not in ["looking", "trying", "working", "here", "good", "software", "developer"]:
                extracted["name"] = name
                break
    
    # Extract job titles - very specific patterns
    job_patterns = [
        r"i am a ([a-z\s]+(?:engineer|developer|analyst|manager|designer|architect))",
        r"i work as a ([a-z\s]+(?:engineer|developer|analyst|manager|designer|architect))",
        r"i'm a ([a-z\s]+(?:engineer|developer|analyst|manager|designer|architect))"
    ]
    for pattern in job_patterns:
        match = re.search(pattern, input_lower)
        if match:
            job = match.group(1).strip()
            if len(job) > 5:  # Must be substantial
                extracted["job_title"] = job.title()
                break
    
    # Extract years of experience - strict number requirement
    exp_patterns = [
        r"(\d+)\s*years?\s*(?:of\s*)?experience",
        r"(\d+)\s*years?\s*in\s*(?:software|development|programming)",
        r"(\d+)\+?\s*years?\s*(?:experience|exp)"
    ]
    for pattern in exp_patterns:
        match = re.search(pattern, input_lower)
        if match:
            years = match.group(1)
            if int(years) > 0 and int(years) < 50:  # Reasonable range
                extracted["experience"] = f"{years} years"
                break
    
    # Extract skills - only clear technology mentions
    found_skills = []
    # More specific skill detection
    skill_patterns = [
        r"\b(python|javascript|java|react|angular|vue|node\.?js|django|flask)\b",
        r"\b(sql|postgresql|mysql|mongodb|redis)\b",
        r"\b(aws|azure|gcp|docker|kubernetes)\b",
        r"\b(git|github|gitlab)\b"
    ]
    for pattern in skill_patterns:
        matches = re.findall(pattern, input_lower)
        for match in matches:
            if match not in found_skills:
                found_skills.append(match.title())
    
    if found_skills:
        extracted["skills"] = ", ".join(found_skills)
    
    # Extract location - be more specific
    if "remote" in input_lower and "work" in input_lower:
        extracted["location"] = "Remote"
    
    return extracted


def validate_extracted_field(field_name: str, value: Any) -> tuple[bool, str]:
    """Validate that extracted field value is reasonable.
    
    Args:
        field_name: Name of the field
        value: Extracted value
        
    Returns:
        Tuple of (is_valid, cleaned_value or error_message)
    """
    if value is None or str(value).strip() == "":
        return False, "Empty value"
    
    value_str = str(value).strip()
    
    # Field-specific validation
    if field_name == "name":
        # Name should be 2-50 characters, no special chars
        if len(value_str) < 2 or len(value_str) > 50:
            return False, "Name should be 2-50 characters"
        if any(char in value_str for char in ['@', '#', '$', '%', '&']):
            return False, "Name contains invalid characters"
        # Check it's not a common non-name
        invalid_names = ["yes", "hello", "hi", "ok", "sure", "exactly", "good", "nice"]
        if value_str.lower() in invalid_names:
            return False, "Not a valid name"
        return True, value_str.title()
    
    elif field_name == "job_title":
        # Job title should be 3-100 characters
        if len(value_str) < 3 or len(value_str) > 100:
            return False, "Job title should be 3-100 characters"
        # Check it's not a generic response
        invalid_titles = ["yes", "sure", "exactly", "good", "nice", "ok"]
        if value_str.lower() in invalid_titles:
            return False, "Not a valid job title"
        return True, value_str.title()
    
    elif field_name == "experience":
        # Should contain years or have numbers
        import re
        if re.search(r'\d+', value_str):
            return True, value_str
        return False, "Experience should include years or numbers"
    
    elif field_name == "skills":
        # Should be meaningful skills, not generic responses
        if len(value_str) < 3:
            return False, "Skills should be more specific"
        invalid_skills = ["yes", "sure", "good", "nice", "everything", "all"]
        if value_str.lower() in invalid_skills:
            return False, "Please specify actual technologies or skills"
        return True, value_str
    
    elif field_name == "location":
        # Location should be city/remote/hybrid
        if len(value_str) < 2:
            return False, "Location should be more specific"
        return True, value_str.title()
    
    else:
        # Generic validation for other fields
        if len(value_str) < 2:
            return False, "Response too short"
        invalid_responses = ["yes", "ok", "sure", "exactly", "good", "nice"]
        if value_str.lower() in invalid_responses:
            return False, "Please provide more specific information"
        return True, value_str
    
    return True, value_str


def extract_field_value_with_llm(
    user_input: str, 
    field_name: str, 
    state: GraphState,
    context: dict = None
) -> Tuple[Any, bool]:
    """Extract field value from user input using LLM interpretation.
    
    Returns:
        Tuple of (extracted_value, needs_clarification)
        - extracted_value: Cleaned/interpreted value or None if user skipped
        - needs_clarification: True if user response was unclear
    """
    try:
        # Get conversation context
        conversation_history = ""
        last_ai_question = "Unknown"
        
        if context:
            conversation_history = format_conversation_history(context.get("recent_messages", []))
            last_ai_question = context.get("last_ai_question", "Unknown")
        
        # Create context-aware extraction prompt with better field understanding
        extraction_prompt = f"""CONVERSATION CONTEXT:
{conversation_history}

SITUATION:
- I asked: "{last_ai_question}"
- User responded: "{user_input}"
- Target field: {field_name}

TASK: Determine if user is answering the question about {field_name}.

STRICT ANALYSIS RULES:
1. User DIRECTLY answers about {field_name} → Extract clean value
2. User gives confirmation ("yes", "sure", "exactly") → "NEED_CLARIFICATION" 
3. User gives vague response ("good", "nice", "ok") → "NEED_CLARIFICATION"
4. User provides unrelated info → "USER_CONTEXT"
5. User explicitly skips ("skip", "not sure", "don't know") → "SKIP"

FIELD-SPECIFIC EXTRACTION:
• job_title: Extract only actual job roles, not confirmations
• experience: Must include numbers ("5 years", "2+ years")
• skills: Extract specific technologies/tools only
• responsibilities: Extract specific job duties/tasks
• name: Extract only from clear introductions
• location: Extract cities, states, or "Remote"
• department: Extract specific team/dept names

EXAMPLES:
Q: "What's your current role?" A: "software engineer" → "Software Engineer"
Q: "How many years experience?" A: "about 5 years" → "5 years"
Q: "What technologies?" A: "Python and React" → "Python, React"
Q: "Your current role?" A: "yes exactly" → "NEED_CLARIFICATION"
Q: "What's your name?" A: "hello there" → "NEED_CLARIFICATION"

Respond with ONLY: extracted value, "USER_CONTEXT", "NEED_CLARIFICATION", or "SKIP"."""

        if hasattr(st, 'session_state') and 'groq_client' in st.session_state:
            groq_client = st.session_state.groq_client
            
            logger.info(f"Extracting {field_name} from: '{user_input}'")
            
            completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": extraction_prompt}],
                model=settings.llm_model,
                max_tokens=100,
                temperature=0.1
            )
            
            llm_response = completion.choices[0].message.content.strip()
            
            logger.info(f"LLM extraction result: '{llm_response}'")
            
            # Handle LLM response
            if llm_response.upper() == "NEED_CLARIFICATION":
                logger.info(f"❓ Clarification needed for {field_name}")
                return None, True
            elif llm_response.upper() == "SKIP":
                logger.info(f"⏭️ User skipped {field_name}")
                return None, False
            elif llm_response.upper() == "USER_CONTEXT":
                logger.info(f"👤 User providing context, not answering {field_name}")
                return None, False  # Don't extract, but also don't need clarification
            else:
                logger.info(f"✅ Extracted: '{llm_response}' for {field_name}")
                return llm_response, False
            
        else:
            # Fallback without LLM
            return extract_field_value_basic(user_input, field_name)
            
    except Exception as e:
        logger.error(f"Error in extraction: {e}")
        return extract_field_value_basic(user_input, field_name)


def extract_field_value_basic(
    user_input: str, 
    field_name: str
) -> Tuple[Any, bool]:
    """Basic extraction without LLM (fallback only)."""
    cleaned_input = user_input.strip()
    
    if not cleaned_input:
        return None, True  # Need clarification
    
    # Check for skip indicators
    skip_indicators = ["skip", "not sure", "don't know", "pass", "later"]
    if any(indicator in cleaned_input.lower() for indicator in skip_indicators):
        return None, False  # User skipped
    
    # Basic extraction - minimal processing to preserve user intent
    if field_name == "job_title":
        return cleaned_input.title(), False
    elif field_name == "department":
        return cleaned_input.title(), False
    elif field_name == "employment_type":
        # Only extract if explicitly mentioned
        input_lower = cleaned_input.lower()
        if "full-time" in input_lower or "full time" in input_lower:
            return "full_time", False
        elif "part-time" in input_lower or "part time" in input_lower:
            return "part_time", False
        elif "contract" in input_lower and ("contractor" in input_lower or "contract work" in input_lower):
            return "contract", False
        elif "intern" in input_lower:
            return "internship", False
        else:
            return None, True  # Need clarification - must be explicit
    else:
        return cleaned_input, False


def generate_clarification_question(
    user_input: str,
    field_name: str,
    state: GraphState
) -> str:
    """Generate dynamic clarification question using LLM."""
    try:
        clarification_prompt = f"""The user said "{user_input}" when asked about {field_name} for a job description.

This response needs clarification. Generate a friendly, specific question that:
1. References what they said
2. Asks for the specific information needed
3. Gives a helpful example if appropriate

Keep it under 25 words and natural.

Examples:
- "You mentioned 'maybe remote' - would this be fully remote, hybrid, or office-based?"
- "I see you said 'some experience' - could you tell me how many years?"
- "You mentioned 'fronte development' - did you mean frontend development?"

Generate clarification question:"""

        if hasattr(st, 'session_state') and 'groq_client' in st.session_state:
            groq_client = st.session_state.groq_client
            
            completion = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": clarification_prompt}],
                model=settings.llm_model,
                max_tokens=50,
                temperature=0.3
            )
            
            return completion.choices[0].message.content.strip()
        else:
            return f"Could you clarify your response about {field_name.replace('_', ' ')}?"
            
    except Exception as e:
        logger.error(f"Error generating clarification: {e}")
        return f"Could you tell me more about the {field_name.replace('_', ' ')}?"


def needs_clarification_for_field(field_name: str, state: GraphState) -> bool:
    """Check if field needs clarification."""
    return field_name in state.get("clarification_needed", [])


def get_raw_response_for_field(field_name: str, state: GraphState) -> Optional[str]:
    """Get the raw user response for a specific field."""
    return state.get("raw_responses", {}).get(field_name)


def get_processed_value_for_field(field_name: str, state: GraphState) -> Any:
    """Get the processed value for a specific field."""
    return state.get("processed_data", {}).get(field_name)


def format_conversation_history(messages: list) -> str:
    """Format conversation history for LLM context."""
    if not messages:
        return "No previous conversation"
    
    formatted = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            formatted.append(f"AI: {content}")
        elif role == "user":
            formatted.append(f"User: {content}")
    
    return "\n".join(formatted)


# Helper functions for extraction (kept minimal)

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


def extract_technical_skills(text: str) -> list[str]:
    """Extract technical skills from text."""
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


def extract_soft_skills(text: str) -> list[str]:
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


def extract_programming_languages(text: str) -> list[str]:
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


def extract_frameworks_tools(text: str) -> list[str]:
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