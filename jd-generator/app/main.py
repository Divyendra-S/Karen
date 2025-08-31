"""Main entry point for the JD Generator Streamlit application."""

import streamlit as st
from pathlib import Path
import sys
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.utils.logger import logger
from core.models.graph_state import create_initial_graph_state
from core.graph.nodes.greeting import greeting_node
from core.services.direct_speech_service import get_direct_speech_service
from audio_recorder_streamlit import audio_recorder
from groq import Groq
from typing import List, Dict, Any
import json


def initialize_session_state():
    """Initialize session state for conversation."""
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = create_initial_graph_state()
        # Run greeting node to start conversation
        st.session_state.conversation_state = greeting_node(
            st.session_state.conversation_state
        )

    if "groq_client" not in st.session_state:
        try:
            st.session_state.groq_client = Groq(api_key=settings.get_llm_api_key())
        except ValueError:
            st.error("Groq API key not configured. Please check your .env file.")
            st.stop()
    
    if "speech_service" not in st.session_state:
        st.session_state.speech_service = get_direct_speech_service()


def process_user_message(user_input: str):
    """Process user message and get AI response with enhanced conversation."""
    if not user_input.strip():
        return

    # Log the user input
    logger.info(f"User input received: {user_input}")

    # Add user message to state
    user_message = {
        "role": "user",
        "content": user_input,
        "message_type": "response",
        "timestamp": "2025-08-30T22:00:00Z",
    }
    st.session_state.conversation_state["messages"].append(user_message)

    # FIRST: Extract data and update state before AI response
    extract_job_data(user_input, st.session_state.conversation_state)
    add_field_validation_feedback(st.session_state.conversation_state)
    update_current_field_after_extraction(st.session_state.conversation_state)

    # Show processing indicator
    with st.spinner("🤖 Thinking..."):
        try:
            # Create enhanced conversation context with UPDATED state
            job_data = st.session_state.conversation_state.get("job_data", {})

            # System prompt for HR assistance
            system_prompt = f"""You're helping an HR professional create a detailed job description. Keep the conversation friendly but focused on gathering all the necessary information.

Current field: {st.session_state.conversation_state.get('current_field', 'job_title')}
Collected: {list(st.session_state.conversation_state.get('job_data', {}).keys())}

JD FIELDS TO COLLECT:
Job Title → Department → Years of Experience → Employment Type → Location → Key Responsibilities → Required Skills → Educational Qualification → Compensation Range → Additional Details

CONVERSATION STYLE:
- Max 2 sentences, keep it brief and professional
- Ask 1-2 questions at a time, not overwhelming
- Use friendly acknowledgments: "Got it!", "Perfect!", "Great!"
- Natural HR language: "What's the job title?" not "Please specify the job title"

EXAMPLES:
❌ "Please provide the employment type for this position"
✅ "Got it! Is this a full-time role or something else?"

❌ "What are the educational requirements for this position?"
✅ "Perfect! What education level should candidates have?"

Stay helpful and professional while gathering complete job details!"""

            conversation_context = [{"role": "system", "content": system_prompt}]

            # Add recent conversation history (last 8 messages for context)
            recent_messages = st.session_state.conversation_state["messages"][-8:]
            for msg in recent_messages:
                conversation_context.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

            # Make LLM call with enhanced prompt
            logger.info("=== GROQ LLM CALL START ===")
            logger.info(f"Model: {settings.llm_model}")
            logger.info(f"Message Count: {len(conversation_context)}")
            logger.info(f"Last User Message: {user_input}")
            logger.info("Temperature: 0.7, Max Tokens: 150")

            completion = st.session_state.groq_client.chat.completions.create(
                messages=conversation_context,
                model=settings.llm_model,
                max_tokens=150,  # Shorter responses for natural conversation
                temperature=0.7,  # Balanced creativity
                top_p=0.9,
            )

            ai_response = completion.choices[0].message.content
            logger.info("=== GROQ LLM RESPONSE ===")
            logger.info(f"Full Response: {ai_response}")
            logger.info(f"Response Length: {len(ai_response)} characters")
            logger.info("=== GROQ LLM CALL END ===")

            # Also show in UI for debugging
            st.sidebar.write("**Last LLM Call:**")
            st.sidebar.write(f"Input: {user_input[:50]}...")
            st.sidebar.write(f"Response: {ai_response[:100]}...")
            st.sidebar.write(f"Length: {len(ai_response)} chars")

            # Show extraction debug info
            current_field = st.session_state.conversation_state.get("current_field")
            job_data = st.session_state.conversation_state.get("job_data", {})
            st.sidebar.write(f"**Current Field:** {current_field}")
            st.sidebar.write(f"**Job Data Keys:** {list(job_data.keys())}")
            if current_field and current_field in job_data:
                st.sidebar.write(
                    f"**{current_field}:** {str(job_data[current_field])[:50]}..."
                )

            # Add AI response to state
            ai_message = {
                "role": "assistant",
                "content": ai_response,
                "message_type": "question",
                "timestamp": "2025-08-30T22:00:00Z",
            }
            st.session_state.conversation_state["messages"].append(ai_message)

            # Update conversation phase if needed
            update_conversation_phase(st.session_state.conversation_state)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            st.error(f"Sorry, I encountered an error: {e}")


def extract_job_data(user_input: str, state):
    """Extract structured job data using LLM intelligence with rule-based fallback."""
    current_field = state.get('current_field')
    input_text = user_input.strip()
    
    # Initialize job_data if it doesn't exist
    if 'job_data' not in state:
        state['job_data'] = {}
    
    if not current_field or not input_text:
        return
    
    logger.info(f"Extracting field '{current_field}' from input: '{input_text}'")
    
    # Try LLM extraction first
    try:
        extracted_data = extract_with_llm(input_text, current_field, st.session_state.groq_client)
        
        if extracted_data and extracted_data.get('confidence', 0) > 0.7:
            # Use LLM extracted data
            state['job_data'][current_field] = extracted_data
            logger.info(f"LLM extracted {current_field}: {extracted_data}")
            return
        else:
            logger.info(f"LLM extraction confidence too low, using fallback for {current_field}")
    
    except Exception as e:
        logger.error(f"LLM extraction failed for {current_field}: {e}")
    
    # Fallback to rule-based extraction
    logger.info(f"Using rule-based extraction for {current_field}")
    _extract_with_rules(input_text, current_field, state)


def _extract_with_rules(input_text: str, current_field: str, state):
    """Rule-based extraction fallback."""
    if current_field == 'job_title':
        cleaned_title = input_text.strip().title()
        if len(cleaned_title) >= 3 and not any(char in cleaned_title for char in ['@', '#', '$', '%', '&']):
            state['job_data']['job_title'] = {"title": cleaned_title, "confidence": 0.8}
            logger.info(f"Rule-based stored job_title: {cleaned_title}")
        else:
            logger.warning(f"Job title validation failed for: {cleaned_title}")
    
    elif current_field == 'department':
        cleaned_dept = input_text.strip().title()
        state['job_data']['department'] = {"department": cleaned_dept, "confidence": 0.8}
        logger.info(f"Rule-based stored department: {cleaned_dept}")
    
    elif current_field == 'employment_type':
        input_lower = input_text.lower()
        type_mapping = {
            'full-time': 'full_time', 'full': 'full_time', 'permanent': 'full_time',
            'part-time': 'part_time', 'part': 'part_time',
            'contract': 'contract', 'contractor': 'contract', 'consulting': 'contract',
            'internship': 'internship', 'intern': 'internship',
            'temp': 'temporary', 'temporary': 'temporary',
            'freelance': 'freelance', 'freelancer': 'freelance'
        }
        for key, value in type_mapping.items():
            if key in input_lower:
                state['job_data']['employment_type'] = {"type": value, "confidence": 0.8}
                logger.info(f"Rule-based stored employment_type: {value}")
                break
    
    elif current_field == 'location':
        location_data = parse_location_from_input(input_text)
        if location_data:
            location_data['confidence'] = 0.8
            state['job_data']['location'] = location_data
    
    elif current_field == 'experience':
        experience_data = parse_experience_from_input(input_text)
        if experience_data:
            experience_data['confidence'] = 0.8
            state['job_data']['experience'] = experience_data
    
    elif current_field == 'responsibilities':
        responsibilities = parse_list_from_input(input_text)
        if responsibilities:
            state['job_data']['responsibilities'] = {"responsibilities": responsibilities, "confidence": 0.8}
    
    elif current_field == 'skills':
        skills_data = parse_skills_from_input(input_text)
        if skills_data:
            skills_data['confidence'] = 0.8
            state['job_data']['skills'] = skills_data
    
    elif current_field == 'education':
        education_data = parse_education_from_input(input_text)
        if education_data:
            education_data['confidence'] = 0.8
            state['job_data']['education'] = education_data
    
    elif current_field == 'salary':
        salary_data = parse_salary_from_input(input_text)
        if salary_data:
            salary_data['confidence'] = 0.8
            state['job_data']['salary'] = salary_data
    
    elif current_field == 'additional_requirements':
        if not any(word in input_text.lower() for word in ['none', 'nothing', 'skip']):
            state['job_data']['additional_requirements'] = {"requirements": [input_text], "confidence": 0.8}


def parse_location_from_input(input_text: str) -> dict:
    """Parse location data from user input."""
    import re
    
    input_lower = input_text.lower()
    
    # Determine location type
    if any(word in input_lower for word in ["remote", "work from home", "wfh", "anywhere"]):
        location_type = "remote"
        city = None
        state_val = None
    elif any(word in input_lower for word in ["hybrid", "flexible", "mix"]):
        location_type = "hybrid"
        city, state_val = extract_city_state_from_text(input_text)
    else:
        location_type = "on_site"
        city, state_val = extract_city_state_from_text(input_text)
    
    return {
        "location_type": location_type,
        "city": city,
        "state": state_val,
        "country": "United States"
    }


def parse_experience_from_input(input_text: str) -> dict:
    """Parse experience data from user input."""
    import re
    
    input_lower = input_text.lower()
    
    # Handle button format ranges first (e.g., "3-5 years")
    range_match = re.search(r'(\d+)-(\d+)\s*years?', input_lower)
    if range_match:
        years_min = int(range_match.group(1))
        years_max = int(range_match.group(2))
    else:
        # Extract single years or years+
        years_match = re.search(r'(\d+)(?:\+)?\s*(?:years?|yrs?)', input_lower)
        years_min = int(years_match.group(1)) if years_match else 0
        years_max = years_min + 2 if years_min > 0 else None
    
    # Determine level based on years or keywords
    if years_min == 0 or any(word in input_lower for word in ["entry", "new grad", "0-2", "graduate"]):
        level = "entry"
    elif years_min <= 3 or any(word in input_lower for word in ["junior", "jr", "1-3"]):
        level = "junior"
    elif years_min >= 5 or any(word in input_lower for word in ["senior", "sr", "lead", "5+"]):
        level = "senior"
    elif years_min >= 10 or any(word in input_lower for word in ["principal", "staff", "architect", "10+"]):
        level = "principal"
    else:
        level = "mid"
    
    # Check leadership
    leadership_required = any(word in input_lower for word in ["leadership", "lead", "manage", "supervise"])
    
    # Extract industry
    industries = ["fintech", "healthcare", "education", "e-commerce", "gaming", "automotive", "aerospace"]
    industry_experience = ", ".join([ind for ind in industries if ind in input_lower])
    
    return {
        "level": level,
        "years_min": years_min,
        "years_max": years_max,
        "industry_experience": industry_experience or None,
        "leadership_required": leadership_required
    }


def parse_skills_from_input(input_text: str) -> dict:
    """Parse skills data from user input."""
    import re
    
    # Split by categories or commas
    skills_data = {
        "technical_skills": [],
        "soft_skills": [],
        "programming_languages": [],
        "frameworks_tools": [],
        "certifications": []
    }
    
    # Common skills for categorization
    programming_languages = ["python", "javascript", "java", "c++", "c#", "go", "rust", "typescript", "php", "ruby", "sql"]
    frameworks_tools = ["react", "angular", "vue", "django", "flask", "node.js", "spring", "aws", "azure", "docker", "kubernetes"]
    soft_skills = ["communication", "leadership", "teamwork", "problem solving", "analytical", "creative", "time management"]
    
    # Parse by comma-separated items
    items = [item.strip() for item in re.split(r'[,\n•\-\*]', input_text) if item.strip()]
    
    for item in items:
        item_lower = item.lower()
        
        # Categorize skills
        if any(lang in item_lower for lang in programming_languages):
            skills_data["programming_languages"].append(item)
        elif any(tool in item_lower for tool in frameworks_tools):
            skills_data["frameworks_tools"].append(item)
        elif any(soft in item_lower for soft in soft_skills):
            skills_data["soft_skills"].append(item)
        elif any(cert_word in item_lower for cert_word in ["certified", "certification", "aws", "google", "microsoft"]):
            skills_data["certifications"].append(item)
        else:
            skills_data["technical_skills"].append(item)
    
    return skills_data


def parse_education_from_input(input_text: str) -> dict:
    """Parse education data from user input."""
    input_lower = input_text.lower()
    
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
    else:
        level = "bachelor"  # Default
    
    # Extract field of study
    common_fields = ["computer science", "engineering", "business", "marketing", "finance", "mathematics"]
    field_of_study = None
    for field in common_fields:
        if field in input_lower:
            field_of_study = field.title()
            break
    
    return {
        "level": level,
        "field_of_study": field_of_study,
        "is_required": True
    }


def parse_salary_from_input(input_text: str) -> dict:
    """Parse salary data from user input."""
    import re
    
    if any(word in input_text.lower() for word in ["skip", "not specify", "competitive", "negotiable"]):
        return {"type": "competitive", "is_negotiable": True}
    
    # Handle button format ranges (e.g., "$50k-80k")
    range_pattern = r'\$?(\d+)k?-(\d+)k'
    range_match = re.search(range_pattern, input_text.lower())
    if range_match:
        min_val = int(range_match.group(1))
        max_val = int(range_match.group(2))
        # Convert k to thousands if needed
        if min_val < 1000:
            min_val *= 1000
        if max_val < 1000:
            max_val *= 1000
        amounts = [min_val, max_val]
    else:
        # Extract salary numbers (existing logic)
        salary_pattern = r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        matches = re.findall(salary_pattern, input_text.replace(',', ''))
        if not matches:
            return None
        amounts = [float(match.replace(',', '')) for match in matches]
    
    # Determine frequency
    frequency = "annual"
    if any(word in input_text.lower() for word in ["hour", "hourly", "/hr"]):
        frequency = "hourly"
    elif any(word in input_text.lower() for word in ["month", "monthly", "/mo"]):
        frequency = "monthly"
    
    return {
        "min_salary": min(amounts) if amounts else None,
        "max_salary": max(amounts) if len(amounts) > 1 else None,
        "currency": "USD",
        "frequency": frequency,
        "is_negotiable": "negotiable" in input_text.lower()
    }


def parse_list_from_input(input_text: str) -> list:
    """Parse a list of items from user input."""
    import re
    
    # Split by common separators
    items = re.split(r'[,\n•\-\*;]', input_text)
    
    # Clean and filter items
    cleaned_items = []
    for item in items:
        cleaned = re.sub(r'^\d+[\.\)]\s*', '', item.strip())
        cleaned = cleaned.strip('•-* ')
        if len(cleaned) > 2:
            cleaned_items.append(cleaned)
    
    return cleaned_items


def extract_city_state_from_text(text: str) -> tuple:
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


def add_field_validation_feedback(state):
    """Add validation feedback to help user provide better data."""
    current_field = state.get('current_field')
    job_data = state.get('job_data', {})
    
    if not current_field:
        return
    
    # Clear previous validation feedback
    state['validation_feedback'] = []
    
    # Check if current field needs validation feedback
    if current_field == 'job_title':
        if current_field in job_data:
            title = job_data[current_field]
            if len(title) < 3:
                state.setdefault('validation_feedback', []).append(
                    f"Job title seems too short. Please provide a more complete title."
                )
    
    elif current_field == 'location':
        if current_field in job_data:
            location_data = job_data[current_field]
            if location_data.get('location_type') == 'on_site' and not location_data.get('city'):
                state.setdefault('validation_feedback', []).append(
                    f"For on-site work, please specify the city and state you prefer."
                )


def update_current_field_after_extraction(state):
    """Update current field to next field after successful data extraction."""
    current_field = state.get('current_field')
    job_data = state.get('job_data', {})
    
    # Define field order
    field_order = [
        'job_title', 
        'department',
        'experience',
        'employment_type',
        'location',
        'responsibilities',
        'skills',
        'education',
        'salary',
        'additional_requirements'
    ]
    
    # Find current field index
    try:
        current_index = field_order.index(current_field)
    except ValueError:
        logger.warning(f"Unknown field in update_current_field: {current_field}")
        return
    
    # Check if current field is complete and move to next
    if is_field_complete(current_field, job_data):
        # Move to next field
        next_index = current_index + 1
        if next_index < len(field_order):
            next_field = field_order[next_index]
            state['current_field'] = next_field
            logger.info(f"Moving to next field: {next_field}")
        else:
            # All fields completed
            state['current_field'] = None
            state['is_complete'] = True
            logger.info("All fields completed!")
    else:
        # Stay on current field if not complete
        logger.info(f"Staying on field {current_field} - not complete yet")




def is_field_complete(field_name: str, job_data: dict) -> bool:
    """Check if a specific field has been successfully collected with valid data."""
    if field_name not in job_data:
        return False

    field_value = job_data[field_name]

    # Handle both LLM-extracted (dict with nested data) and rule-based (simple) data
    if isinstance(field_value, dict):
        # LLM-extracted data structure
        if field_name == "job_title":
            return bool(field_value.get("title"))
        elif field_name == "department":
            return bool(field_value.get("department"))
        elif field_name == "employment_type":
            return bool(field_value.get("type"))
        elif field_name == "location":
            return bool(field_value.get("type"))
        elif field_name == "experience":
            return bool(field_value.get("level") and "years_min" in field_value)
        elif field_name == "responsibilities":
            return bool(field_value.get("responsibilities"))
        elif field_name == "skills":
            skills_data = field_value
            return any(skills for key, skills in skills_data.items() 
                      if key != "confidence" and skills)
        elif field_name == "education":
            return bool(field_value.get("level"))
        elif field_name == "salary":
            return True  # Optional field
        elif field_name == "additional_requirements":
            return True  # Optional field
    else:
        # Legacy simple data structure (backward compatibility)
        if field_name in ["job_title", "department", "employment_type"]:
            return bool(field_value and len(str(field_value).strip()) > 0)
        elif field_name == "location":
            return (
                isinstance(field_value, dict)
                and field_value.get("location_type")
                and (
                    field_value.get("location_type") == "remote" or field_value.get("city")
                )
            )
        elif field_name == "experience":
            return (
                isinstance(field_value, dict)
                and field_value.get("level")
                and "years_min" in field_value
            )
        elif field_name == "responsibilities":
            return isinstance(field_value, list) and len(field_value) > 0
        elif field_name == "skills":
            if isinstance(field_value, dict):
                return any(skills for skills in field_value.values() if skills)
            return bool(field_value)
        elif field_name == "education":
            return isinstance(field_value, dict) and field_value.get("level")
        else:
            return bool(field_value)

    return bool(field_value)


def update_conversation_phase(state):
    """Update conversation phase based on collected data."""
    job_data = state.get("job_data", {})

    if len(job_data) == 0:
        state["conversation_phase"] = "greeting"
    elif len(job_data) <= 3:
        state["conversation_phase"] = "collecting_basic_info"
    elif len(job_data) <= 6:
        state["conversation_phase"] = "collecting_details"
    else:
        state["conversation_phase"] = "finalizing"


def generate_field_options(field_name: str) -> List[str]:
    """Generate 4 quick options for the current field being collected."""
    options_map = {
        'job_title': ["Software Engineer", "Product Manager", "Data Analyst", "Marketing Manager"],
        'department': ["Engineering", "Product", "Marketing", "Sales"],
        'experience': ["0-2 years", "3-5 years", "5-10 years", "10+ years"],
        'employment_type': ["Full-time", "Part-time", "Contract", "Internship"], 
        'location': ["Remote", "Hybrid", "On-site", "Flexible"],
        'responsibilities': ["Technical Development", "Team Management", "Strategy & Planning", "Customer Relations"],
        'skills': ["Python/Programming", "Leadership/Management", "Data Analysis", "Communication"],
        'education': ["Bachelor's Degree", "Master's Degree", "PhD/Doctorate", "High School/Diploma"],
        'salary': ["$50k-80k", "$80k-120k", "$120k-180k", "Competitive/Negotiable"],
        'additional_requirements': ["Travel Required", "Security Clearance", "Flexible Hours", "None"]
    }
    
    return options_map.get(field_name, ["Option 1", "Option 2", "Option 3", "Other"])


# LLM Extraction Prompts for each field
EXTRACTION_PROMPTS = {
    'job_title': """Extract the job title from this HR input: "{input}"
Return JSON: {{"title": "cleaned job title", "level": "junior/mid/senior/principal", "confidence": 0.8}}""",
    
    'department': """Extract department from this HR input: "{input}"
Return JSON: {{"department": "department name", "category": "engineering/product/marketing/sales/hr/finance/operations", "confidence": 0.8}}""",
    
    'experience': """Extract experience requirements from this HR input: "{input}"
Return JSON: {{"years_min": number, "years_max": number, "level": "entry/junior/mid/senior/principal", "leadership_required": true/false, "confidence": 0.8}}""",
    
    'employment_type': """Extract employment type from this HR input: "{input}"
Return JSON: {{"type": "full_time/part_time/contract/internship/temporary/freelance", "confidence": 0.8}}""",
    
    'location': """Extract location details from this HR input: "{input}"
Return JSON: {{"type": "remote/hybrid/on_site", "city": "city name or null", "state": "state name or null", "country": "United States", "flexibility": true/false, "confidence": 0.8}}""",
    
    'responsibilities': """Extract key responsibilities list from this HR input: "{input}"
Return JSON: {{"responsibilities": ["responsibility 1", "responsibility 2", "responsibility 3"], "category": "technical/managerial/strategic/operational", "confidence": 0.8}}""",
    
    'skills': """Extract skills from this HR input: "{input}"
Return JSON: {{"technical_skills": [], "soft_skills": [], "programming_languages": [], "frameworks_tools": [], "certifications": [], "confidence": 0.8}}""",
    
    'education': """Extract education requirements from this HR input: "{input}"
Return JSON: {{"level": "high_school/associate/bachelor/master/doctorate", "field_of_study": "field name or null", "is_required": true/false, "confidence": 0.8}}""",
    
    'salary': """Extract salary information from this HR input: "{input}"
Return JSON: {{"min_salary": number_or_null, "max_salary": number_or_null, "currency": "USD", "frequency": "annual/monthly/hourly", "is_negotiable": true/false, "type": "fixed/competitive", "confidence": 0.8}}""",
    
    'additional_requirements': """Extract additional job requirements from this HR input: "{input}"
Return JSON: {{"requirements": ["req1", "req2"], "travel_required": true/false, "clearance_needed": true/false, "other_notes": "text or null", "confidence": 0.8}}"""
}


def extract_with_llm(user_input: str, field_name: str, groq_client) -> Dict[str, Any]:
    """Use LLM to intelligently extract structured data from user input."""
    if field_name not in EXTRACTION_PROMPTS:
        return None
    
    prompt = EXTRACTION_PROMPTS[field_name].format(input=user_input)
    
    extraction_request = [
        {"role": "system", "content": "You are a data extraction expert for HR job descriptions. Extract information and return ONLY valid JSON with the exact structure requested. If unsure, set confidence lower."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        completion = groq_client.chat.completions.create(
            messages=extraction_request,
            model=settings.llm_model,
            max_tokens=300,
            temperature=0.1,  # Low temperature for consistent extraction
        )
        
        response_text = completion.choices[0].message.content.strip()
        logger.info(f"LLM extraction response for {field_name}: {response_text}")
        
        # Extract JSON from response (handle cases where LLM adds extra text)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            extracted = json.loads(json_text)
            return extracted
        else:
            logger.warning(f"No valid JSON found in LLM response: {response_text}")
            return None
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in LLM extraction: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM extraction error: {e}")
        return None


def generate_professional_job_description(job_data: Dict[str, Any], groq_client) -> str:
    """Generate a professional job description using LLM based on collected data."""
    
    # Create structured prompt for JD generation
    jd_prompt = f"""Create a professional job description based on this information:

JOB DATA:
{json.dumps(job_data, indent=2)}

FORMAT: Generate a complete, professional job description following standard HR format:

**[JOB TITLE]**
[Company Name - leave placeholder]

**About the Role:**
[2-3 sentences describing the position and its impact]

**Key Responsibilities:**
• [Responsibility 1]
• [Responsibility 2] 
• [Responsibility 3]
[Add more as needed from data]

**Required Qualifications:**
• [Education requirement]
• [Experience requirement]
• [Required skills]
[Add more from data]

**Preferred Qualifications:**
• [Preferred skills and experience]
• [Nice-to-have qualifications]

**What We Offer:**
• [Compensation range if provided]
• [Benefits if provided]
• [Work arrangement details]

**Application Instructions:**
[Standard application process text]

GUIDELINES:
- Use professional, engaging language
- Be specific about requirements
- Include all collected information naturally
- Make it compelling for candidates
- Follow standard job posting format
- Use bullet points for readability
- Keep sections well-organized

Generate a complete, ready-to-post job description now:"""

    try:
        jd_generation_request = [
            {"role": "system", "content": "You are an expert HR professional who writes compelling, professional job descriptions. Create a complete, well-formatted job posting that attracts qualified candidates."},
            {"role": "user", "content": jd_prompt}
        ]
        
        completion = groq_client.chat.completions.create(
            messages=jd_generation_request,
            model=settings.llm_model,
            max_tokens=1500,  # Longer for complete JD
            temperature=0.3,  # Slightly creative but professional
        )
        
        generated_jd = completion.choices[0].message.content.strip()
        logger.info(f"Generated JD length: {len(generated_jd)} characters")
        return generated_jd
        
    except Exception as e:
        logger.error(f"JD generation error: {e}")
        return f"Error generating job description: {e}"


def check_if_interview_complete(job_data: Dict[str, Any]) -> bool:
    """Check if enough information has been collected to generate a JD."""
    required_fields = ['job_title', 'department', 'experience', 'employment_type', 'location']
    
    for field in required_fields:
        if not is_field_complete(field, job_data):
            return False
    
    return True


def display_conversation():
    """Display the conversation history."""
    messages = st.session_state.conversation_state.get("messages", [])

    if not messages:
        st.info(
            "👋 Welcome! I'm ready to interview you about your career goals. Just say hello to get started!"
        )
        return

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
        elif role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**You:** {content}")

    # Show current status using proper completion logic
    job_data = st.session_state.conversation_state.get("job_data", {})
    if job_data:
        field_names = [
            "background_intro",
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
        completed_field_names = [
            field for field in field_names if is_field_complete(field, job_data)
        ]
        if completed_field_names:
            st.info(f"✨ Information gathered: {', '.join(completed_field_names)}")


def main():
    """Main application function."""

    # Page configuration
    st.set_page_config(
        page_title="Career Interview Assistant",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded",
    )


    # Initialize session state
    initialize_session_state()

    # Application header
    st.title("💼 Career Interview Assistant")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.info(f"Environment: {settings.app_env}")
        st.info(f"Model: {settings.llm_model}")

        # API Status
        try:
            api_key = settings.get_llm_api_key()
            st.success(f"✅ API Key: {api_key[:10]}...")
        except Exception:
            st.error("❌ API Key not configured")

        # Conversation Status
        st.header("Conversation Status")
        if "conversation_state" in st.session_state:
            msg_count = len(st.session_state.conversation_state.get("messages", []))
            phase = st.session_state.conversation_state.get(
                "conversation_phase", "unknown"
            )
            st.metric("Messages", msg_count)
            st.metric("Phase", phase)

            # Show gathered information
            job_data = st.session_state.conversation_state.get("job_data", {})
            if job_data:
                st.write("**Your Information:**")
                for key, value in job_data.items():
                    st.write(f"• {key}: {str(value)[:30]}...")

        if st.button("Clear Conversation"):
            st.session_state.clear()
            st.rerun()

        if st.button("🧪 Test LLM"):
            if st.session_state.get("groq_client"):
                with st.spinner("Testing..."):
                    try:
                        test_completion = (
                            st.session_state.groq_client.chat.completions.create(
                                messages=[{"role": "user", "content": "Say hello"}],
                                model=settings.llm_model,
                                max_tokens=50,
                            )
                        )
                        st.success(
                            f"✅ LLM Working: {test_completion.choices[0].message.content}"
                        )
                    except Exception as e:
                        st.error(f"❌ LLM Error: {e}")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Chat Interface")

        # Display conversation history
        chat_container = st.container()
        with chat_container:
            display_conversation()

        # Quick Options Section
        current_field = st.session_state.conversation_state.get('current_field')
        if current_field and not st.session_state.conversation_state.get('is_complete'):
            st.markdown("**Quick Options:** *(click one or type your own below)*")
            options = generate_field_options(current_field)
            
            col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
            option_cols = [col_opt1, col_opt2, col_opt3, col_opt4]
            
            for i, option in enumerate(options):
                with option_cols[i]:
                    if st.button(option, key=f"quick_option_{current_field}_{i}", use_container_width=True):
                        process_user_message(option)
                        st.rerun()

        # Input area with inline voice recording
        col_input, col_voice, col_send = st.columns([6, 1, 1])
        
        with col_input:
            user_input = st.text_input(
                "Your message:", 
                placeholder="Type your response here...", 
                key="user_input",
                label_visibility="collapsed"
            )
        
        with col_voice:
            # Direct inline voice recorder
            audio_bytes = audio_recorder(
                text="🎤",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="1x",
                key="inline_voice_recorder"
            )
            
        with col_send:
            send_button = st.button("Send", type="primary", use_container_width=True)
        
        # Handle voice input with session state to prevent loops
        if audio_bytes:
            # Check if this is a new recording
            last_audio_hash = st.session_state.get("last_audio_hash")
            current_audio_hash = hash(audio_bytes)
            
            if last_audio_hash != current_audio_hash:
                st.session_state.last_audio_hash = current_audio_hash
                
                with st.spinner("🔄 Processing voice..."):
                    try:
                        speech_service = st.session_state.speech_service
                        audio_dict = {
                            'bytes': audio_bytes,
                            'sample_rate': 44100,
                            'format': 'wav'
                        }
                        
                        transcribed_text = speech_service.transcribe_audio_dict(audio_dict)
                        
                        if transcribed_text:
                            # Store in session state instead of immediate processing
                            st.session_state.pending_voice_message = transcribed_text
                            st.rerun()
                        else:
                            st.warning("Could not transcribe audio. Please try again.")
                            
                    except Exception as e:
                        st.error(f"Voice processing error: {e}")
        
        # Process pending voice message
        if st.session_state.get("pending_voice_message"):
            voice_message = st.session_state.pending_voice_message
            del st.session_state.pending_voice_message
            process_user_message(voice_message)
            st.rerun()
        
        # Handle text input
        elif send_button and user_input:
            process_user_message(user_input)
            st.rerun()

    with col2:
        st.header("Interview Progress")

        # Get current job data
        job_data = st.session_state.conversation_state.get("job_data", {})

        # Progress indicator using proper field completion logic
        total_fields = 10
        field_names = [
            "job_title",
            "department",
            "experience",
            "employment_type",
            "location",
            "responsibilities",
            "skills",
            "education",
            "salary",
            "additional_requirements",
        ]
        completed_fields = len(
            [field for field in field_names if is_field_complete(field, job_data)]
        )
        progress_value = completed_fields / total_fields if total_fields > 0 else 0

        st.progress(progress_value)
        st.write(f"Topics Covered: {completed_fields}/{total_fields}")

        # Field checklist
        with st.expander("Job Description Fields", expanded=True):
            field_mapping = {
                "Job Title": "job_title",
                "Department": "department", 
                "Years of Experience Required": "experience",
                "Employment Type": "employment_type",
                "Location & Work Mode": "location",
                "Key Responsibilities": "responsibilities",
                "Technical Skills": "skills",
                "Educational Qualification": "education",
                "Compensation Range": "salary",
                "Additional Requirements": "additional_requirements",
            }

            for display_name, field_key in field_mapping.items():
                is_completed = is_field_complete(field_key, job_data)
                st.checkbox(
                    display_name,
                    value=is_completed,
                    disabled=True,
                    key=f"field_{field_key}",
                )

        # Export section
        st.header("Your Job Description")
        
        # Check if interview is complete enough to generate JD
        if check_if_interview_complete(job_data):
            if st.button("Generate Professional JD", use_container_width=True, type="primary"):
                with st.spinner("🚀 Generating professional job description..."):
                    generated_jd = generate_professional_job_description(job_data, st.session_state.groq_client)
                    st.session_state.generated_jd = generated_jd
                    st.rerun()
            
            # Show generated JD if available
            if st.session_state.get("generated_jd"):
                st.subheader("Generated Job Description")
                st.markdown(st.session_state.generated_jd)
                
                # Download button for generated JD
                if st.download_button(
                    label="📄 Download JD as Text",
                    data=st.session_state.generated_jd,
                    file_name=f"job_description_{job_data.get('job_title', {}).get('title', 'position').lower().replace(' ', '_')}.txt",
                    mime="text/plain",
                    use_container_width=True
                ):
                    st.success("Job description downloaded!")
        else:
            st.info("Complete the interview to generate your professional job description...")
            missing_fields = []
            required_fields = ['job_title', 'department', 'experience', 'employment_type', 'location']
            for field in required_fields:
                if not is_field_complete(field, job_data):
                    missing_fields.append(field.replace('_', ' ').title())
            if missing_fields:
                st.caption(f"Still need: {', '.join(missing_fields)}")

    # Footer
    st.markdown("---")
    st.caption("Powered by LangGraph and Groq")

    # Log application start
    logger.info(f"Application started in {settings.app_env} mode")


if __name__ == "__main__":
    main()