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

            # System prompt for humanized interview conversation
            system_prompt = f"""You are a friendly, empathetic HR interviewer having a genuine conversation with someone about their career dreams. Talk like a real person who cares about their success.

Current conversation phase: {st.session_state.conversation_state.get('conversation_phase', 'greeting')}

FIELDS TO COLLECT (in order):
background_intro → job_title → department → employment_type → location → experience → responsibilities → skills → education → salary → benefits → additional_requirements

CURRENT STATUS:
- Target field: {st.session_state.conversation_state.get('current_field', 'background_intro')}
- Already collected: {list(st.session_state.conversation_state.get('job_data', {}).keys())}

HUMAN CONVERSATION STYLE:
- Talk like you're having coffee with a friend who's job hunting
- Use casual, warm language with personality
- Show you're genuinely excited about their career journey
- React naturally to what they share - be surprised, impressed, curious
- Use human expressions: "Oh nice!", "I love that!", "Wow, that's awesome!"
- Ask follow-ups like a real person would: "That sounds really cool!", "I bet that's challenging!"

RESPONSE TONE:
- Sound genuinely interested and enthusiastic
- Use contractions (I'm, you're, that's, it's)
- Add personality with natural reactions
- Be supportive and encouraging like a good friend
- Keep it brief but warm (1-2 sentences max)

EXAMPLES OF HUMANIZED RESPONSES:
❌ "Thank you for providing that information. I will now ask about the next field."
✅ "Oh nice! ML Engineer - that's such a hot field right now! So what kind of team are you hoping to join?"

❌ "Please specify your employment preferences."
✅ "Love it! Are you thinking full-time or are you open to contract work too?"

❌ "What are your location requirements?"
✅ "Sweet! Where do you want to work - are you a remote person or do you like being in the office?"

Be genuine, enthusiastic, and conversational. React to what they tell you like a real person would!"""

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
    """Extract structured job data from user input based on current field."""
    current_field = state.get('current_field')
    input_text = user_input.strip()
    
    # Initialize job_data if it doesn't exist
    if 'job_data' not in state:
        state['job_data'] = {}
    
    if not current_field or not input_text:
        return
    
    # Log what we're extracting
    logger.info(f"Extracting field '{current_field}' from input: '{input_text}'")
    
    # Extract data based on the specific field being collected
    if current_field == 'background_intro':
        state['job_data']['background_intro'] = input_text
        logger.info(f"Stored background_intro: {input_text}")
    
    elif current_field == 'job_title':
        # Clean and validate job title
        cleaned_title = input_text.strip().title()
        if len(cleaned_title) >= 3 and not any(char in cleaned_title for char in ['@', '#', '$', '%', '&']):
            state['job_data']['job_title'] = cleaned_title
            logger.info(f"Stored job_title: {cleaned_title}")
        else:
            logger.warning(f"Job title validation failed for: {cleaned_title}")
    
    elif current_field == 'department':
        cleaned_dept = input_text.strip().title()
        state['job_data']['department'] = cleaned_dept
        logger.info(f"Stored department: {cleaned_dept}")
    
    elif current_field == 'employment_type':
        # Map to standard employment types
        input_lower = input_text.lower()
        type_mapping = {
            'full': 'full_time', 'permanent': 'full_time', 'full-time': 'full_time',
            'part': 'part_time', 'part-time': 'part_time',
            'contract': 'contract', 'contractor': 'contract', 'consulting': 'contract',
            'intern': 'internship', 'internship': 'internship',
            'temp': 'temporary', 'temporary': 'temporary',
            'freelance': 'freelance', 'freelancer': 'freelance'
        }
        for key, value in type_mapping.items():
            if key in input_lower:
                state['job_data']['employment_type'] = value
                logger.info(f"Stored employment_type: {value}")
                break
    
    elif current_field == 'location':
        # Parse location data
        location_data = parse_location_from_input(input_text)
        if location_data:
            state['job_data']['location'] = location_data
    
    elif current_field == 'experience':
        # Parse experience data
        experience_data = parse_experience_from_input(input_text)
        if experience_data:
            state['job_data']['experience'] = experience_data
    
    elif current_field == 'responsibilities':
        # Parse responsibilities list
        responsibilities = parse_list_from_input(input_text)
        if responsibilities:
            state['job_data']['responsibilities'] = responsibilities
    
    elif current_field == 'skills':
        # Parse skills object
        skills_data = parse_skills_from_input(input_text)
        if skills_data:
            state['job_data']['skills'] = skills_data
    
    elif current_field == 'education':
        # Parse education data
        education_data = parse_education_from_input(input_text)
        if education_data:
            state['job_data']['education'] = education_data
    
    elif current_field == 'salary':
        # Parse salary data
        salary_data = parse_salary_from_input(input_text)
        if salary_data:
            state['job_data']['salary'] = salary_data
    
    elif current_field == 'benefits':
        # Parse benefits list
        benefits = parse_list_from_input(input_text)
        if benefits:
            state['job_data']['benefits'] = benefits
    
    elif current_field == 'additional_requirements':
        if not any(word in input_text.lower() for word in ['none', 'nothing', 'skip']):
            state['job_data']['additional_requirements'] = input_text


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
    
    # Extract years
    years_match = re.search(r'(\d+)(?:\+)?\s*(?:years?|yrs?)', input_lower)
    years_min = int(years_match.group(1)) if years_match else 0
    
    # Determine level
    if any(word in input_lower for word in ["entry", "new grad", "0-2", "graduate"]):
        level = "entry"
    elif any(word in input_lower for word in ["junior", "jr", "1-3"]):
        level = "junior"
    elif any(word in input_lower for word in ["senior", "sr", "lead", "5+"]):
        level = "senior"
    elif any(word in input_lower for word in ["principal", "staff", "architect", "10+"]):
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
    
    if any(word in input_text.lower() for word in ["skip", "not specify", "competitive"]):
        return None
    
    # Extract salary numbers
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
        'background_intro',
        'job_title', 
        'department',
        'employment_type',
        'location',
        'experience',
        'responsibilities',
        'skills',
        'education',
        'salary',
        'benefits',
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

    # Check field-specific completion criteria
    if field_name in ["background_intro", "job_title", "department", "employment_type"]:
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
            # At least one skill category should have content
            return any(skills for skills in field_value.values() if skills)
        return bool(field_value)

    elif field_name == "education":
        return isinstance(field_value, dict) and field_value.get("level")

    elif field_name == "salary":
        # Salary is optional, so None or empty is considered complete
        return True  # Always complete regardless of content

    elif field_name == "benefits":
        # Benefits is optional, so empty list is considered complete
        return True  # Always complete regardless of content

    elif field_name == "additional_requirements":
        # Additional requirements is optional
        return True  # Always complete regardless of content

    else:
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
        total_fields = 12
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
        completed_fields = len(
            [field for field in field_names if is_field_complete(field, job_data)]
        )
        progress_value = completed_fields / total_fields if total_fields > 0 else 0

        st.progress(progress_value)
        st.write(f"Topics Covered: {completed_fields}/{total_fields}")

        # Field checklist
        with st.expander("Interview Topics", expanded=True):
            field_mapping = {
                "Background": "background_intro",
                "Ideal Job Title": "job_title",
                "Preferred Department": "department",
                "Work Arrangement": "employment_type",
                "Location Preference": "location",
                "Your Experience": "experience",
                "Preferred Responsibilities": "responsibilities",
                "Your Technical Skills": "skills",
                "Your Education": "education",
                "Salary Expectations": "salary",
                "Important Benefits": "benefits",
                "Other Preferences": "additional_requirements",
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
        if st.button("Preview Your JD", use_container_width=True):
            st.info(
                "Your personalized job description will appear here once the interview is complete..."
            )

        if st.button("Download Your JD", use_container_width=True, type="secondary"):
            st.info("Download feature coming soon...")

    # Footer
    st.markdown("---")
    st.caption("Powered by LangGraph and Groq")

    # Log application start
    logger.info(f"Application started in {settings.app_env} mode")


if __name__ == "__main__":
    main()