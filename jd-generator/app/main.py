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
from core.services.llm_service import LLMService, EvaluationRequest
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
    
    if "llm_service" not in st.session_state:
        st.session_state.llm_service = LLMService()


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

    # FIRST: Use LLM evaluation to extract multiple fields from user response
    evaluate_and_update_state(user_input, st.session_state.conversation_state)

    # Show processing indicator
    with st.spinner("🤖 Thinking..."):
        try:
            # Create enhanced conversation context with UPDATED state
            job_data = st.session_state.conversation_state.get("job_data", {})
            collected_fields = list(job_data.keys())

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


def evaluate_and_update_state(user_input: str, state):
    """Use LLM evaluation to extract multiple fields from user response."""
    current_field = state.get("current_field")
    job_data = state.get("job_data", {})
    
    # Skip evaluation if we're in greeting phase
    if not current_field:
        return
    
    try:
        # Create evaluation request
        evaluation_request = EvaluationRequest(
            user_input=user_input,
            current_field=current_field,
            job_data=job_data,
            conversation_context=get_conversation_context(state)
        )
        
        # Perform LLM evaluation
        llm_service = st.session_state.llm_service
        evaluation_result = llm_service.evaluate_user_response(evaluation_request)
        
        logger.info(f"Evaluation result: {evaluation_result}")
        
        # Update state based on evaluation
        if evaluation_result.needs_clarification:
            # Set clarification flag for next response
            state["needs_clarification"] = True
            state["validation_errors"] = ["Response needs clarification"]
            logger.info("Response needs clarification")
        else:
            # Process extracted fields (even if there are validation issues)
            extracted_count = 0
            for field_name, field_value in evaluation_result.extracted_fields.items():
                if field_value is not None and field_value != "" and field_value != []:
                    state["job_data"][field_name] = field_value
                    logger.info(f"Extracted and stored {field_name}: {field_value}")
                    extracted_count += 1
            
            # Use corrected input if provided
            if evaluation_result.corrected_input:
                logger.info(f"Using corrected input: {evaluation_result.corrected_input}")
            
            # Log validation issues as warnings but don't block processing
            if evaluation_result.validation_issues:
                logger.info(f"Validation notes: {evaluation_result.validation_issues}")
            
            # Update current field based on what we've collected
            update_current_field_after_evaluation(state)
            
            logger.info(f"Successfully extracted {extracted_count} fields from user response")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        # Fallback to basic extraction for current field only
        extract_single_field(user_input, state, current_field)


def get_conversation_context(state) -> list:
    """Get conversation context for evaluation."""
    messages = state.get("messages", [])
    if len(messages) <= 2:
        return []
    
    # Get last few messages for context
    recent_messages = messages[-3:]
    context_messages = []
    for msg in recent_messages:
        context_messages.append({
            "role": msg.get("role", ""),
            "content": msg.get("content", "")[:200]  # Truncate for context
        })
    
    return context_messages


def update_current_field_after_evaluation(state):
    """Update current field based on evaluation results."""
    job_data = state.get("job_data", {})
    
    # Define field priority order
    field_priority = [
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
    
    # Find the next field that needs to be collected
    for field in field_priority:
        if not is_field_complete(field, job_data):
            state["current_field"] = field
            logger.info(f"Next field to collect: {field}")
            return
    
    # All fields complete
    state["current_field"] = None
    state["is_complete"] = True
    logger.info("All fields completed after evaluation!")


def extract_single_field(user_input: str, state, field_name: str):
    """Fallback single field extraction for critical fields."""
    logger.info(f"Using fallback extraction for field: {field_name}")
    
    if "job_data" not in state:
        state["job_data"] = {}
    
    input_text = user_input.strip()
    if not input_text:
        return
    
    # Simple fallback extraction for current field only
    if field_name == "background_intro":
        state["job_data"]["background_intro"] = input_text
    elif field_name == "job_title":
        cleaned_title = input_text.strip().title()
        if len(cleaned_title) >= 3:
            state["job_data"]["job_title"] = cleaned_title


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
    """Display the conversation history with enhanced formatting."""
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
        except:
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

        # Input area
        user_input = st.text_input(
            "Your message:", placeholder="Type your response here...", key="user_input"
        )

        col1_1, col1_2 = st.columns([1, 5])
        with col1_1:
            if st.button("🎤 Voice Input", use_container_width=True):
                st.info("Voice input feature coming soon...")
        with col1_2:
            if st.button("Send", type="primary", use_container_width=True):
                if user_input:
                    # Process the user message
                    process_user_message(user_input)
                    # Clear the input
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

        progress = st.progress(progress_value)
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