"""Main entry point for the JD Generator Streamlit application."""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.utils.logger import logger
from core.models.graph_state import create_initial_graph_state
from core.graph.nodes.greeting import greeting_node
from groq import Groq


def initialize_session_state():
    """Initialize session state for conversation."""
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = create_initial_graph_state()
        # Run greeting node to start conversation
        st.session_state.conversation_state = greeting_node(st.session_state.conversation_state)
    
    if "groq_client" not in st.session_state:
        try:
            st.session_state.groq_client = Groq(api_key=settings.get_llm_api_key())
        except ValueError:
            st.error("Groq API key not configured. Please check your .env file.")
            st.stop()


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
        "timestamp": "2025-08-30T22:00:00Z"
    }
    st.session_state.conversation_state["messages"].append(user_message)
    
    # Show processing indicator
    with st.spinner("🤖 Thinking..."):
        try:
            # Create enhanced conversation context
            job_data = st.session_state.conversation_state.get("job_data", {})
            collected_fields = list(job_data.keys())
            
            # Enhanced system prompt for employee interview perspective
            system_prompt = f"""You are an expert HR interviewer conducting a conversation with a job candidate to understand their background and preferences for creating a personalized job description.

Current conversation phase: {st.session_state.conversation_state.get('conversation_phase', 'greeting')}
Fields already collected: {collected_fields}

Guidelines:
- Be warm, professional, and conversational like a friendly HR interviewer
- If the candidate greets you or introduces themselves, acknowledge them by name warmly
- Ask ONE specific question at a time about their background, skills, and job preferences
- Build on their responses naturally, showing genuine interest
- Focus on understanding THEIR experience, skills, and what THEY want in a role
- Ask about their current role, skills they want to use, preferred work environment, etc.
- Keep responses concise but engaging
- Show enthusiasm about understanding their career goals

Focus on gathering from the candidate: their current/desired job_title, preferred department, their experience_level, their key_skills, responsibilities they enjoy, preferred location, desired salary_range, important benefits."""
            
            conversation_context = [{"role": "system", "content": system_prompt}]
            
            # Add recent conversation history (last 8 messages for context)
            recent_messages = st.session_state.conversation_state["messages"][-8:]
            for msg in recent_messages:
                conversation_context.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Make LLM call with enhanced prompt
            logger.info(f"=== GROQ LLM CALL START ===")
            logger.info(f"Model: {settings.llm_model}")
            logger.info(f"Message Count: {len(conversation_context)}")
            logger.info(f"Last User Message: {user_input}")
            logger.info(f"Temperature: 0.8, Max Tokens: 250")
            
            completion = st.session_state.groq_client.chat.completions.create(
                messages=conversation_context,
                model=settings.llm_model,
                max_tokens=250,
                temperature=0.8,  # More creative responses
                top_p=0.9
            )
            
            ai_response = completion.choices[0].message.content
            logger.info(f"=== GROQ LLM RESPONSE ===")
            logger.info(f"Full Response: {ai_response}")
            logger.info(f"Response Length: {len(ai_response)} characters")
            logger.info(f"=== GROQ LLM CALL END ===")
            
            # Also show in UI for debugging
            st.sidebar.write("**Last LLM Call:**")
            st.sidebar.write(f"Input: {user_input[:50]}...")
            st.sidebar.write(f"Response: {ai_response[:100]}...")
            st.sidebar.write(f"Length: {len(ai_response)} chars")
            
            # Extract any job data from the conversation
            extract_job_data(user_input, st.session_state.conversation_state)
            
            # Add AI response to state
            ai_message = {
                "role": "assistant",
                "content": ai_response,
                "message_type": "question",
                "timestamp": "2025-08-30T22:00:00Z"
            }
            st.session_state.conversation_state["messages"].append(ai_message)
            
            # Update conversation phase if needed
            update_conversation_phase(st.session_state.conversation_state)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            st.error(f"Sorry, I encountered an error: {e}")


def extract_job_data(user_input: str, state):
    """Extract job data from user input."""
    input_lower = user_input.lower()
    
    # Simple extraction logic
    if any(word in input_lower for word in ['engineer', 'developer', 'analyst', 'manager', 'designer']):
        if 'job_title' not in state['job_data']:
            state['job_data']['job_title'] = user_input
    
    if any(word in input_lower for word in ['engineering', 'marketing', 'sales', 'hr', 'finance']):
        if 'department' not in state['job_data']:
            state['job_data']['department'] = user_input
    
    if any(word in input_lower for word in ['years', 'experience', 'senior', 'junior', 'mid-level']):
        if 'experience' not in state['job_data']:
            state['job_data']['experience'] = user_input


def update_conversation_phase(state):
    """Update conversation phase based on collected data."""
    job_data = state.get('job_data', {})
    
    if len(job_data) == 0:
        state['conversation_phase'] = 'greeting'
    elif len(job_data) <= 3:
        state['conversation_phase'] = 'collecting_basic_info'
    elif len(job_data) <= 6:
        state['conversation_phase'] = 'collecting_details'
    else:
        state['conversation_phase'] = 'finalizing'


def display_conversation():
    """Display the conversation history with enhanced formatting."""
    messages = st.session_state.conversation_state.get("messages", [])
    
    if not messages:
        st.info("👋 Welcome! I'm ready to help you create a job description. Just say hello to get started!")
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
    
    # Show current status
    job_data = st.session_state.conversation_state.get("job_data", {})
    if job_data:
        st.info(f"✨ Collected so far: {', '.join(job_data.keys())}")


def main():
    """Main application function."""
    
    # Page configuration
    st.set_page_config(
        page_title="JD Generator Assistant",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Application header
    st.title("📝 AI-Powered Job Description Generator")
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
            phase = st.session_state.conversation_state.get("conversation_phase", "unknown")
            st.metric("Messages", msg_count)
            st.metric("Phase", phase)
            
            # Show collected data
            job_data = st.session_state.conversation_state.get("job_data", {})
            if job_data:
                st.write("**Collected Data:**")
                for key, value in job_data.items():
                    st.write(f"• {key}: {str(value)[:30]}...")
        
        if st.button("Clear Conversation"):
            st.session_state.clear()
            st.rerun()
            
        if st.button("🧪 Test LLM"):
            if st.session_state.get("groq_client"):
                with st.spinner("Testing..."):
                    try:
                        test_completion = st.session_state.groq_client.chat.completions.create(
                            messages=[{"role": "user", "content": "Say hello"}],
                            model=settings.llm_model,
                            max_tokens=50
                        )
                        st.success(f"✅ LLM Working: {test_completion.choices[0].message.content}")
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
            "Your message:",
            placeholder="Type your response here...",
            key="user_input"
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
        st.header("Progress Tracker")
        
        # Get current job data
        job_data = st.session_state.conversation_state.get("job_data", {})
        
        # Progress indicator
        total_fields = 12
        completed_fields = len([v for v in job_data.values() if v])
        progress_value = completed_fields / total_fields if total_fields > 0 else 0
        
        progress = st.progress(progress_value)
        st.write(f"Fields Completed: {completed_fields}/{total_fields}")
        
        # Field checklist
        with st.expander("Required Fields", expanded=True):
            field_mapping = {
                "Job Title": "job_title",
                "Department": "department", 
                "Employment Type": "employment_type",
                "Location": "location",
                "Experience": "experience",
                "Responsibilities": "responsibilities",
                "Technical Skills": "skills",
                "Soft Skills": "soft_skills",
                "Education": "education",
                "Salary Range": "salary",
                "Benefits": "benefits",
                "Additional Info": "additional_requirements"
            }
            
            for display_name, field_key in field_mapping.items():
                is_completed = field_key in job_data and job_data[field_key]
                st.checkbox(display_name, value=is_completed, disabled=True, key=f"field_{field_key}")
        
        # Export section
        st.header("Export Options")
        if st.button("Preview JD", use_container_width=True):
            st.info("JD preview will appear here once generated...")
        
        if st.button("Download JD", use_container_width=True, type="secondary"):
            st.info("Download feature coming soon...")
    
    # Footer
    st.markdown("---")
    st.caption("Powered by LangGraph and OpenAI")
    
    # Log application start
    logger.info(f"Application started in {settings.app_env} mode")


if __name__ == "__main__":
    main()