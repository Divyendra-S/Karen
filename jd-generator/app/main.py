"""Clean main entry point for JD Generator - UI orchestration only."""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.utils.logger import logger
from core.models.graph_state import create_initial_graph_state, add_message_to_state
from core.graph.builder import create_jd_graph
from groq import Groq


def initialize_session_state():
    """Initialize session state for conversation."""
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = create_initial_graph_state()
        logger.info("Initialized new conversation state")
    
    if "input_value" not in st.session_state:
        st.session_state.input_value = ""
    
    if "graph" not in st.session_state:
        st.session_state.graph = create_jd_graph()
        logger.info("Initialized graph")
    
    if "groq_client" not in st.session_state:
        try:
            st.session_state.groq_client = Groq(api_key=settings.get_llm_api_key())
            logger.info("Initialized Groq client")
        except ValueError:
            st.error("Groq API key not configured. Please check your .env file.")
            st.stop()


def process_user_message(user_input: str):
    """Process user message through the graph - clean orchestration only."""
    if not user_input.strip():
        return
    
    logger.info(f"User input received: '{user_input}'")
    
    try:
        # Add user message to state
        user_message = {
            "role": "user",
            "content": user_input,
            "message_type": "response",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.conversation_state = add_message_to_state(
            st.session_state.conversation_state,
            user_message
        )
        
        # Create a user response processing graph
        from core.graph.builder import create_user_response_graph
        response_graph = create_user_response_graph()
        
        # Run through response processing graph
        with st.spinner("🤖 Processing..."):
            result = response_graph.invoke(st.session_state.conversation_state)
            st.session_state.conversation_state = result
            
        logger.info("Graph processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        st.error(f"Sorry, I encountered an error: {e}")


def display_conversation():
    """Display the conversation history."""
    messages = st.session_state.conversation_state.get("messages", [])
    
    if not messages:
        # Generate initial greeting and first question
        try:
            with st.spinner("🤖 Starting conversation..."):
                result = st.session_state.graph.invoke(st.session_state.conversation_state)
                st.session_state.conversation_state = result
                messages = result.get("messages", [])
        except Exception as e:
            logger.error(f"Error generating initial conversation: {e}")
            st.info("👋 Hello! I'm here to learn about your professional background. Tell me about yourself!")
            return
    
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)
        elif role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**You:** {content}")
    
    # Show current status
    processed_data = st.session_state.conversation_state.get("processed_data", {})
    if processed_data:
        field_display_names = {
            "name": "Name",
            "job_title": "Current Role",
            "department": "Department", 
            "employment_type": "Employment Type",
            "location": "Location",
            "experience": "Experience",
            "responsibilities": "Responsibilities",
            "skills": "Skills",
            "education": "Education",
            "salary": "Salary Expectations",
            "benefits": "Benefits",
            "additional_requirements": "Additional Info"
        }
        
        collected_names = [
            field_display_names.get(key, key.title()) 
            for key in processed_data.keys()
        ]
        st.info(f"✨ Collected so far: {', '.join(collected_names)}")


def main():
    """Main application function - UI only."""
    
    # Page configuration
    st.set_page_config(
        page_title="Professional Profile Interview",
        page_icon="👤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Application header
    st.title("👤 Professional Profile Interview")
    st.markdown("*Tell me about your professional background and experience*")
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
        st.header("Interview Progress")
        if "conversation_state" in st.session_state:
            msg_count = len(st.session_state.conversation_state.get("messages", []))
            phase = st.session_state.conversation_state.get("conversation_phase", "starting")
            st.metric("Messages", msg_count)
            st.metric("Phase", phase.replace("_", " ").title())
            
            # Show collected information
            st.write("**Your Information:**")
            processed_data = st.session_state.conversation_state.get("processed_data", {})
            if processed_data:
                for key, value in processed_data.items():
                    display_name = key.replace('_', ' ').title()
                    display_value = str(value)[:30] + "..." if len(str(value)) > 30 else str(value)
                    st.write(f"• **{display_name}**: {display_value}")
            else:
                st.write("*Share your background to get started*")
        
        if st.button("Clear Conversation"):
            st.session_state.clear()
            st.rerun()
    
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
            placeholder="Tell me about your professional background...",
            key="user_input_field",
            value=st.session_state.input_value
        )
        
        col1_1, col1_2 = st.columns([1, 5])
        with col1_1:
            if st.button("🎤 Voice Input", use_container_width=True):
                st.info("Voice input feature coming soon...")
        with col1_2:
            send_clicked = st.button("Send", type="primary", use_container_width=True)
            
        # Handle send button click
        if send_clicked and user_input.strip():
            process_user_message(user_input)
            st.session_state.input_value = ""
            st.rerun()
        elif send_clicked:
            st.warning("Please enter a message before sending.")
    
    with col2:
        st.header("Your Profile")
        
        # Get current processed data
        if "conversation_state" in st.session_state:
            processed_data = st.session_state.conversation_state.get("processed_data", {})
        else:
            processed_data = {}
        
        # Progress indicator
        total_fields = 8  # Essential fields for a profile
        completed_fields = len([v for v in processed_data.values() if v and str(v).strip()])
        progress_value = completed_fields / total_fields if total_fields > 0 else 0
        
        progress_container = st.container()
        with progress_container:
            st.progress(progress_value)
            st.write(f"Profile Completeness: {completed_fields}/{total_fields}")
        
        # Profile summary
        with st.expander("Profile Summary", expanded=True):
            field_mapping = {
                "Name": "name",
                "Current Role": "job_title",
                "Department": "department", 
                "Work Type": "employment_type",
                "Location": "location",
                "Experience": "experience",
                "Key Skills": "skills",
                "Education": "education"
            }
            
            for display_name, field_key in field_mapping.items():
                is_completed = (
                    field_key in processed_data and 
                    processed_data[field_key] and 
                    str(processed_data[field_key]).strip()
                )
                if is_completed:
                    value = str(processed_data[field_key])
                    st.success(f"✅ **{display_name}**: {value}")
                else:
                    st.info(f"⏳ {display_name}")
        
        # Export section
        st.header("Export Profile")
        if completed_fields >= 3:
            if st.button("Generate Resume Summary", use_container_width=True):
                st.info("Resume generation will be added soon...")
        else:
            st.info("Complete more of your profile to enable export options")
    
    # Footer
    st.markdown("---")
    st.caption("AI-Powered Professional Profile Interview")


if __name__ == "__main__":
    main()