"""Test suite for functional voice integration.

Tests the pure functional components of the voice pipeline integration
following functional programming principles.
"""

import pytest
import time
from typing import Dict, Any
from dataclasses import replace

from core.types.voice_types import (
    AudioFrame,
    TranscriptFrame,
    ResponseFrame,
    VoiceConfig,
    VoiceModel,
    ConversationState,
    ConversationMode,
    VoicePipelineResult,
)
from core.services.voice_events import (
    VoiceEvent,
    VoiceEventType,
    create_voice_state_reducer,
    create_conversation_state_initial,
    extract_conversation_commands,
    apply_conversation_commands,
)
from core.services.voice_integration import (
    create_voice_state_bridge,
    create_field_extractor_bridge,
    extract_field_value_functional,
    calculate_extraction_confidence,
)
from core.models.graph_state import create_initial_graph_state


class TestVoiceTypes:
    """Test immutable voice data types."""
    
    def test_voice_config_immutability(self):
        """Test that VoiceConfig is immutable."""
        config = VoiceConfig(
            google_api_key="test_key",
            voice_model=VoiceModel.PUCK
        )
        
        # Test immutability
        new_config = config.with_voice(VoiceModel.AOEDE)
        
        assert config.voice_model == VoiceModel.PUCK
        assert new_config.voice_model == VoiceModel.AOEDE
        assert config.google_api_key == new_config.google_api_key
    
    def test_conversation_state_transformations(self):
        """Test pure state transformations."""
        initial_state = create_conversation_state_initial()
        
        # Test transcript update
        with_transcript = initial_state.with_transcript("Hello world")
        assert initial_state.current_transcript == ""
        assert with_transcript.current_transcript == "Hello world"
        
        # Test mode change
        voice_mode = initial_state.with_mode(ConversationMode.VOICE)
        assert initial_state.mode == ConversationMode.VOICE  # Default
        assert voice_mode.mode == ConversationMode.VOICE
        
        # Test speaking state
        speaking_state = initial_state.with_speaking(True)
        assert initial_state.is_speaking == False
        assert speaking_state.is_speaking == True
    
    def test_voice_pipeline_result_factories(self):
        """Test VoicePipelineResult factory methods."""
        success_result = VoicePipelineResult.success_result(
            transcript="hello",
            response_text="hi there",
            response_audio=b"audio_data",
            processing_time=150.0
        )
        
        assert success_result.success == True
        assert success_result.transcript == "hello"
        assert success_result.processing_time_ms == 150.0
        
        error_result = VoicePipelineResult.error_result("test error", 100.0)
        assert error_result.success == False
        assert error_result.error_message == "test error"
        assert error_result.transcript is None


class TestVoiceEvents:
    """Test functional voice event processing."""
    
    def test_voice_event_creation(self):
        """Test voice event creation functions."""
        from core.services.voice_events import (
            create_audio_event,
            create_transcript_event,
            create_response_event,
        )
        
        # Audio event
        audio_event = create_audio_event(b"audio_data", 16000)
        assert audio_event.type == VoiceEventType.AUDIO_RECEIVED
        assert audio_event.get("audio_data") == b"audio_data"
        assert audio_event.get("sample_rate") == 16000
        
        # Transcript event
        transcript = TranscriptFrame(
            text="hello world",
            confidence=0.9,
            timestamp=time.time(),
            is_final=True
        )
        transcript_event = create_transcript_event(transcript)
        assert transcript_event.type == VoiceEventType.TRANSCRIPT_READY
        assert transcript_event.get("text") == "hello world"
        assert transcript_event.get("confidence") == 0.9
    
    def test_state_reducer_purity(self):
        """Test that state reducer is a pure function."""
        reducer = create_voice_state_reducer()
        initial_state = create_conversation_state_initial()
        
        # Create test event
        event = VoiceEvent(
            VoiceEventType.TRANSCRIPT_READY,
            {"text": "software engineer", "confidence": 0.8}
        )
        
        # Apply reducer
        new_state = reducer(initial_state, event)
        
        # Verify original state unchanged (immutability)
        assert initial_state.current_transcript == ""
        assert new_state.current_transcript == "software engineer"
        
        # Verify pure function behavior (same input = same output)
        second_application = reducer(initial_state, event)
        assert second_application.current_transcript == new_state.current_transcript
    
    def test_conversation_command_extraction(self):
        """Test functional command extraction."""
        # Test mode switch commands
        commands1 = extract_conversation_commands("switch to text mode please")
        assert commands1.get("mode_switch") == ConversationMode.TEXT
        
        commands2 = extract_conversation_commands("let's use voice mode")
        assert commands2.get("mode_switch") == ConversationMode.VOICE
        
        # Test navigation commands
        commands3 = extract_conversation_commands("go back to the previous question")
        assert commands3.get("navigation") == "back"
        
        commands4 = extract_conversation_commands("skip this question")
        assert commands4.get("navigation") == "skip"
        
        # Test empty input
        commands5 = extract_conversation_commands("")
        assert len(commands5) == 0


class TestVoiceIntegration:
    """Test voice integration with graph state."""
    
    def test_voice_to_graph_bridge(self):
        """Test functional bridge from voice state to graph state."""
        bridge_func = create_voice_state_bridge()
        
        # Create test graph state
        graph_state = create_initial_graph_state()
        graph_state["current_field"] = "job_title"
        graph_state["messages"] = [
            {"role": "user", "content": "software engineer", "timestamp": time.time()}
        ]
        
        # Convert to voice state
        voice_state = bridge_func(graph_state)
        
        assert voice_state.mode == ConversationMode.VOICE
        assert voice_state.current_transcript == "software engineer"
        assert voice_state.metadata["current_field"] == "job_title"
    
    def test_field_extraction_functional(self):
        """Test functional field extraction."""
        # Test job title extraction
        title = extract_field_value_functional("I need a software engineer", "job_title")
        assert title == "Software Engineer"
        
        # Test department extraction
        dept = extract_field_value_functional("engineering department", "department")
        assert dept == "Engineering"
        
        # Test employment type extraction
        emp_type = extract_field_value_functional("full time position", "employment_type")
        assert emp_type == "Full-time"
        
        # Test location extraction
        location = extract_field_value_functional("remote work", "location")
        assert location == "Remote"
    
    def test_confidence_calculation(self):
        """Test confidence calculation for extractions."""
        # Test high confidence cases
        high_conf = calculate_extraction_confidence("software engineer position", "job_title")
        assert high_conf > 0.7
        
        # Test low confidence cases
        low_conf = calculate_extraction_confidence("um", "job_title")
        assert low_conf < 0.5
        
        # Test field-specific bonuses
        job_conf = calculate_extraction_confidence("senior engineer", "job_title")
        dept_conf = calculate_extraction_confidence("engineering team", "department")
        assert job_conf > 0.6
        assert dept_conf > 0.6
    
    def test_field_extractor_bridge(self):
        """Test field extraction bridge function."""
        extractor = create_field_extractor_bridge()
        
        initial_job_data = {"department": "existing_dept"}
        
        # Extract job title
        updated_data = extractor(
            "Senior Software Engineer",
            "job_title", 
            initial_job_data
        )
        
        assert "job_title" in updated_data
        assert updated_data["job_title"]["value"] == "Senior Software Engineer"
        assert updated_data["job_title"]["source"] == "voice"
        assert updated_data["job_title"]["confidence"] > 0.0
        
        # Verify original data preserved
        assert updated_data["department"] == "existing_dept"


class TestFunctionalComposition:
    """Test functional programming patterns in voice pipeline."""
    
    def test_function_composition(self):
        """Test function composition utilities."""
        from core.services.voice_pipeline import compose, pipe
        
        # Test compose
        add_one = lambda x: x + 1
        multiply_two = lambda x: x * 2
        add_three = lambda x: x + 3
        
        composed = compose(add_three, multiply_two, add_one)
        result = composed(5)  # ((5 + 1) * 2) + 3 = 15
        assert result == 15
        
        # Test pipe
        piped_result = pipe(5, add_one, multiply_two, add_three)
        assert piped_result == 15
    
    def test_immutable_data_flow(self):
        """Test that data flow maintains immutability."""
        # Create initial data
        original_context = {
            "field": "job_title",
            "data": {"existing": "value"}
        }
        
        # Apply transformations
        def add_field(context: Dict[str, Any], field: str, value: Any) -> Dict[str, Any]:
            new_data = {**context["data"], field: value}
            return {**context, "data": new_data}
        
        updated_context = add_field(original_context, "new_field", "new_value")
        
        # Verify original unchanged
        assert "new_field" not in original_context["data"]
        assert "new_field" in updated_context["data"]
        assert original_context["data"]["existing"] == "value"
    
    def test_higher_order_functions(self):
        """Test higher-order function patterns."""
        from core.services.voice_pipeline import with_retry
        
        # Create a function that fails first few times
        call_count = 0
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Simulated failure")
            return "success"
        
        # Wrap with retry
        @with_retry(max_attempts=5)
        async def wrapped_function():
            return flaky_function()
        
        # Should succeed after retries
        import asyncio
        result = asyncio.run(wrapped_function())
        assert result == "success"
        assert call_count == 3


class TestVoiceServiceIntegration:
    """Test voice service integration components."""
    
    @pytest.fixture
    def mock_voice_config(self):
        """Fixture for test voice configuration."""
        return VoiceConfig(
            google_api_key="test_key",
            voice_model=VoiceModel.PUCK,
            sample_rate=16000,
            enable_interruption=True
        )
    
    def test_voice_service_creation(self, mock_voice_config):
        """Test functional voice service creation."""
        from core.services.pipecat_voice_service import FunctionalVoiceService
        
        service = FunctionalVoiceService(mock_voice_config)
        assert service.get_config() == mock_voice_config
        assert not service.is_running()
        
        # Test configuration update
        new_config = mock_voice_config.with_voice(VoiceModel.AOEDE)
        new_service = service.with_config(new_config)
        
        assert service.get_config().voice_model == VoiceModel.PUCK
        assert new_service.get_config().voice_model == VoiceModel.AOEDE
    
    def test_streamlit_integration_functions(self):
        """Test Streamlit integration helper functions."""
        from core.services.pipecat_voice_service import create_streamlit_voice_interface
        
        interface = create_streamlit_voice_interface()
        
        # Verify all required functions exist
        required_functions = ["initialize", "process_audio", "start_voice", "stop_voice"]
        for func_name in required_functions:
            assert func_name in interface
            assert callable(interface[func_name])


# Integration test
def test_end_to_end_functional_flow():
    """Test complete functional flow from voice input to graph state update."""
    
    # Create initial states
    graph_state = create_initial_graph_state()
    graph_state["current_field"] = "job_title"
    
    voice_state = create_conversation_state_initial()
    
    # Create transcript event
    transcript_event = VoiceEvent(
        VoiceEventType.TRANSCRIPT_READY,
        {
            "text": "Senior Software Engineer",
            "confidence": 0.9,
            "is_final": True
        }
    )
    
    # Process through voice pipeline
    from core.services.voice_integration import create_voice_to_graph_processor
    
    processor = create_voice_to_graph_processor()
    updated_graph_state = processor(transcript_event, graph_state)
    
    # Verify functional processing
    assert "job_title" in updated_graph_state.get("job_data", {})
    job_title_data = updated_graph_state["job_data"]["job_title"]
    assert job_title_data["value"] == "Senior Software Engineer"
    assert job_title_data["source"] == "voice"
    assert job_title_data["confidence"] > 0.5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])