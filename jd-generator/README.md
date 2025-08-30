# AI-Powered Job Description Generator

An intelligent conversational agent that conducts interview-style requirement gathering sessions to generate comprehensive Job Descriptions (JDs) using LangGraph orchestration and Streamlit UI with voice capabilities.

## Features

- **Conversational AI**: Natural dialogue flow for gathering job requirements
- **Voice Input**: Support for voice-based input using OpenAI Whisper
- **Smart Orchestration**: LangGraph-powered state management and flow control
- **Professional Output**: Generate well-formatted, comprehensive job descriptions
- **Export Options**: Download JDs in multiple formats

## Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI API key (for GPT-4 and Whisper)
- Optional: Anthropic API key (for Claude)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd jd-generator
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. Run the application:
```bash
streamlit run app/main.py
```

## Project Structure

```
jd-generator/
├── app/                    # Streamlit application
│   ├── main.py            # Entry point
│   ├── config.py          # Configuration
│   └── utils/             # Utilities
├── core/                   # Core business logic
│   ├── models/            # Data models
│   ├── graph/             # LangGraph implementation
│   ├── prompts/           # Prompt templates
│   └── services/          # External services
├── ui/                     # UI components
│   ├── components/        # Streamlit components
│   └── styles/            # Custom styling
└── tests/                  # Test suite
```

## Usage

1. Start the application
2. Begin conversation with the AI assistant
3. Answer questions about the job position
4. Review the generated job description
5. Export in your preferred format

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
ruff check .
```

### Type Checking
```bash
mypy .
```

## License

[Your License]

## Contributing

[Contributing Guidelines]