The application is currently under active development. Several of the features highlighted in the video pitch are being refined and remain in the modification phase.

Omnix AI Assistant: Your Multi-Modal Intelligent Companion
This project implements a Flask-based backend for the Omnix AI Assistant, a powerful and versatile AI system designed to handle a wide range of tasks, from creative conversations to complex research and autonomous web interactions. It integrates cutting-edge large language models and specialized AI agents to deliver a comprehensive intelligent experience.

Core Modes & Capabilities
The Omnix AI Assistant operates across distinct modes, each optimized for specific types of tasks:

1. Chat Mode
Purpose: Ideal for rapid, creative tasks, quick Q&A, and general conversations.

Technology: Powered by Google Gemini 2.5 Flash and a custom transformer for efficient and engaging interactions.

Example: As demonstrated, it can generate a futuristic poem about AI collaborating with humans, showcasing its creative prowess.

2. Research Mode
Purpose: Designed for serious tasks requiring in-depth information gathering, analysis, and synthesis. It prioritizes accuracy and trust in its responses.

Technology: Leverages Brave Search API for web searches and integrates with PraisonAI's MCP search tools for active information gathering. It uses Google Gemini-Flash for synthesizing the gathered information.

Example: The system can answer complex queries like "promising battery technology developments in 2025" by actively collecting and processing information from the web, providing a detailed, source-backed summary.

3. Browser Agent
Purpose: Enables the AI to autonomously navigate, interact with, and extract information from websites.

Technology: Utilizes the browser-use library with Microsoft Edge for web automation and a dedicated agent for intelligent interaction. This agent can combine its capabilities with the Research Agent for deeper information retrieval and complex web scraping tasks.

Example: Acts as a research assistant by going to Amazon.com to compare products (e.g., Sony and Bose headphones). It navigates to product pages, extracts specific details like price, star rating, and battery life, and then presents a concise summary.

4. Advanced Agent (Complex Task Solving)
Purpose: The powerhouse mode for tackling highly complex challenges, including intricate coding problems, advanced mathematical computations, and deep research requiring multi-step reasoning.

Technology: Combines the strengths of DeepSeek R10528 and Google Gemini 2.5 Pro. It employs a highly tuned chain of thought and leverages specialized PraisonAI MCP servers for sequential thinking and post-reinforcement learning, granting it unique and robust problem-solving capabilities.

Additional Features
Speech-to-Text (STT): Transcribes audio input using the Whisper model, enabling voice commands and interactions.

Text-to-Speech (TTS): Converts text to natural-sounding speech using the ElevenLabs API, providing an auditory response experience.

Real-time Task Status: Streams live updates for ongoing browser automation tasks, keeping users informed of progress.

Persistent Task Tracking: Uses an SQLite database to store and manage the status of browser tasks.

Technologies Used
Backend: Python 3, Flask

AI/ML Models: Google Gemini (2.5 Flash, 2.5 Pro), DeepSeek R10528 (via PraisonAI)

AI Frameworks/Libraries: LangChain Google Generative AI, PraisonAI Agents (for sequential thinking, GitHub interaction, MCP servers)

Browser Automation: browser-use library (requires Microsoft Edge)

Search: Brave Search API

Voice: Whisper (STT), ElevenLabs API (TTS)

Database: SQLite3

Environment Management: python-dotenv

Asynchronous Operations: asyncio, concurrent.futures.ThreadPoolExecutor

Setup
Prerequisites
Python 3.8+: Ensure Python is installed on your system.

Microsoft Edge Browser: The browser automation feature requires Microsoft Edge to be installed.

API Keys: You will need API keys for the following services:

Google Gemini API: For GOOGLE_API_KEY.

ElevenLabs API: For ELEVENLABS_API_KEY.

Brave Search API: For BRAVE_API_KEY.

GitHub Personal Access Token: For GITHUB_PERSONAL_ACCESS_TOKEN (if using the complex task agent with GitHub features).

Installation
Clone the repository:

git clone <repository_url>
cd <repository_directory>

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
Create a requirements.txt file with the following content:

Flask
python-dotenv
langchain-google-genai
google-generativeai
pydantic
requests
bravepy
whisper
pydub
praisonaiagents
browser-use

Then install them:

pip install -r requirements.txt

Note: browser-use might have additional system dependencies related to browser drivers. Refer to its documentation if you encounter issues.

Configure Environment Variables:
Create a .env file in the root directory of the project and add your API keys and browser paths:

EDGE_EXECUTABLE=/path/to/msedge.exe  # e.g., C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe or /usr/bin/microsoft-edge
EDGE_USER_DATA_DIR=/path/to/edge/user/data/directory # e.g., C:\Users\YourUser\AppData\Local\Microsoft\Edge\User Data or ~/.config/microsoft-edge
GOOGLE_API_KEY=YOUR_GOOGLE_GEMINI_API_KEY
ELEVENLABS_API_KEY=YOUR_ELEVENLABS_API_KEY
BRAVE_API_KEY=YOUR_BRAVE_SEARCH_API_KEY
GITHUB_PERSONAL_ACCESS_TOKEN=YOUR_GITHUB_PERSONAL_ACCESS_TOKEN

Important: Replace placeholder values with your actual API keys and correct paths. The EDGE_USER_DATA_DIR should point to a directory where Edge can store user profiles (it will create subdirectories within it).

Usage
Run the Flask application:

python main.py

The application will start on http://0.0.0.0:8000.

Access the Frontend:
Open your web browser and navigate to http://localhost:8000. The index.html file in the frontend directory will be served.

API Endpoints
The backend exposes the following API endpoints:

GET /: Serves the index.html file from the frontend directory.

POST /chat:

Request: JSON with {"message": "Your message here"}

Response: JSON with {"response": "AI's response"}

Description: Sends a user message to the gemini-2.5-flash model for general chat (Chat Mode).

POST /run_task:

Request: JSON with {"task": "Description of browser task"}

Response: JSON with {"task_id": "unique_task_id"}

Description: Initiates a browser automation task in the background (Browser Agent). Returns a task_id to track its progress.

GET /stream/<task_id>:

Response: Server-Sent Events (SSE) stream.

Description: Provides real-time updates on the status and results of a specific browser task identified by task_id.

POST /research:

Request: JSON with {"query": "Your research query"}

Response: JSON with {"summary": "Comprehensive answer", "sources": [{"title": "...", "url": "..."}]}

Description: Performs a web search using Brave and synthesizes a summary using an AI agent (Research Mode).

POST /complex_task:

Request: JSON with {"prompt": "Your complex task description"}

Response: JSON with {"response": "Solution or detailed plan"}

Description: Uses multiple AI agents to break down and solve complex problems, potentially involving code generation or GitHub interaction (Advanced Agent).

POST /stt:

Request: multipart/form-data with an audio file.

Response: JSON with {"text": "Transcribed text"}

Description: Transcribes an audio file to text using the Whisper model.

POST /tts:

Request: JSON with {"text": "Text to convert to speech"}

Response: audio/mpeg file (MP3).

Description: Converts the provided text into an audio file using ElevenLabs.

Database
The application uses an SQLite database (tasks.db) to store the status and results of browser automation tasks. The database is initialized automatically on application startup.

Logging
Basic logging is configured to output informational messages and errors to the console, which can be helpful for debugging.

Cleanup
The application attempts to clean up temporary browser profiles created during browser automation tasks. Temporary audio files for STT/TTS are also removed after processing.