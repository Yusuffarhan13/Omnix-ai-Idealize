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

