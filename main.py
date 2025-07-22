import asyncio
import os
import uuid
import json
import sqlite3
import shutil
import requests
from flask import Flask, request, jsonify, Response, send_from_directory
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pydantic import SecretStr, ConfigDict
from browser_use import Agent as BrowserAgent # Renamed to avoid conflict
from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.llm import ChatGoogle
from concurrent.futures import ThreadPoolExecutor
import logging
import atexit
import queue
import whisper
from pydub import AudioSegment
from brave import Brave
from typing import Literal
os.environ["OPENAI_API_KEY"] = "not-needed"
from praisonaiagents import Agent, MCP # New import for PraisonAI

# --- Initialization ---
load_dotenv()

# FIX: Set a placeholder for the OpenAI API key to satisfy the praisonaiagents library's default check.


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__, static_folder='frontend', template_folder='frontend')
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# --- Configuration ---
EDGE_EXECUTABLE = os.getenv("EDGE_EXECUTABLE")
EDGE_USER_DATA_DIR = os.getenv("EDGE_USER_DATA_DIR")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

if not all([EDGE_EXECUTABLE, EDGE_USER_DATA_DIR, GOOGLE_API_KEY, ELEVENLABS_API_KEY, BRAVE_API_KEY, GITHUB_PERSONAL_ACCESS_TOKEN]):
    raise ValueError("Please set all required API keys and paths in your .env file, including GITHUB_PERSONAL_ACCESS_TOKEN.")

# Initialize Brave Search client
brave_client = Brave(BRAVE_API_KEY)

# --- LLM Initialization ---

# LLM for Complex tasks (using Langchain for its features)
llm_pro = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    api_key=SecretStr(GOOGLE_API_KEY),
    temperature=0.1
)

# LLM for general chat (using Google's native SDK)
genai.configure(api_key=GOOGLE_API_KEY)
chat_llm_flash = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# --- Database Setup for Browser Tasks ---
DB_FILE = 'tasks.db'

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
init_db()

# --- Real-time Update Queue for Browser Tasks ---
class UpdateManager:
    def __init__(self):
        self.listeners = {}

    def subscribe(self, task_id):
        q = queue.Queue()
        self.listeners[task_id] = q
        return q

    def publish(self, task_id, data):
        if task_id in self.listeners:
            self.listeners[task_id].put(data)

    def unsubscribe(self, task_id):
        if task_id in self.listeners:
            del self.listeners[task_id]

update_manager = UpdateManager()

# --- Thread Pool for Background Tasks ---
executor = ThreadPoolExecutor(max_workers=5)
atexit.register(lambda: executor.shutdown(wait=False))


# --- Browser Control Agent Logic ---
async def execute_browser_task(task_id, task_description):
    profile_name = f"Task_{task_id}"
    
    def update_status(status, result=None):
        with app.app_context():
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE tasks SET status = ?, result = ? WHERE id = ?", (status, result, task_id))
                conn.commit()
            update_manager.publish(task_id, {'status': status, 'result': result})

    browser_session = None
    try:
        update_status('RUNNING', 'Initializing browser...')
        
        browser_llm = ChatGoogle(model='gemini-2.5-flash', api_key=GOOGLE_API_KEY)
        
        browser_session = BrowserSession(
            browser_profile=BrowserProfile(
                profile_name=profile_name,
                user_data_dir=EDGE_USER_DATA_DIR,
                executable_path=EDGE_EXECUTABLE,
                viewport_expansion=0,
            )
        )
        
        agent = BrowserAgent(
            task=task_description, 
            browser_session=browser_session, 
            llm=browser_llm, 
            max_actions_per_step=15
        )
        
        update_status('RUNNING', 'Agent is running...')
        await agent.run(max_steps=30)
        
        update_status('COMPLETED', "Task finished.")
    except Exception as e:
        app.logger.error(f"Task {task_id} failed: {e}", exc_info=True)
        update_status('FAILED', f"An unexpected error occurred: {e}")
    finally:
        if browser_session: 
            await browser_session.close()
        profile_path = os.path.join(EDGE_USER_DATA_DIR, "Profiles", profile_name)
        if os.path.isdir(profile_path):
            try:
                shutil.rmtree(profile_path)
                app.logger.info(f"Successfully cleaned up profile: {profile_path}")
            except Exception as e:
                app.logger.error(f"Error cleaning up profile {profile_name}: {e}")
        update_manager.unsubscribe(task_id)

# --- Flask Routes ---
@app.route('/')
def home():
    return send_from_directory(app.template_folder, 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    if not user_message: return jsonify({'error': 'Message cannot be empty'}), 400
    try:
        response = chat_llm_flash.generate_content(user_message)
        return jsonify({'response': response.text})
    except Exception as e:
        app.logger.error(f"Chat failed: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred during chat: {e}"}), 500

@app.route('/run_task', methods=['POST'])
def run_task():
    data = request.get_json()
    task_description = data.get('task', '').strip()
    if not task_description: return jsonify({'error': 'Task description cannot be empty'}), 400
    task_id = str(uuid.uuid4())
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO tasks (id, description, status) VALUES (?, ?, ?)", (task_id, task_description, 'PENDING'))
        conn.commit()
    executor.submit(asyncio.run, execute_browser_task(task_id, task_description))
    return jsonify({'task_id': task_id})

@app.route('/stream/<task_id>')
def stream(task_id):
    def event_stream():
        q = update_manager.subscribe(task_id)
        try:
            while True:
                data = q.get()
                yield f"data: {json.dumps(data)}\n\n"
                if data.get('status') in ['COMPLETED', 'FAILED']: break
        finally:
            update_manager.unsubscribe(task_id)
    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/research', methods=['POST'])
def research_agent():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query: return jsonify({'error': 'Query cannot be empty'}), 400
    
    try:
        headers = {"X-Subscription-Token": BRAVE_API_KEY}
        params = {"q": query, "count": 5}
        response = requests.get("https://api.search.brave.com/res/v1/web/search", params=params, headers=headers)
        response.raise_for_status()
        search_results = response.json()

        context = ""
        sources = []
        for result in search_results.get("web", {}).get("results", []):
            title = result.get("title", "No Title")
            url = result.get("url", "#")
            description = result.get("description", "No description available.")
            context += f"Title: {title}\nURL: {url}\nSnippet: {description}\n\n"
            sources.append({"title": title, "url": url})
        
        # Use PraisonAI for sequential thinking with Gemini
        sequential_agent = Agent(
            instructions="You are a helpful assistant that can break down complex problems. Use the available tools when relevant to perform step-by-step analysis.",
            llm="gemini/gemini-1.5-pro-latest",
            tools=MCP("npx -y @modelcontextprotocol/server-sequential-thinking", env={"GOOGLE_API_KEY": GOOGLE_API_KEY})
        )
        thinking_result = sequential_agent.start(f"Based on the following context, break down the answer to the query: '{query}'.\n\nContext:\n{context}")
        
        final_prompt = f"""
        User Query: "{query}"
        Initial Search Context:
        {context}
        Processed Thoughts from Sequential Thinking Agent:
        {thinking_result}
        Your task is to synthesize all this information into a final, comprehensive answer.
        """
        response = chat_llm_flash.generate_content(final_prompt)
        return jsonify({'summary': response.text, 'sources': sources})
    except Exception as e:
        app.logger.error(f"Research agent failed: {e}", exc_info=True)
        return jsonify({'error': 'Failed to perform research.'}), 500

@app.route('/complex_task', methods=['POST'])
def complex_task_agent():
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    if not prompt: return jsonify({'error': 'Prompt cannot be empty'}), 400
    try:
        # Step 1: Use PraisonAI for sequential thinking with Gemini
        sequential_agent = Agent(
            instructions="You are a helpful assistant that can break down complex problems. Use the available tools when relevant to perform step-by-step analysis.",
            llm="gemini/gemini-1.5-pro-latest",
            tools=MCP("npx -y @modelcontextprotocol/server-sequential-thinking", env={"GOOGLE_API_KEY": GOOGLE_API_KEY})
        )
        thinking_result = sequential_agent.start(f"Break down the steps to solve the following task: {prompt}")

        github_context = ""
        if "code" in prompt.lower() or "github" in prompt.lower():
            # Step 2: Use PraisonAI for GitHub interaction with Gemini
            github_agent = Agent(
                instructions="You are a helpful assistant that can interact with GitHub. Use the available tools when relevant to answer user questions.",
                llm="gemini/gemini-1.5-pro-latest",
                tools=MCP("npx -y @modelcontextprotocol/server-github", env={"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_PERSONAL_ACCESS_TOKEN, "GOOGLE_API_KEY": GOOGLE_API_KEY})
            )
            # You might need to formulate a more specific query for the GitHub agent based on the prompt
            github_context = github_agent.start(f"Based on the task '{prompt}', what information can you find on GitHub?")

        system_prompt = """You are an expert-level AI assistant for complex problem-solving.
        You have been provided with a step-by-step plan from a sequential thinking agent and, optionally, context from a GitHub agent.
        Your task is to synthesize this information to produce a final, complete, and accurate solution.
        If the task requires code, write clean, efficient, and well-commented Python code.
        Format your response clearly, using Markdown for code blocks and explanations."""
       
        full_prompt = f"""
        {system_prompt}
        Original User Task:
        {prompt}
        Sequential Thinking Breakdown:
        {thinking_result}
        GitHub Context:
        {github_context or "Not applicable."}
        Now, provide the final solution.
        """
        response = llm_pro.invoke(full_prompt)
        return jsonify({'response': response.content})
    except Exception as e:
        app.logger.error(f"Complex task agent failed: {e}", exc_info=True)
        return jsonify({'error': 'Failed to process the complex task.'}), 500


@app.route('/stt', methods=['POST'])
def stt():
    if 'audio' not in request.files: return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    temp_audio_path = f"./temp_audio_{uuid.uuid4()}.webm"
    audio_file.save(temp_audio_path)
    wav_path = f"./temp_audio_{uuid.uuid4()}.wav"
    try:
        audio = AudioSegment.from_file(temp_audio_path, format="webm")
        audio.export(wav_path, format="wav")
        result = whisper_model.transcribe(wav_path, fp16=False)
        return jsonify({'text': result["text"]})
    except Exception as e:
        app.logger.error(f"STT failed: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred during STT: {e}"}), 500
    finally:
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
        if os.path.exists(wav_path): os.remove(wav_path)

@app.route('/tts', methods=['POST'])
def tts():
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text: return jsonify({'error': 'No text provided'}), 400

    voice_id = "21m00Tcm4TlvDq8ikWAM" # Rachel
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
    }

    try:
        response = requests.post(tts_url, json=payload, headers=headers)
        if response.status_code == 200:
            audio_filename = f"./temp_tts_{uuid.uuid4()}.mp3"
            with open(audio_filename, "wb") as f:
                f.write(response.content)
            return send_from_directory(os.path.dirname(audio_filename), os.path.basename(audio_filename), as_attachment=True)
        else:
            app.logger.error(f"ElevenLabs API request failed: {response.status_code} - {response.text}")
            return jsonify({'error': 'Failed to generate speech.'}), response.status_code
    except Exception as e:
        app.logger.error(f"TTS failed: {e}", exc_info=True)
        return jsonify({'error': f"An error occurred during TTS: {e}"}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8000, debug=False)
