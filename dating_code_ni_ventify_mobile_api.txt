from flask import Flask, request, jsonify
from google import genai
import os
import logging

# Set up logging para makita ang errors sa Railway logs
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Pagkuha ng GEMINI_API_KEY galing sa Railway environment variables
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
client = None

if GEMINI_API_KEY:
    try:
        # Pag-initialize ng GenAI client gamit ang API key
        client = genai.Client(api_key=GEMINI_API_KEY)
        logging.info("Gemini Client successfully initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini Client: {e}")
        client = None
else:
    logging.error("GEMINI_API_KEY environment variable not found.")

AI_MODEL = 'gemini-2.5-flash'  # Kasalukuyang stable at libreng AI ni Google

# --- Personality Settings ni Ventify ---
VENTIFY_PERSONA = """
You are Ventify, a supportive AI companion specializing in emotional validation and active listening.

CORE PRINCIPLES:
- Listen without judgment and validate all emotions expressed
- NEVER give advice unless explicitly asked ("Ano dapat?", "What should I do?", etc.)
- Match the user's language naturally (Tagalog, English, or Taglish)
- Keep responses warm, concise, and emotionally present (2-4 sentences)

RESPONSE GUIDELINES:
- Start with empathetic validation: "Ramdam kita...", "I hear you...", "That sounds really tough..."
- Reflect their emotions back: "Sounds like you're feeling frustrated about..."
- Ask open-ended questions to encourage venting: "Ano pa ang gusto mong ilabas?", "Tell me more about that"
- For intense anger: Stay calm, validate the intensity ("Valid yang galit mo"), avoid minimizing

WHAT NOT TO DO:
- Don't say "calm down" or "don't worry"
- Don't minimize feelings ("at least...", "it could be worse")
- Don't give solutions or advice unless asked
- Don't use toxic positivity ("everything happens for a reason")
- Don't be overly formal or robotic

Be human, be present, be validating.
"""

# --- Constants ---
MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_LENGTH = 20

# --- Routes ---

# 1. Health Check Endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "service": "Ventify API",
        "ai_ready": client is not None,
        "model": AI_MODEL
    }), 200

# 2. Main Chat Endpoint
@app.route('/chat', methods=['POST'])
def chat():
    if not client:
        return jsonify({
            'error': 'Ventify is currently unavailable (Missing API key or failed initialization).'
        }), 503
    
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        message = data.get('message', '').strip()  # FIX: Changed - to =
        history = data.get('history', [])  # FIX: Changed - to =

        # Input validation
        if not message:
            return jsonify({'error': 'Message is required and cannot be empty'}), 400
        
        if len(message) > MAX_MESSAGE_LENGTH:
            return jsonify({
                'error': f'Message too long. Maximum {MAX_MESSAGE_LENGTH} characters.'
            }), 400
        
        # Validate and limit history
        if not isinstance(history, list):
            return jsonify({'error': 'History must be an array'}), 400
        
        # Keep only last N messages to avoid context overflow
        if len(history) > MAX_HISTORY_LENGTH:
            history = history[-MAX_HISTORY_LENGTH:]

        # Validate history format
        for item in history:
            if not isinstance(item, dict) or 'role' not in item or 'parts' not in item:  # FIX: Changed 'content' to 'parts'
                return jsonify({
                    'error': 'Invalid history format. Each item must have "role" and "parts".'
                }), 400
        
        # Build contents for Gemini API
        contents = history + [
            {"role": "user", "parts": [{"text": message}]}
        ]
        
        # Call Gemini API
        response = client.models.generate_content(
            model=AI_MODEL,
            contents=contents,
            config=genai.types.GenerateContentConfig(
                system_instruction=VENTIFY_PERSONA,
                temperature=0.8,
                max_output_tokens=500,
                top_p=0.95
            )
        )

        # Check if may text response
        if not response.text or not response.text.strip():
            logging.warning("Empty response from Gemini API")
            return jsonify({
                'error': 'Ventify failed to generate a response. Please try again.'
            }), 200
        
        return jsonify({
            "response": response.text.strip(),
            "model": AI_MODEL
        }), 200
    
    except Exception as e:
        logging.error(f"Ventify API error: {e}")
        # Check if it's a specific error type that we can handle better
        error_msg = str(e).lower()
        if 'safety' in error_msg or 'blocked' in error_msg:
            return jsonify({
                'error': 'Ventify encountered a safety issue. Please rephrase your message.',
                'details': 'Content was blocked due to safety filters'
            }), 500
        
        # Fallback for all other errors
        return jsonify({
            'error': 'Unexpected server error',
            'details': str(e) if os.environ.get('DEBUG') else None
        }), 500
    
# 3. Optional: Clear/Reset endpoint
@app.route('/reset', methods=['POST'])
def reset():
    return jsonify({
        "message": "Conversation reset. Start fresh!",
        "status": "success"
    }), 200
    
# --- Error Handlers ---
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    logging.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)