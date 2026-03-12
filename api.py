#!/usr/bin/env python3
"""
Hiroyuki SLM API Service
Flask-based API for Hiroyuki-style chat responses
"""

import json
import os
import sys
from flask import Flask, request, jsonify
from slm_model import HiroyukiChat, load_quotes, load_responses

app = Flask(__name__)

# Global chat handler
chat_handler = None


def init_chat():
    """Initialize the chat handler"""
    global chat_handler
    
    # Determine paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    quotes_path = os.path.join(base_dir, 'quotes.json')
    responses_path = os.path.join(base_dir, 'responces.json')
    
    # Fallback to home/engine/project if files not in base_dir
    if not os.path.exists(quotes_path):
        quotes_path = '/home/engine/project/quotes.json'
    if not os.path.exists(responses_path):
        responses_path = '/home/engine/project/responces.json'
    
    print(f"Loading quotes from: {quotes_path}")
    print(f"Loading responses from: {responses_path}")
    
    chat_handler = HiroyukiChat(quotes_path, responses_path)
    print("Chat handler initialized successfully")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'hiroyuki-slm-4bit',
        'version': '1.0.0'
    })


@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat endpoint
    
    Request body:
    {
        "message": "user input string"
    }
    
    Response:
    {
        "response": "model output string"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing "message" field'}), 400
            
        user_message = data['message']
        
        if not isinstance(user_message, str):
            return jsonify({'error': '"message" must be a string'}), 400
            
        # Generate response
        response = chat_handler.generate_response(user_message)
        
        return jsonify({
            'response': response,
            'input': user_message
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat/stream', methods=['POST'])
def chat_stream():
    """
    Streaming chat endpoint (simple implementation - returns full response)
    For production, would implement Server-Sent Events
    """
    return chat()


@app.route('/models/info', methods=['GET'])
def models_info():
    """Get model information"""
    return jsonify({
        'model_name': 'hiroyuki-slm-4bit',
        'quantization': '4-bit',
        'parameters': '~2M',
        'vocab_size': chat_handler.tokenizer.vocab_size,
        'context_length': 32
    })


def create_app():
    """Create and configure the Flask app"""
    init_chat()
    return app


if __name__ == '__main__':
    # Initialize chat handler
    init_chat()
    
    # Run the server
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"Starting Hiroyuki SLM API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
