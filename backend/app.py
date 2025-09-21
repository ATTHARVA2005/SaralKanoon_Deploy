# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.pdf_processor import extract_text_from_pdf
from utils.ai_client import GeminiClient, translate_text, text_to_speech

# Initialize Flask App and CORS
app = Flask(__name__)
# This is crucial to allow your React frontend to communicate with this backend
CORS(app) 

# In-memory storage for the extracted text of the last uploaded document.
# For a hackathon, this is sufficient. For production, you'd use a more robust cache like Redis.
document_context_store = {
    "text": None
}

# Initialize the AI client once
try:
    ai_client = GeminiClient()
except ValueError as e:
    print(f"Failed to initialize GeminiClient: {e}")
    ai_client = None

# --- API Endpoints ---

# Translation endpoint
@app.route('/translate', methods=['POST'])
def translate_section():
    """
    Endpoint to translate a section of the document to a target language.
    Expects JSON: { "section": "summary"|"keyClauses"|"redFlags", "text": "...", "target_lang": "hi"|"en"|... }
    """
    data = request.get_json()
    if not data or 'section' not in data or 'text' not in data or 'target_lang' not in data:
        return jsonify({"error": "Missing required fields."}), 400

    # Only allow translation of keyClauses/redFlags if explicitly requested
    if data['section'] in ['keyClauses', 'redFlags'] and data.get('target_lang') == 'hi':
        # Only translate if user requested
        pass

    try:
        translated = translate_text(data['text'], data['target_lang'])
        return jsonify({"translated": translated})
    except Exception as e:
        print(f"Error in /translate: {e}")
        return jsonify({"error": "Translation failed."}), 500

# Audio endpoint
@app.route('/audio', methods=['POST'])
def audio_section():
    """
    Endpoint to convert a section of text to speech audio.
    Expects JSON: { "text": "...", "lang": "en"|"hi"|... }
    Returns: audio file (mp3/wav)
    """
    data = request.get_json()
    if not data or 'text' not in data or 'lang' not in data:
        return jsonify({"error": "Missing required fields."}), 400
    try:
        audio_bytes = text_to_speech(data['text'], data['lang'])
        # Return as file download
        return (audio_bytes, 200, {
            'Content-Type': 'audio/mpeg',
            'Content-Disposition': 'attachment; filename="output.mp3"'
        })
    except Exception as e:
        print(f"Error in /audio: {e}")
        return jsonify({"error": "Audio conversion failed."}), 500

@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    """
    Endpoint to upload a PDF, extract text, and get the initial analysis.
    The extracted text is stored in memory for follow-up questions.
    """
    if not ai_client:
        return jsonify({"error": "AI client is not initialized. Check API key."}), 500

    if 'document' not in request.files:
        return jsonify({"error": "No document file provided"}), 400
    
    pdf_file = request.files['document']
    
    if pdf_file.filename == '' or not pdf_file.filename.endswith('.pdf'):
        return jsonify({"error": "Please provide a valid PDF file"}), 400

    try:
        extracted_text = extract_text_from_pdf(pdf_file.stream)
        if not extracted_text.strip():
            return jsonify({"error": "Could not extract text from PDF"}), 400

        # Store the text for the Q&A endpoint
        document_context_store["text"] = extracted_text
        
        # Get the analysis from the AI client
        analysis_result = ai_client.analyze_document(extracted_text)
        
        if "error" in analysis_result:
             return jsonify(analysis_result), 500

        return jsonify(analysis_result)

    except Exception as e:
        print(f"An error occurred in /analyze: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Endpoint to ask a follow-up question about the most recently uploaded document.
    """
    if not ai_client:
        return jsonify({"error": "AI client is not initialized. Check API key."}), 500

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    
    user_question = data['question']
    
    # Retrieve the stored document text
    document_text = document_context_store.get("text")
    
    if not document_text:
        return jsonify({"error": "No document has been analyzed yet. Please upload a document first."}), 400

    try:
        answer = ai_client.answer_question(document_text, user_question)
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"An error occurred in /ask: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500
    
@app.route('/compare', methods=['POST'])
def compare_pdfs():
    """
    Endpoint to upload two PDFs, compare them, and return the differences.
    """
    if not ai_client:
        return jsonify({"error": "AI client is not initialized"}), 500

    if 'old_document' not in request.files or 'new_document' not in request.files:
        return jsonify({"error": "Both 'old_document' and 'new_document' files are required"}), 400

    old_pdf = request.files['old_document']
    new_pdf = request.files['new_document']

    if old_pdf.filename == '' or new_pdf.filename == '':
        return jsonify({"error": "Please provide both documents"}), 400

    try:
        # Extract text from both documents
        old_text = extract_text_from_pdf(old_pdf.stream)
        new_text = extract_text_from_pdf(new_pdf.stream)

        if not old_text.strip() or not new_text.strip():
            return jsonify({"error": "Could not extract text from one or both PDFs."}), 400

        # Call the new comparison method in the AI client
        comparison_result = ai_client.compare_documents(old_text, new_text)

        if "error" in comparison_result:
            return jsonify(comparison_result), 500

        return jsonify(comparison_result)

    except Exception as e:
        print(f"An error occurred in /compare: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

# --- Main entry point ---
if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:5000
    # The debug=True flag allows for hot-reloading when you save changes.
    app.run(debug=True, port=5000)
