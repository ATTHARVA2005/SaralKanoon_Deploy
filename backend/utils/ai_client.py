# backend/utils/ai_client.py

import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from gtts import gTTS

# Load environment variables from a .env file
load_dotenv()

class GeminiClient:
    """
    A client to interact with the Google Gemini API, specifically tuned
    for the Saral Kanoon application.
    """
    def __init__(self, model_name="gemini-2.5-flash"):
        """
        Initializes the Gemini client.
        """
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
        
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model_name)
        # Initialize vision model for PDF processing
        self.vision_model = genai.GenerativeModel('gemini-2.5-flash')

    def extract_text_from_image(self, image_bytes: bytes) -> str:
        """
        Extracts text from a PDF page using Gemini's vision capabilities.
        
        Args:
            image_bytes: The bytes of the PDF page rendered as an image
            
        Returns:
            str: Extracted text from the image
        """
        from PIL import Image
        import io
        
        prompt = """
        Extract all text from this image. This is a page from a legal document.
        Return only the extracted text, maintaining the original formatting where possible.
        Do not include any additional commentary or analysis.
        """
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Create image parts for Gemini
            image_part = {"mime_type": "image/png", "data": image_bytes}
            
            # Generate content with proper image formatting
            response = self.vision_model.generate_content([prompt, image_part])
            response.resolve()  # Ensure the response is complete
            return response.text.strip()
        except Exception as e:
            print(f"Error in vision processing: {e}")
            return ""

    def analyze_document(self, document_text: str) -> dict:
        """
        Analyzes the full text of a legal document and returns a structured JSON.
        """
        prompt = f"""
        **Instruction:**
        You are an expert legal assistant named "Saral Kanoon" for an Indian audience. Your task is to analyze the provided legal document text and return a valid JSON object.
        The JSON object must have three keys: "summary", "keyClauses", and "redFlags".

        1.  **summary**: A concise, easy-to-understand summary of the document's main purpose in plain and simple to understand English without using uncommon english words.
        2.  **keyClauses**: An array of objects, where each object represents one of the 3-5 most important clauses. Each object must have a "title" and a "detail" explaining its impact on the user.
        3.  **redFlags**: An array of objects identifying clauses that are risky, unfair, or unusual. Each object must have a "title" and a "detail" explaining the potential risk. If there are no red flags, return an empty array.

        **Document Text to Analyze:**
        ---
        {document_text}
        ---

        **JSON Response:**
        """
        try:
            response = self.model.generate_content(prompt)
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from AI response.")
            return {"error": "Could not parse the AI's analysis."}
        except Exception as e:
            print(f"Error during analysis: {e}")
            return {"error": "An error occurred during document analysis."}

    def answer_question(self, document_text: str, user_question: str) -> str:
        """
        Answers a user's question based ONLY on the provided document context.
        """
        prompt = f"""
        **Instruction:**
        You are a Q&A assistant for "Saral Kanoon". Answer the user's question in simplest english possible based *ONLY* on the provided document text.
        Do not use any external knowledge. If the answer is not in the document, you MUST state: "The answer to that question could not be found in the provided document."

        **Document Text:**
        ---
        {document_text}
        ---

        **User's Question:**
        "{user_question}"

        **Answer:**
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error during Q&A: {e}")
            return "Sorry, an error occurred while answering your question."
    
    def compare_documents(self, old_doc_text: str, new_doc_text: str) -> dict:
        """
        Compares two legal documents and highlights the differences and risks.
        """
        prompt = f"""
        **Instruction:**
        You are an expert legal assistant, "Saral Kanoon", specializing in contract comparison for an Indian audience.
        Analyze the "Old Document" and the "New Document" provided below. Your goal is to identify all significant differences and assess the risk of these changes for the person signing the new contract.

        Your response MUST be a valid JSON object with four keys: "overallRiskAssessment", "newClauses", "removedClauses", and "modifiedClauses".

        1.  **overallRiskAssessment**: An object with a "rating" (e.g., "Low Risk", "Medium Risk", "High Risk") and a "summary" explaining the overall implication of the changes.
        2.  **newClauses**: An array of objects for clauses present in the New Document but absent in the Old. Each object must have a "title" and a "detail" explaining the new obligation or term.
        3.  **removedClauses**: An array of objects for clauses from the Old Document that are now missing. Each object must have a "title" and a "detail" explaining what protection or term has been lost.
        4.  **modifiedClauses**: This is the most important part. An array of objects for clauses that have been changed. Each object MUST have:
            - "clauseTitle": The name of the clause (e.g., "Notice Period").
            - "oldTextSummary": A brief summary of the clause in the Old Document.
            - "newTextSummary": A brief summary of the clause in the New Document.
            - "riskAnalysis": A clear explanation of the change's impact and any new risks involved.

        **Old Document Text:**
        ---
        {old_doc_text}
        ---

        **New Document Text:**
        ---
        {new_doc_text}
        ---

        **JSON Response:**
        """
        try:
            response = self.model.generate_content(prompt)
            cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_response)
        except Exception as e:
            print(f"Error during document comparison: {e}")
            return {"error": "An error occurred during document comparison."}

# --- Translation Utility ---
def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text to the target language using Gemini API.
    """
    # For demo: Use Gemini for translation, fallback to English if not supported
    prompt = f"""
    Translate the following text to {target_lang}:
    ---
    {text}
    ---
    Only return the translated text, no explanation.
    """
    try:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# --- Text-to-Speech Utility ---
def text_to_speech(text: str, lang: str) -> bytes:
    """
    Convert text to speech audio using gTTS.
    Returns audio bytes (mp3).
    """
    try:
        tts = gTTS(text=text, lang=lang)
        from io import BytesIO
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read()
    except Exception as e:
        print(f"TTS error: {e}")
        return b''

# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    client = GeminiClient()
    
    print("--- Testing Single Document Analysis ---")
    sample_text = "This agreement has a lock-in period of 6 months. If the Tenant vacates before this period, the security deposit shall be forfeited."
    analysis = client.analyze_document(sample_text)
    print(json.dumps(analysis, indent=2))

    print("\n--- Testing Q&A ---")
    question = "What happens if I leave early?"
    answer = client.answer_question(sample_text, question)
    print(f"Q: {question}\nA: {answer}")

    print("\n--- Testing Document Comparison ---")
    old_contract = "The notice period for termination is 30 days."
    new_contract = "The notice period for termination is 60 days. The tenant must also pay a penalty of one month's rent if terminating before the full lease term."
    comparison = client.compare_documents(old_contract, new_contract)
    print(json.dumps(comparison, indent=2))
