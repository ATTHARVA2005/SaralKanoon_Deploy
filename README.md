# ⚖️ SaralKanoon - Legal Document Analyzer

**SaralKanoon** is an intelligent legal document analysis tool that simplifies the complexity of legal documents.  
It provides **summaries**, **key clause explanations**, and **red flag detection** – helping users quickly understand contracts, agreements, and more.  

Supports both **digital PDFs** and **scanned documents** with advanced text extraction & AI-powered insights.

## ✨Features

- 🧠 **Smart Document Analysis**: Analyzes legal documents and provides easy-to-understand summaries
- 📑 **Key Clause Extraction**: Identifies and explains important clauses in simple terms
- 🚩 **Red Flag Detection**: Highlights potentially risky or unfavorable clauses
- 📂 **Advanced PDF Processing**: 
  - Supports both digital and scanned documents
  - Parallel processing for faster analysis
  - Optimized image handling for better performance
-💬 **Interactive Q&A**: Ask questions about your document and get contextual answers
-🔍 **Document Comparison**: Compare different versions of legal documents to identify changes

## 🛠️Tech Stack

### Frontend
-⚛️ React with TypeScript
-⚡ Vite for build tooling
-🎨 TailwindCSS for styling
-🌐 Axios for API communication

### Backend
-🐍 Python with Flask
-🤖 Google Gemini API for AI analysis
-📄 PyMuPDF for PDF processing
-⚡ Concurrent processing for improved performance

## 🚀 Getting Started

### ✅Prerequisites
- Python 3
- Node.js and npm
- Google Gemini API key

### 📥Installation

1. Clone the repository:
```bash
git clone https://github.com/Anubhab19518/legal-document-analyzer.git
cd legal-document-analyzer
```

2. Set up the backend:
```bash
cd backend
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file in the backend directory with:
```
GEMINI_API_KEY=your_api_key_here
```

4. Set up the frontend:
```bash
cd frontend
npm install
```

### ▶️Running the Application

1. Start the backend server:
```bash
cd backend
python app.py
```

2. Start the frontend development server:
```bash
cd frontend
npm run dev
```

3. Access the application at `http://localhost:5173`

## 📖Usage

1. Upload a legal document (PDF format)
2. Wait for the analysis to complete
3. Review the simplified summary, key clauses, and potential red flags
4. Use the Q&A feature to ask specific questions about the document

## 🔍Features in Detail

### 📊Document Analysis
- Provides a concise summary of the document's main purpose
- Identifies 3-5 most important clauses with explanations
- Highlights potentially unfavorable terms or conditions

### 📄PDF Processing
- Efficient parallel processing for faster analysis
- Supports both text-based and scanned documents
- Optimized image handling for better performance

### 💬Interactive Q&A
- Ask questions about specific parts of the document
- Get contextual answers based on document content
- Natural language understanding for better responses

## 👥Team
1. Anubhab Das : Fullstack Developer
2. Attharva Gupta : Backend Developer
3. Tamogno Roy : Frontend Developer
4. Anuska Basak : Full Stack Developer
## 🙏Acknowledgments

- Google Gemini API for powering the AI analysis
- PyMuPDF for PDF processing capabilities
- React and Vite communities for excellent development tools
