# AI Policy Document Comparator

A Streamlit web application that compares two AI policy documents using NLP and LLM analysis.

##  Features
- Upload PDF, DOCX, or TXT policy documents
- AI-generated summaries via Groq (Llama 3)
- Common themes with side-by-side comparison
- Key regulatory differences highlighted
- TF-IDF keyword extraction
- Word clouds, radar charts, heatmaps
- Named entity recognition (laws, orgs, locations)
- Deep comparative insights

## Tech Stack
- **Frontend:** Streamlit
- **LLM:** Groq API (Llama 3.3-70b)
- **NLP:** spaCy, NLTK, scikit-learn
- **Visualization:** Plotly, Matplotlib, WordCloud
- **PDF Extraction:** PyMuPDF

## Setup & Run Locally

1. Clone the repository
```bash
   git clone https://github.com/YOUR_USERNAME/ai-policy-comparator.git
   cd ai-policy-comparator
```

2. Install dependencies
```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

3. Add your Groq API key
```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
```

4. Run the app
```bash
   streamlit run app.py
```
## API Key Setup
This app uses the **Groq API** which is completely free.

1. Go to https://console.groq.com
2. Sign up (no credit card needed)
3. Create an API key
4. Either:
   - Paste it in the sidebar when the app opens, OR
   - Create a `.env` file: `GROQ_API_KEY=your_key_here`

##  Sample Documents Tested
- EU AI Act (2024)
- India National AI Strategy (NITI Aayog)

## Get a Free Groq API Key
Visit https://console.groq.com to get a free API key.