# AI Policy Document Comparator

A Streamlit web application that compares two AI policy documents using NLP and LLM analysis.

## Live Demo
[Try it here](https://ai-policy-comparator-agbd2qwmevavsgylv7dlvu.streamlit.app/)

## What It Does
Upload any two AI policy documents (PDF, DOCX, or TXT) and instantly get:
- **Policy Summaries** — structured breakdown of each document
- **Common Themes** — shared regulatory priorities side by side
- **Key Differences** — how the two approaches diverge and why it matters
- **Keywords & Concepts** — word clouds, TF-IDF charts, policy concept frequency heatmaps
- **Deep Insights** — philosophical stance, geopolitical context, regulatory maturity analysis

## Tech Stack
- **Frontend:** Streamlit
- **LLM:** Groq API (Llama 3.3-70b) — free tier
- **NLP:** NLTK, scikit-learn (TF-IDF)
- **Visualization:** Plotly, Matplotlib, WordCloud
- **PDF Extraction:** PyMuPDF

## Sample Documents Tested
- EU AI Act (2024) — 593,000 chars
- India National AI Strategy — NITI Aayog — 278,000 chars
- Similarity Score: 60/100 — same topic, very different regulatory philosophies

## Run Locally

1. Clone the repository
```bash
   git clone https://github.com/sneha060405/ai-policy-comparator.git
   cd ai-policy-comparator
```

2. Install dependencies
```bash
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"
```

3. Add your Groq API key (free at https://console.groq.com)
```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
```

4. Run the app
```bash
   streamlit run app.py
```

## API Key
The hosted version requires no setup — just open the link and upload documents.

To run locally, get a free Groq API key at https://console.groq.com (no credit card needed) and add it to your `.env` file.