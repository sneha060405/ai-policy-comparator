import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """Remove noise from extracted text."""
    # Remove excessive whitespace and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Remove page numbers like "Page 1 of 20"
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove special unicode chars
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.strip()

def remove_stopwords(text: str) -> str:
    """Remove stopwords for keyword extraction."""
    words = nltk.word_tokenize(text.lower())
    return ' '.join([w for w in words if w.isalpha() and w not in STOP_WORDS])

def chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    """Split long text into chunks for API processing."""
    paragraphs = text.split('\n\n')
    chunks, current = [], ''
    for para in paragraphs:
        if len(current) + len(para) < max_chars:
            current += para + '\n\n'
        else:
            if current:
                chunks.append(current.strip())
            current = para + '\n\n'
    if current:
        chunks.append(current.strip())
    return chunks