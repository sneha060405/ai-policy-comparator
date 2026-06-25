import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import spacy
from spacy.cli import download as spacy_download

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Load spaCy model - auto download if not present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

POLICY_CONCEPTS = [
    "risk assessment", "algorithmic accountability", "transparency",
    "explainability", "fairness", "bias", "data protection", "privacy",
    "human oversight", "fundamental rights", "safety", "security",
    "innovation", "regulatory sandbox", "conformity assessment",
    "high-risk AI", "prohibited AI", "general purpose AI", "foundation model",
    "governance", "liability", "enforcement", "compliance", "audit",
    "trustworthy AI", "ethics", "accountability", "sustainability",
    "digital infrastructure", "capacity building", "public sector",
    "healthcare AI", "autonomous systems", "facial recognition",
    "natural language processing", "machine learning", "deep learning"
]

STOP_WORDS = set(stopwords.words('english')).union({
    'shall', 'may', 'also', 'within', 'including', 'provide', 'ensure',
    'pursuant', 'regard', 'relevant', 'accordance', 'without', 'therefore',
    'however', 'thus', 'whereas', 'herein', 'thereof', 'article', 'section'
})

def extract_keywords_tfidf(text1: str, text2: str, top_n: int = 30) -> dict:
    """Extract top keywords from each document using TF-IDF."""
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words=list(STOP_WORDS),
        ngram_range=(1, 2),
        min_df=1
    )

    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        feature_names = vectorizer.get_feature_names_out()

        doc1_scores = dict(zip(feature_names, tfidf_matrix[0].toarray()[0]))
        doc2_scores = dict(zip(feature_names, tfidf_matrix[1].toarray()[0]))

        doc1_top = sorted(doc1_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        doc2_top = sorted(doc2_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        return {
            "doc1_keywords": [(kw, round(score * 100, 2)) for kw, score in doc1_top],
            "doc2_keywords": [(kw, round(score * 100, 2)) for kw, score in doc2_top]
        }
    except Exception as e:
        return {"doc1_keywords": [], "doc2_keywords": [], "error": str(e)}

def extract_policy_concepts(text: str) -> dict:
    """Count occurrences of known AI policy concepts."""
    text_lower = text.lower()
    concept_counts = {}
    for concept in POLICY_CONCEPTS:
        count = len(re.findall(r'\b' + re.escape(concept) + r'\b', text_lower))
        if count > 0:
            concept_counts[concept] = count
    return dict(sorted(concept_counts.items(), key=lambda x: x[1], reverse=True))

def extract_named_entities(text: str) -> dict:
    """Extract named entities (orgs, laws, places) using spaCy."""
    # Limit to first 10000 chars for speed
    doc = nlp(text[:10000])
    entities = {"organizations": [], "laws": [], "locations": [], "others": []}

    seen = set()
    for ent in doc.ents:
        if ent.text.lower() in seen or len(ent.text) < 3:
            continue
        seen.add(ent.text.lower())

        if ent.label_ in ("ORG", "PERSON"):
            entities["organizations"].append(ent.text)
        elif ent.label_ in ("LAW",):
            entities["laws"].append(ent.text)
        elif ent.label_ in ("GPE", "LOC"):
            entities["locations"].append(ent.text)
        else:
            entities["others"].append(ent.text)

    return {k: list(set(v))[:15] for k, v in entities.items()}

def get_word_frequency(text: str, top_n: int = 100) -> dict:
    """Get word frequency for word cloud generation."""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in STOP_WORDS]
    return dict(Counter(filtered).most_common(top_n))