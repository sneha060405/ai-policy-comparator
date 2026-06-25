import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

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
    text_lower = text.lower()
    concept_counts = {}
    for concept in POLICY_CONCEPTS:
        count = len(re.findall(r'\b' + re.escape(concept) + r'\b', text_lower))
        if count > 0:
            concept_counts[concept] = count
    return dict(sorted(concept_counts.items(), key=lambda x: x[1], reverse=True))

def extract_named_entities(text: str) -> dict:
    """Extract named entities using NLTK instead of spaCy."""
    entities = {"organizations": [], "laws": [], "locations": [], "others": []}
    try:
        tokens = nltk.word_tokenize(text[:10000])
        tagged = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(tagged, binary=False)
        seen = set()
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                name = ' '.join(c[0] for c in chunk)
                if name.lower() in seen or len(name) < 3:
                    continue
                seen.add(name.lower())
                label = chunk.label()
                if label in ('ORGANIZATION',):
                    entities['organizations'].append(name)
                elif label in ('GPE', 'LOCATION'):
                    entities['locations'].append(name)
                elif label in ('PERSON',):
                    entities['others'].append(name)
    except Exception:
        pass
    return {k: list(set(v))[:15] for k, v in entities.items()}

def get_word_frequency(text: str, top_n: int = 100) -> dict:
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in STOP_WORDS]
    return dict(Counter(filtered).most_common(top_n))