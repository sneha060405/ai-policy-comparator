from core.summarizer import summarize_document, generate_insights
from core.theme_analyzer import extract_themes_and_differences
from core.keyword_extractor import (
    extract_keywords_tfidf,
    extract_policy_concepts,
    extract_named_entities,
    get_word_frequency
)

def run_full_comparison(text1: str, text2: str, name1: str, name2: str) -> dict:
    """
    Orchestrate the full document comparison pipeline.
    Returns all analysis results as a structured dict.
    """
    results = {}

    # 1. Summaries
    results['summary1'] = summarize_document(text1, name1)
    results['summary2'] = summarize_document(text2, name2)

    # 2. Themes and differences
    results['themes_and_diffs'] = extract_themes_and_differences(text1, text2, name1, name2)

    # 3. Comparative insights
    results['insights'] = generate_insights(text1, text2, name1, name2)

    # 4. Keywords
    results['keywords'] = extract_keywords_tfidf(text1, text2)

    # 5. Policy concepts per document
    results['concepts1'] = extract_policy_concepts(text1)
    results['concepts2'] = extract_policy_concepts(text2)

    # 6. Named entities
    results['entities1'] = extract_named_entities(text1)
    results['entities2'] = extract_named_entities(text2)

    # 7. Word frequencies for word clouds
    results['wordfreq1'] = get_word_frequency(text1)
    results['wordfreq2'] = get_word_frequency(text2)

    return results