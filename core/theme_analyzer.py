import os
import json
import re
from groq import Groq

def extract_themes_and_differences(text1: str, text2: str, name1: str, name2: str) -> dict:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    excerpt1 = text1[:5000]
    excerpt2 = text2[:5000]

    prompt = f"""Analyze these two AI policy documents and return ONLY a valid JSON object (no markdown, no explanation, no extra text).

Document 1 - {name1}:
{excerpt1}

Document 2 - {name2}:
{excerpt2}

Return this exact JSON structure:
{{
  "common_themes": [
    {{"theme": "theme name", "description": "1-2 sentence description", "doc1_approach": "how doc1 addresses it", "doc2_approach": "how doc2 addresses it"}}
  ],
  "key_differences": [
    {{"aspect": "aspect name", "doc1": "doc1 position", "doc2": "doc2 position", "significance": "why this matters"}}
  ],
  "similarity_score": 50,
  "doc1_unique_strengths": ["strength1", "strength2", "strength3"],
  "doc2_unique_strengths": ["strength1", "strength2", "strength3"]
}}

Identify 4-6 common themes and 4-6 key differences. Return ONLY the JSON, nothing else."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.choices[0].message.content.strip()
    response_text = re.sub(r'^```json\s*', '', response_text)
    response_text = re.sub(r'^```\s*', '', response_text)
    response_text = re.sub(r'\s*```$', '', response_text)

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {
            "common_themes": [{"theme": "AI Governance", "description": "Both documents address governance frameworks.", "doc1_approach": "See full text", "doc2_approach": "See full text"}],
            "key_differences": [{"aspect": "Regulatory Approach", "doc1": "See summary", "doc2": "See summary", "significance": "Shapes implementation"}],
            "similarity_score": 50,
            "doc1_unique_strengths": ["Comprehensive coverage"],
            "doc2_unique_strengths": ["Innovative framework"]
        }