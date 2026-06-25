import os
from groq import Groq

def summarize_document(text: str, doc_name: str) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    excerpt = text[:12000] if len(text) > 12000 else text

    prompt = f"""You are an expert AI policy analyst. Analyze this AI policy document titled "{doc_name}" and provide a structured summary.

Document Content:
{excerpt}

Provide a comprehensive summary with these sections:
1. **Overview** (2-3 sentences on what this document is and its purpose)
2. **Key Objectives** (bullet points of main goals)
3. **Regulatory Approach** (how it plans to govern AI - risk-based, sector-specific, etc.)
4. **Target Sectors/Applications** (which AI domains it covers)
5. **Enforcement & Compliance** (penalties, bodies, mechanisms mentioned)
6. **Innovation vs. Safety Balance** (how it balances promoting AI vs. protecting citizens)

Be concise but thorough. Use bullet points where appropriate."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_insights(text1: str, text2: str, name1: str, name2: str) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    excerpt1 = text1[:6000]
    excerpt2 = text2[:6000]

    prompt = f"""You are an expert AI policy researcher. Compare these two AI policy documents and generate deep comparative insights.

Document 1 - {name1}:
{excerpt1}

Document 2 - {name2}:
{excerpt2}

Provide the following analysis:

## 🎯 Philosophical Stance
Compare the fundamental philosophy of each document toward AI (precautionary vs. permissive, innovation-first vs. safety-first).

## 🌍 Geopolitical Context
How does each document reflect the country/region's strategic position in global AI competition?

## ⚖️ Regulatory Maturity
Which document appears more mature/comprehensive? What gaps exist in each?

## 🔮 Future Implications
What are the potential real-world impacts of each approach on AI development and deployment?

## 💡 What Each Can Learn from the Other
Specific recommendations for improvement based on the comparison.

Keep each section to 3-5 sentences. Be analytical and substantive."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content