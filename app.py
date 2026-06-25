import streamlit as st
import os
from dotenv import load_dotenv
from core.extractor import extract_text
from core.comparator import run_full_comparison
from utils.visualizer import (
    make_wordcloud, make_keyword_bar, make_concepts_radar,
    make_similarity_gauge, make_concepts_heatmap
)

load_dotenv()

st.set_page_config(
    page_title="AI Policy Comparator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .theme-card {
        background: #f8f9ff;
        border-left: 4px solid #3949ab;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .diff-card {
        background: #fff8f8;
        border-left: 4px solid #e53935;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .doc1-color { color: #4A90D9; font-weight: 600; }
    .doc2-color { color: #E8534A; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>⚖️ AI Policy Document Comparator</h1>
    <p style="margin:0; opacity:0.85; font-size:1.1rem">
        Upload two AI policy documents to generate summaries, compare themes,
        and extract key regulatory concepts using NLP + Groq AI (Llama 3)
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 How to Use")
    st.info(
        "1. Upload **two policy documents** (PDF, DOCX, or TXT)\n"
        "2. Give each document a name\n"
        "3. Click **Run Comparison**\n"
        "4. Explore results across the tabs"
    )
    st.markdown("---")
    st.markdown("### 📄 Supported Formats")
    st.markdown("• PDF (`.pdf`)\n• Word Document (`.docx`)\n• Plain Text (`.txt`)")
    st.markdown("---")
    st.markdown("### ⚙️ Groq API Key")
    api_key_input = st.text_input(
        "Groq API Key (Free)", type="password",
        help="Get your free key at console.groq.com",
        value=os.getenv("GROQ_API_KEY", "")
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

# ── File Upload ───────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📄 Document 1")
    doc1_name = st.text_input("Document 1 Name", value="EU AI Act", key="name1")
    file1 = st.file_uploader("Upload Document 1", type=["pdf", "docx", "txt"], key="file1")

with col2:
    st.markdown("#### 📄 Document 2")
    doc2_name = st.text_input("Document 2 Name", value="India AI Mission", key="name2")
    file2 = st.file_uploader("Upload Document 2", type=["pdf", "docx", "txt"], key="file2")

# ── Run Button ────────────────────────────────────────────────────────────────
run_btn = st.button(
    "🚀 Run Comparison",
    type="primary",
    use_container_width=True,
    disabled=(file1 is None or file2 is None)
)

if file1 is None or file2 is None:
    st.caption("⬆️ Upload both documents to enable comparison.")

if run_btn and file1 and file2:
    if not os.getenv("GROQ_API_KEY"):
        st.error("❌ Please provide your Groq API key in the sidebar. Get one free at console.groq.com")
        st.stop()

    with st.spinner("📖 Extracting text from documents..."):
        try:
            text1 = extract_text(file1)
            text2 = extract_text(file2)
            st.success(f"✅ Extracted {len(text1):,} chars from {doc1_name} | {len(text2):,} chars from {doc2_name}")
        except Exception as e:
            st.error(f"❌ Extraction error: {e}")
            st.stop()

    with st.spinner("🤖 Analyzing documents with Groq AI (Llama 3) + NLP... (this may take 30-60 seconds)"):
        progress = st.progress(0, text="Starting analysis...")
        try:
            progress.progress(10, "Generating summaries...")
            results = run_full_comparison(text1, text2, doc1_name, doc2_name)
            progress.progress(100, "Complete!")
        except Exception as e:
            st.error(f"❌ Analysis error: {e}")
            st.stop()

    st.success("✅ Analysis complete! Explore the results below.")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Summaries",
        "🔗 Common Themes",
        "⚡ Key Differences",
        "🔍 Keywords & Concepts",
        "💡 Deep Insights"
    ])

    # ── TAB 1: Summaries ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("## 📋 Policy Summaries")
        score = results['themes_and_diffs'].get('similarity_score', 50)
        st.plotly_chart(make_similarity_gauge(score), use_container_width=True)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"### 🔵 {doc1_name}")
            st.markdown(f'<div class="section-card">{results["summary1"]}</div>', unsafe_allow_html=True)
        with col_s2:
            st.markdown(f"### 🔴 {doc2_name}")
            st.markdown(f'<div class="section-card">{results["summary2"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 💪 Unique Strengths")
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            st.markdown(f"**{doc1_name}**")
            for s in results['themes_and_diffs'].get('doc1_unique_strengths', []):
                st.markdown(f"✅ {s}")
        with col_u2:
            st.markdown(f"**{doc2_name}**")
            for s in results['themes_and_diffs'].get('doc2_unique_strengths', []):
                st.markdown(f"✅ {s}")

    # ── TAB 2: Common Themes ──────────────────────────────────────────────────
    with tab2:
        st.markdown("## 🔗 Common Themes")
        themes = results['themes_and_diffs'].get('common_themes', [])

        if themes:
            for i, theme in enumerate(themes, 1):
                with st.expander(f"**{i}. {theme.get('theme', 'Theme')}**", expanded=(i <= 3)):
                    st.markdown(f"**Overview:** {theme.get('description', '')}")
                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        st.markdown(f'<div class="theme-card"><b class="doc1-color">🔵 {doc1_name}</b><br>{theme.get("doc1_approach", "N/A")}</div>', unsafe_allow_html=True)
                    with col_t2:
                        st.markdown(f'<div class="theme-card"><b class="doc2-color">🔴 {doc2_name}</b><br>{theme.get("doc2_approach", "N/A")}</div>', unsafe_allow_html=True)
        else:
            st.warning("No common themes extracted.")

        st.markdown("---")
        st.markdown("### 🌡️ Policy Concept Coverage")
        fig_heat = make_concepts_heatmap(results['concepts1'], results['concepts2'], doc1_name, doc2_name)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── TAB 3: Key Differences ────────────────────────────────────────────────
    with tab3:
        st.markdown("## ⚡ Key Differences")
        diffs = results['themes_and_diffs'].get('key_differences', [])

        if diffs:
            for i, diff in enumerate(diffs, 1):
                st.markdown(f"### {i}. {diff.get('aspect', 'Aspect')}")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown(f'<div class="diff-card"><b class="doc1-color">🔵 {doc1_name}</b><br>{diff.get("doc1", "N/A")}</div>', unsafe_allow_html=True)
                with col_d2:
                    st.markdown(f'<div class="diff-card"><b class="doc2-color">🔴 {doc2_name}</b><br>{diff.get("doc2", "N/A")}</div>', unsafe_allow_html=True)
                st.caption(f"📌 **Why it matters:** {diff.get('significance', '')}")
                st.markdown("---")
        else:
            st.warning("No differences extracted.")

        st.markdown("### 🕸️ Concept Coverage Radar")
        fig_radar = make_concepts_radar(results['concepts1'], results['concepts2'], doc1_name, doc2_name)
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── TAB 4: Keywords & Concepts ────────────────────────────────────────────
    with tab4:
        st.markdown("## 🔍 Keywords & Policy Concepts")

        st.markdown("### ☁️ Word Clouds")
        col_wc1, col_wc2 = st.columns(2)
        with col_wc1:
            fig_wc1 = make_wordcloud(results['wordfreq1'], f"{doc1_name} — Word Cloud", 'Blues')
            st.pyplot(fig_wc1)
        with col_wc2:
            fig_wc2 = make_wordcloud(results['wordfreq2'], f"{doc2_name} — Word Cloud", 'Reds')
            st.pyplot(fig_wc2)

        st.markdown("---")
        st.markdown("### 📊 Top TF-IDF Keywords")
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            fig_k1 = make_keyword_bar(results['keywords']['doc1_keywords'], f"Top Keywords — {doc1_name}", '#4A90D9')
            st.plotly_chart(fig_k1, use_container_width=True)
        with col_k2:
            fig_k2 = make_keyword_bar(results['keywords']['doc2_keywords'], f"Top Keywords — {doc2_name}", '#E8534A')
            st.plotly_chart(fig_k2, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📌 AI Policy Concept Frequency")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown(f"**{doc1_name}**")
            if results['concepts1']:
                for concept, count in list(results['concepts1'].items())[:15]:
                    st.markdown(f"- `{concept}`: **{count}** mentions")
            else:
                st.caption("No predefined concepts found.")
        with col_c2:
            st.markdown(f"**{doc2_name}**")
            if results['concepts2']:
                for concept, count in list(results['concepts2'].items())[:15]:
                    st.markdown(f"- `{concept}`: **{count}** mentions")
            else:
                st.caption("No predefined concepts found.")

        st.markdown("---")
        st.markdown("### 🏛️ Named Entities (Organizations, Laws, Locations)")
        col_e1, col_e2 = st.columns(2)
        for col, entities, name in [(col_e1, results['entities1'], doc1_name), (col_e2, results['entities2'], doc2_name)]:
            with col:
                st.markdown(f"**{name}**")
                if entities.get('organizations'):
                    st.markdown("**Organizations:** " + ", ".join(entities['organizations'][:8]))
                if entities.get('laws'):
                    st.markdown("**Laws/Regulations:** " + ", ".join(entities['laws'][:8]))
                if entities.get('locations'):
                    st.markdown("**Locations:** " + ", ".join(entities['locations'][:8]))

    # ── TAB 5: Deep Insights ──────────────────────────────────────────────────
    with tab5:
        st.markdown("## 💡 Deep Comparative Insights")
        st.markdown(f'<div class="section-card">{results["insights"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📈 Document Statistics")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric(f"{doc1_name} Length", f"{len(text1):,} chars")
        with col_stat2:
            st.metric(f"{doc2_name} Length", f"{len(text2):,} chars")
        with col_stat3:
            st.metric("Similarity Score", f"{score}/100")
        with col_stat4:
            st.metric("Common Themes Found", len(results['themes_and_diffs'].get('common_themes', [])))

st.markdown("---")
st.caption("⚖️ AI Policy Comparator · Powered by Groq AI (Llama 3) · NLP with spaCy & scikit-learn")