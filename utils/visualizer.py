import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io

def make_wordcloud(word_freq: dict, title: str, colormap: str = 'Blues') -> plt.Figure:
    """Generate a word cloud from frequency dict."""
    if not word_freq:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return fig

    wc = WordCloud(
        width=800, height=400,
        background_color='white',
        colormap=colormap,
        max_words=80,
        prefer_horizontal=0.8
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    return fig

def make_keyword_bar(keywords: list[tuple], title: str, color: str = '#4A90D9') -> go.Figure:
    """Bar chart of top TF-IDF keywords."""
    if not keywords:
        return go.Figure()

    df = pd.DataFrame(keywords, columns=['Keyword', 'Score'])
    df = df.head(15).sort_values('Score')

    fig = px.bar(
        df, x='Score', y='Keyword',
        orientation='h',
        title=title,
        color_discrete_sequence=[color]
    )
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis_title="",
        xaxis_title="TF-IDF Score",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def make_concepts_radar(concepts1: dict, concepts2: dict, name1: str, name2: str) -> go.Figure:
    """Radar chart comparing policy concept coverage."""
    # Find top shared and important concepts
    all_concepts = list(set(list(concepts1.keys()) + list(concepts2.keys())))
    top_concepts = sorted(
        all_concepts,
        key=lambda c: concepts1.get(c, 0) + concepts2.get(c, 0),
        reverse=True
    )[:10]

    vals1 = [concepts1.get(c, 0) for c in top_concepts]
    vals2 = [concepts2.get(c, 0) for c in top_concepts]

    # Normalize to 0-100
    max_val = max(max(vals1 + vals2, default=1), 1)
    vals1_norm = [v / max_val * 100 for v in vals1]
    vals2_norm = [v / max_val * 100 for v in vals2]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals1_norm + [vals1_norm[0]],
        theta=top_concepts + [top_concepts[0]],
        fill='toself',
        name=name1,
        line_color='#4A90D9',
        fillcolor='rgba(74,144,217,0.2)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals2_norm + [vals2_norm[0]],
        theta=top_concepts + [top_concepts[0]],
        fill='toself',
        name=name2,
        line_color='#E8534A',
        fillcolor='rgba(232,83,74,0.2)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Policy Concept Coverage Comparison",
        height=500
    )
    return fig

def make_similarity_gauge(score: int) -> go.Figure:
    """Gauge chart showing similarity score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Document Similarity Score", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4A90D9"},
            'steps': [
                {'range': [0, 30], 'color': "#FFE5E5"},
                {'range': [30, 60], 'color': "#FFF3CD"},
                {'range': [60, 100], 'color': "#D4EDDA"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=10))
    return fig

def make_concepts_heatmap(concepts1: dict, concepts2: dict, name1: str, name2: str) -> go.Figure:
    """Heatmap of top concept frequencies side by side."""
    all_concepts = list(set(list(concepts1.keys()) + list(concepts2.keys())))
    top = sorted(all_concepts, key=lambda c: concepts1.get(c, 0) + concepts2.get(c, 0), reverse=True)[:15]

    z = [[concepts1.get(c, 0) for c in top], [concepts2.get(c, 0) for c in top]]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=top,
        y=[name1, name2],
        colorscale='Blues',
        text=z,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    fig.update_layout(
        title="Policy Concept Frequency Heatmap",
        height=280,
        margin=dict(t=50, b=80),
        xaxis_tickangle=-35
    )
    return fig