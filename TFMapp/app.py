# -*- coding: utf-8 -*-
"""
Streamlit App with BERT Sentiment Analysis
"""

import os
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
import feedparser
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import sys, os
sys.path.append(os.path.dirname(__file__))  # añade TFMapp al sys.path si hiciera falta

#  ---- Importación de modelo
try:
    from TFMapp.utils_nlp import load_model_hf  # cuando se ejecuta desde la raíz del repo (Streamlit Cloud)
except ModuleNotFoundError:
    from utils_nlp import load_model_hf          # cuando ejecutas dentro de TFMapp localmente

@st.cache_resource(show_spinner="Cargando modelo...")
def load_model_cached():
    return load_model_hf("iher9812/sentiment_model_full_offline")


# Importa la página de Kaggle desde tu módulo separado (requiere tokenizer, model)
from kaggle_page import render_kaggle_page

from utils_nlp import (
    clean_for_model,
    limpiar_texto_dataset,
    predict_sentiment_batch,
)

# ---- Optional: import from News_DATA with safe fallbacks (solo para Noticias)
try:
    from News_DATA import get_airline_news_clean, AIRLINE_FEEDS
except Exception:
    get_airline_news_clean = None
    AIRLINE_FEEDS = None

# --- 1) PAGE CONFIG ---
st.set_page_config(
    page_title="Sentiment Analysis with BERT",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- 2) STYLES ---
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #E0E0E0; }
    .option-card {
        background-color: #1a1a1a; border: 1px solid #333; padding: 2rem;
        border-radius: 10px; text-align: center; transition: all .3s; cursor: pointer;
    }
    .option-card:hover { background-color: #2a2a2a; border-color: #3b82f6; }
    .option-card h2 { font-size: 1.5rem; color: #fff; }
    .option-card p { color: #aaa; }
    .summary-box {
        background-color: #1a1a1a; border: 1px solid #333; padding: 20px;
        border-radius: 10px; text-align: center; height: 100%;
    }
    .summary-box .example { font-size: .9em; color: #888; font-style: italic; min-height: 50px; }
    hr { border-top: 1px solid #333; }
</style>
""", unsafe_allow_html=True)


# --- 4) NEWS HELPERS ---
@st.cache_data(ttl=600)
def fetch_news(feed_url):
    return feedparser.parse(feed_url).entries

def get_default_feeds():
    return {
        "United Airlines": "https://www.bing.com/news/search?q=united+airlines&cc=us&qft=interval%3d%228%22&format=rss",
        "Southwest Airlines": "https://www.bing.com/news/search?q=Southwest+Airlines&cc=us&qft=interval%3d%229%22&format=rss",
        "Delta Air Lines": "https://www.bing.com/news/search?q=delta+airlines&cc=us&qft=interval%3d%229%22&format=rss",
        "JetBlue Airways": "https://www.bing.com/news/search?q=JetBlue+Airways&cc=us&qft=interval%3d%228%22&format=rss",
        "American Airlines": "https://www.bing.com/news/search?q=American+Airlines&cc=us&qft=interval%3d%228%22&format=rss",
    }

# --- 5) PAGES ---
def render_home_page():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/IMF_Smart_Education.png/1200px-IMF_Smart_Education.png")
    with col2:
        st.title("Sentiment Analysis with BERT Model")

    st.markdown("<p style='text-align:center;color:#AAAAAA;'>Choose how you want to analyze sentiment.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Fila 1: CSV y News
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            '<div class="option-card"><h2>1. Upload a CSV File</h2>'
            '<p>Analyze sentiment from your own dataset.</p></div>',
            unsafe_allow_html=True
        )
        if st.button("Upload CSV", key="btn_upload_csv", use_container_width=True):
            st.session_state.page = 'upload_csv'
            st.rerun()

    with c2:
        st.markdown(
            '<div class="option-card"><h2>2. Analyze Live News</h2>'
            '<p>Fetch and analyze the latest news headlines for major airlines.</p></div>',
            unsafe_allow_html=True
        )
        if st.button("Analyze News", key="btn_news", use_container_width=True):
            st.session_state.page = 'view_news'
            st.rerun()

    # Fila 2: Kaggle centrado
    st.markdown("---")
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown(
            '<div class="option-card"><h2>3. Analyze Kaggle Tweets</h2>'
            '<p>Use the Crowdflower dataset to compare sentiment across airlines.</p></div>',
            unsafe_allow_html=True
        )
        if st.button("Analyze Kaggle Dataset", key="btn_kaggle", use_container_width=True):
            st.session_state.page = 'kaggle'
            st.rerun()

def render_upload_page():
    st.title("Upload Your Dataset")
    if st.button("← Back"):
        st.session_state.page = 'home'
        st.rerun()

    uploaded_file = st.file_uploader(
        "Upload a CSV file. Make sure it has a column with text to analyze.",
        type=['csv']
    )

    if uploaded_file:
        with st.spinner("Reading file..."):
            try:
                try:
                    df = pd.read_csv(uploaded_file)
                except UnicodeDecodeError:
                    df = pd.read_csv(uploaded_file, encoding="latin-1")
                st.session_state.raw_df = df
                st.session_state.page = 'column_select'
                st.rerun()
            except Exception as e:
                st.error(f"Failed to read the CSV file: {e}")

def render_column_selector_page():
    st.title("Select the Column for Analysis")
    if st.button("← Back"):
        st.session_state.page = 'upload_csv'
        if 'raw_df' in st.session_state:
            del st.session_state.raw_df
        st.rerun()

    df = st.session_state.get('raw_df')
    if df is None or df.empty:
        st.warning("No data loaded. Please upload a CSV first.")
        return

    st.write("Data Preview:")
    st.dataframe(df.head())

    text_column = st.selectbox("Which column contains the text you want to analyze?", df.columns)
    apply_dataset_clean = st.checkbox("Add a cleaned text column ('clean_title') to the dataset", value=True)

    if st.button("Analyze Data", use_container_width=True):
        data = df.copy()
        if apply_dataset_clean and text_column:
            data['clean_title'] = data[text_column].astype(str).apply(limpiar_texto_dataset)
        st.session_state.data_to_analyze = data
        st.session_state.text_column = text_column
        st.session_state.page = 'analysis'
        st.rerun()

def render_view_news_page():
    st.title("Analyze Live Airline News")
    if st.button("← Back"):
        st.session_state.page = 'home'
        st.rerun()

    feeds = AIRLINE_FEEDS if isinstance(AIRLINE_FEEDS, dict) and AIRLINE_FEEDS else get_default_feeds()
    selected_airline = st.selectbox("Select an airline to see its news:", options=list(feeds.keys()))

    if not selected_airline:
        return

    with st.spinner(f"Fetching news for {selected_airline}..."):
        entries = fetch_news(feeds[selected_airline])

    if not entries:
        st.warning("Could not fetch news. The feed might be temporarily unavailable.")
        return

    st.subheader(f"Latest Headlines for {selected_airline}")
    for entry in entries:
        title = getattr(entry, 'title', 'Untitled')
        with st.expander(title):
            summary = getattr(entry, 'summary', 'No summary available.')
            st.markdown(summary, unsafe_allow_html=True)
            st.markdown(f"**Published:** {getattr(entry, 'published', 'N/A')}")
            st.markdown(f"**Link:** [Read full article]({getattr(entry, 'link','')})", unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Analyze This News Data", use_container_width=True):
        try:
            if callable(get_airline_news_clean):
                # Usa tu pipeline (recomendado)
                with st.spinner(f"Running News_DATA pipeline for {selected_airline}..."):
                    df_clean, csv_path = get_airline_news_clean(selected_airline)
                text_col = 'text' if 'text' in df_clean.columns else 'title'
# Asegura 'clean_title' para mostrar el toggle en análisis
                if 'clean_title' not in df_clean.columns and text_col in df_clean.columns:
                    df_clean['clean_title'] = df_clean[text_col].astype(str).apply(limpiar_texto_dataset)
                
                st.success(f"CSV cleaned and saved: {csv_path}")
                st.session_state.data_to_analyze = df_clean
                st.session_state.text_column = text_col
                
            else:
                # Fallback simple si no está News_DATA.py
                records = []
                for e in entries:
                    published_dt = None
                    try:
                        if getattr(e, 'published_parsed', None):
                            published_dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc).astimezone()
                        else:
                            published_dt = pd.to_datetime(getattr(e, 'published', None), errors='coerce')
                    except Exception:
                        published_dt = pd.NaT
                    title = getattr(e, 'title', '')
                    records.append({
                        'airline': selected_airline,
                        'title': title,
                        'clean_title': limpiar_texto_dataset(title),
                        'published_at': published_dt,
                        'url': getattr(e, 'link', '')
                    })
                df_clean = pd.DataFrame(records)
                df_clean.drop_duplicates(subset=['airline','title'], inplace=True)
                df_clean.dropna(subset=['title','url'], inplace=True)
                df_clean = df_clean[df_clean['clean_title'].str.len() > 10]
                df_clean.sort_values('published_at', ascending=False, inplace=True)

                st.session_state.data_to_analyze = df_clean
                st.session_state.text_column = 'title'
            st.session_state.page = 'analysis'
            st.rerun()
        except Exception as e:
            st.error(f"Failed to prepare news data: {e}")

def render_analysis_page():
    st.title("Sentiment Analysis")

    # Guardrails: state must exist
    df = st.session_state.get('data_to_analyze')
    text_column = st.session_state.get('text_column')
    if df is None or text_column not in (df.columns if isinstance(df, pd.DataFrame) else []):
        st.error("No dataset available for analysis. Please upload a CSV or fetch news first.")
        if st.button("Go Home"):
            st.session_state.page = 'home'
            st.rerun()
        return

    # Model
    tokenizer, model = load_model_cached()
    if not tokenizer or not model:
        st.warning("Model not loaded. Cannot continue.")
        return
    #st.write("**Model id2label mapping:**", model.config.id2label)
    
    # --- Toggle: solo decide qué columna de texto entra al modelo ---
    # Si existe 'clean_title', ofrecemos el toggle; si no, usamos la columna original.
    use_clean = False
    if 'clean_title' in df.columns:
        use_clean = st.toggle(
            "Use cleaned text for the model (recommended)", 
            value=True, 
            key="tgl_clean_for_model"
        )
    
    text_for_model_col = 'clean_title' if use_clean and 'clean_title' in df.columns else text_column
    
    # Por si alguna noticia viene con NaN
    df_copy = df.copy()
    df_copy[text_for_model_col] = df_copy[text_for_model_col].fillna("").astype(str)
    
    # Realiza el análisis (limpieza suave SOLO al texto que entra a BERT)
    with st.spinner("Analyzing sentiment with the BERT model..."):
        df_copy = df.copy()
        texts_for_model = (
            df_copy[text_for_model_col]
            .fillna('')
            .astype(str)
            .apply(clean_for_model)
            .tolist()
        )
        df_copy['sentiment_label'] = predict_sentiment_batch(texts_for_model, tokenizer, model, batch_size=64)
        sentiment_counts = df_copy['sentiment_label'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)

    # --- Visualization ---
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Sentiment Distribution")
        fig = px.pie(
            sentiment_counts,
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={'Positive':'#3b82f6', 'Neutral':'#9ca3af', 'Negative':'#ef4444'},
            hole=.3
        )
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Example Comments")
        def first_or_na(label):
            subset = df_copy[df_copy['sentiment_label'] == label]
            return subset[text_column].iloc[0] if not subset.empty else "N/A"

        pos_example = first_or_na('Positive')
        neu_example = first_or_na('Neutral')
        neg_example = first_or_na('Negative')

        st.markdown(f'<div class="summary-box"><p style="color:#3b82f6;"><strong>Positive</strong></p><p class="example">"{pos_example}"</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-box" style="margin-top:10px;"><p style="color:#9ca3af;"><strong>Neutral</strong></p><p class="example">"{neu_example}"</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-box" style="margin-top:10px;"><p style="color:#ef4444;"><strong>Negative</strong></p><p class="example">"{neg_example}"</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Keyword Cloud")

    airline_stopwords = set(['united','southwest','delta','jetblue','american','airlines','airline','air'])
    full_text = ' '.join(df_copy[text_column].dropna().astype(str))
    if full_text.strip():
        with st.spinner("Generating word cloud..."):
            filtered_words = [w for w in full_text.lower().split() if w not in airline_stopwords]
            filtered_text = ' '.join(filtered_words)
            if filtered_text.strip():
                wc = WordCloud(width=800, height=300, background_color=None, mode="RGBA", colormap='viridis').generate(filtered_text)
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("Not enough text after filtering airline names to generate a word cloud.")
    else:
        st.info("No text available to generate a word cloud.")

    # --- Download ---
    st.markdown("---")
    st.subheader("Download results")
    cols = [c for c in ['airline','published_at','title','clean_title','url'] if c in df_copy.columns]
    cols = cols + [c for c in df_copy.columns if c not in cols]
    csv_bytes = df_copy[cols].to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        "Download CSV with cleaning and sentiment",
        data=csv_bytes,
        file_name="news_processed_with_sentiment.csv",
        mime="text/csv"
    )

    if st.button("← Back to Home"):
        st.session_state.clear()
        st.session_state.page = 'home'
        st.rerun()
# --- Footer ---
def render_footer():
    st.markdown(
        """
        ---
        <div style='text-align: center; font-size: 0.9em; color: gray;'>
            This is a Master's Final Project for the Master's in Data Science and Business Analytics at IMF Smart Education, developed by Isabella Hernández.
        </div>
        """,
        unsafe_allow_html=True
    )

# --- 6) ROUTER ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if st.session_state.page == 'home':
    render_home_page()
    render_footer()
elif st.session_state.page == 'upload_csv':
    render_upload_page()
    render_footer()
elif st.session_state.page == 'column_select':
    render_column_selector_page()
    render_footer()
elif st.session_state.page == 'view_news':
    render_view_news_page()
    render_footer()
elif st.session_state.page == 'kaggle':
    # IMPORTANTE: tu kaggle_page espera tokenizer y model
    tokenizer, model = load_model_cached()
    if tokenizer and model:
        render_kaggle_page(tokenizer, model)
        render_footer()
    else:
        st.error("No se pudo cargar el modelo para Kaggle.")
elif st.session_state.page == 'analysis':
    render_analysis_page()
    render_footer()
