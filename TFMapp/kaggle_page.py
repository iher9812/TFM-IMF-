import os
import pandas as pd
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px
import torch
import kagglehub
from utils_nlp import clean_for_model, limpiar_texto_dataset, sentiment_map

# ===================== dataset Kaggle =====================
@st.cache_data
def load_kaggle_dataset() -> pd.DataFrame | None:
    """Descarga el dataset crowdflower/twitter-airline-sentiment y devuelve Tweets.csv."""
    path = kagglehub.dataset_download("crowdflower/twitter-airline-sentiment")
    csv_path = os.path.join(path, "Tweets.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")

# ===================== predicción en lotes con barra =====================
def predict_sentiment_batch_with_progress(texts, tokenizer, model, batch_size=32, gif_url: str | None = None):
    labels = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    # Barra de progreso
    progress = st.progress(0, text="Analyzing tweets...")

    # Placeholder para el GIF justo DEBAJO de la barra
    gif_ph = st.empty()
    if gif_url:
        # Mostrar el gato durante todo el procesamiento
        gif_ph.image(gif_url, use_container_width=True)

    # Bucle de predicción
    for b, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch = [str(t) for t in texts[i:i+batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=160, return_tensors="pt")
        with torch.no_grad():
            preds = torch.argmax(model(**inputs).logits, dim=1).tolist()
        labels.extend([sentiment_map.get(p, "Neutral") for p in preds])

        progress.progress(min(b / max(total_batches, 1), 1.0),
                          text=f"Analyzing tweets... {b}/{total_batches}")

    # Si quieres que el gif desaparezca al terminar, descomenta la siguiente línea:
    gif_ph.empty()

    progress.empty()
    return labels

# ===================== página de Kaggle =====================
def render_kaggle_page(tokenizer, model):
    """
    Página Kaggle:
    - Carga automática (Kaggle Hub) sin UI de ruta/subida
    - Preview
    - Toggle "Use cleaned text for the model"
    - Predicción con barra + GIF
    - Pie chart + WordCloud por aerolínea
    """
    st.title("Analyze Kaggle Airline Tweets")

    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        if 'df_kaggle_analyzed' in st.session_state:
            del st.session_state.df_kaggle_analyzed
        st.rerun()

    # --- Cargar dataset (silencioso) ---
    with st.spinner("Loading dataset…"):
        df_orig = load_kaggle_dataset()
    if df_orig is None or df_orig.empty:
        st.error("Could not load the Kaggle dataset automatically. Please check your Kaggle Hub setup.")
        st.stop()

    # --- Trabajo sobre copia (o cache ya analizado) ---
    df = st.session_state.get('df_kaggle_analyzed', df_orig.copy())

    # --- Preview ---
    st.subheader("Preview")
    st.dataframe(df.head())

    # --- Toggle para decidir el texto que alimenta al modelo ---
    st.divider()
    use_clean_for_model = st.toggle(
        "Use cleaned text for the model (recommended: off)",
        value=False,
        key="tgl_kaggle_clean"
    )

    text_for_model_col = 'text'
    if use_clean_for_model:
        if 'clean_title' not in df.columns:
            df['clean_title'] = df['text'].astype(str).apply(limpiar_texto_dataset)
        text_for_model_col = 'clean_title'

    # --- Botón de análisis: barra de progreso + GIF del gatito ---
    gif_url = "https://i.pinimg.com/originals/4c/d9/ce/4cd9ce636c6d5f23688f0fda99cd81cf.gif"
    if st.button("Analyze Tweets", use_container_width=True, key="start_kaggle_analysis"):
        texts_for_model = (
            df[text_for_model_col]
            .fillna('')
            .astype(str)
            .apply(clean_for_model)
            .tolist()
        )
        df['sentiment_label'] = predict_sentiment_batch_with_progress(
            texts_for_model, tokenizer, model, batch_size=32, gif_url=gif_url
        )
        st.session_state.df_kaggle_analyzed = df
        st.success("Analysis complete!")

    # --- Visualización por aerolínea ---
    df_show = st.session_state.get('df_kaggle_analyzed')
    if df_show is not None:
        st.markdown("---")
        st.subheader("Analysis by Airline")

        airlines = df_show['airline'].dropna().unique()
        for airline in airlines:
            with st.expander(f"Sentiment Analysis for {airline}", expanded=False):
                sub = df_show[df_show['airline'] == airline]

                col1, col2 = st.columns([1, 1])

                # 1) Pie chart
                with col1:
                    st.markdown("#### Sentiment Distribution")
                    counts = sub['sentiment_label'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
                    if counts.sum() > 0:
                        fig_pie = px.pie(
                            counts,
                            values=counts.values,
                            names=counts.index,
                            color=counts.index,
                            color_discrete_map={'Positive':'#3b82f6', 'Neutral':'#9ca3af', 'Negative':'#ef4444'},
                            hole=.3
                        )
                        fig_pie.update_layout(
                            showlegend=False,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=20, r=20, t=30, b=20)
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info("No data to display pie chart.")

                # 2) Word cloud
                with col2:
                    st.markdown("#### Common Keywords")
                    text = ' '.join(sub['text'].dropna().astype(str))
                    airline_name_parts = airline.lower().replace('airlines', '').replace('air', '').split()
                    custom_stopwords = set(STOPWORDS).union(airline_name_parts)

                    if text.strip():
                        try:
                            wc = WordCloud(
                                width=400, height=300,
                                background_color=None, mode="RGBA",
                                colormap='viridis',
                                stopwords=custom_stopwords
                            ).generate(text)
                            fig_wc, ax = plt.subplots()
                            ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
                            fig_wc.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
                            st.pyplot(fig_wc)
                        except Exception as e:
                            st.error(f"Could not generate word cloud: {e}")
                    else:
                        st.info("Not enough text to generate a word cloud.")
