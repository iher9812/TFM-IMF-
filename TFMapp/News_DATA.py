
# -*- coding: utf-8 -*-
# News_DATA.py - API pública para adquisición + limpieza + guardado de RSS (Bing News)
import os
import re
from datetime import datetime
import pandas as pd
import feedparser

# Feeds fijos por aerolínea (mantiene tus parámetros originales)
AIRLINE_FEEDS = {
    "United Airlines": "https://www.bing.com/news/search?q=united+airlines&cc=us&qft=interval%3d%228%22&form=PTFTNR&format=rss",
    "US Airways": "https://www.bing.com/news/search?q=US+Airways&cc=us&qft=interval%3d%228%22&form=PTFTNR&format=rss",
    "Southwest": "https://www.bing.com/news/search?q=Southwest&cc=us&qft=interval%3d%229%22&form=PTFTNR&format=rss",
    "Delta Air Lines": "https://www.bing.com/news/search?q=delta+airlines&cc=us&qft=interval%3d%229%22&form=PTFTNR&format=rss",
    "Virgin America": "https://www.bing.com/news/search?q=Virgin+America&cc=us&qft=interval%3d%229%22&form=PTFTNR&format=rss",
    "JetBlue": "https://www.bing.com/news/search?q=JetBlue&cc=us&qft=interval%3d%228%22&form=PTFTNR&format=rss",
}

def gentle_clean(text: str) -> str:
    """Limpieza suave: mantiene mayúsculas, emojis y acentos; quita URLs/HTML/espacios dobles."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'(https?://\S+|www\.\S+)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_airline_news_clean(airline: str, out_dir: str = "data", filename_prefix: str = "news_clean", min_chars: int = 5):
    """
    Pipeline completo para pastel/nube:
      1) Descarga/parsea RSS según aerolínea
      2) Limpia títulos/resúmenes con limpieza suave
      3) Guarda CSV con timestamp
      4) Devuelve (DataFrame limpio, ruta CSV)
    """
    if airline not in AIRLINE_FEEDS:
        raise ValueError(f"Aerolínea no soportada: {airline}. Opciones: {list(AIRLINE_FEEDS.keys())}")

    rss_url = AIRLINE_FEEDS[airline]
    feed = feedparser.parse(rss_url)
    entries = getattr(feed, "entries", []) or []

    rows = []
    for item in entries:
        raw_title = getattr(item, "title", "")
        raw_summary = getattr(item, "summary", "")
        link = getattr(item, "link", "")
        published = getattr(item, "published", "") or getattr(item, "updated", "")

        clean_title = gentle_clean(raw_title)
        clean_summary = gentle_clean(raw_summary)

        if len(clean_title) >= min_chars:
            rows.append({
                "airline": airline,
                "title": raw_title,
                "summary": raw_summary,
                "link": link,
                "published": published,
                "text": clean_title,
                "text_summary": clean_summary,
            })

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"{filename_prefix}_{airline.replace(' ', '_')}_{ts}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return df, csv_path
