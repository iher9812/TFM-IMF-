# TFM: Streamlit Airline Sentiment Analysis

BERT + Streamlit para analizar sentimiento en noticias/tweets de aerolíneas.

## Requisitos
- Python 3.11
- Ver `requirements.txt`

## Instalación (local)
python -m venv .venv
. .venv/Scripts/activate   # Windows
pip install -r requirements.txt

## Ejecutar (local)
streamlit run TFMapp/app.py

python -m streamlit run 'app.py' --server.headless true


## Despliegue (Streamlit Cloud)
Elige `TFMapp/app.py` como Main file. La app descarga/carga el modelo según README_models.md.

## Estructura
- TFMapp/ … app.py y utilidades
- BERTmodel/ … (material del modelo)
- DatasetNEWS/ … datos/notebooks