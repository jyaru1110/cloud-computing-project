FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

WORKDIR /app

# Dependencias de serving
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Dependencias de analisis
RUN pip install --no-cache-dir \
    scikit-learn>=1.4.0 \
    statsmodels>=0.14.0 \
    seaborn>=0.13.0 \
    wordcloud>=1.9.0 \
    nltk>=3.8.0 \
    google-cloud-storage>=2.18.0

COPY src /app/src
COPY statistical_toolbelt /app/statistical_toolbelt

EXPOSE 8080
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
