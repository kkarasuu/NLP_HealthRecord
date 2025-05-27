FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

RUN mkdir -p /usr/share/nltk_data && \
    python -m nltk.downloader -d /usr/share/nltk_data punkt

ENV NLTK_DATA=/usr/share/nltk_data

COPY . .

CMD ["uvicorn", "web_app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
