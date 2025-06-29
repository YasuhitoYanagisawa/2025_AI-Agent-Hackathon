FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run が使用するポート番号に合わせる
ENV PORT 8080
EXPOSE 8080

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]