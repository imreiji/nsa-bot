FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY nsabot ./nsabot
RUN useradd --create-home nsa && mkdir /data && chown nsa /data
USER nsa
ENV NSA_DB_PATH=/data/nsa.db PYTHONUNBUFFERED=1
CMD ["python", "-m", "nsabot.bot"]
