FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install deps then remove build tools in the SAME layer to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir streamlit watchdog \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/* /root/.cache

COPY . .

# Streamlit headless config
RUN mkdir -p /app/.streamlit && printf '[browser]\ngatherUsageStats = false\n[server]\nheadless = true\nport = 8501\nenableCORS = false\n' > /app/.streamlit/config.toml

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"]