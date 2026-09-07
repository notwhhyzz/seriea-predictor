FROM python:3.11-slim

WORKDIR /app

# System deps (build essentials for scipy/sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir streamlit watchdog

COPY . .

# Streamlit headless config
RUN mkdir -p /app/.streamlit && printf '[browser]\ngatherUsageStats = false\n[server]\nheadless = true\nport = 8501\nenableCORS = false\n' > /app/.streamlit/config.toml

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/dashboard.py", "--server.port", "8501", "--server.address", "0.0.0.0"]