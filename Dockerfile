# UK Energy Intelligence — Streamlit demo container.
# Targets Hugging Face Spaces (Docker SDK), which runs as UID 1000 on port 7860.

FROM python:3.11-slim

# HF Spaces runs containers as a non-root user with UID 1000.
RUN useradd --create-home --uid 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

# Install Python deps first so this layer caches when only application code changes.
COPY --chown=user requirements-deploy.txt ./
RUN pip install --no-cache-dir --user -r requirements-deploy.txt

# Application code and model artifacts.
COPY --chown=user src/ src/
COPY --chown=user artifacts/ artifacts/
COPY --chown=user app.py ./

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
