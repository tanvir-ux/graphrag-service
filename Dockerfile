# Live demo image: FastAPI + web UI over the built-in synthetic corpus, deterministic mock LLM by default.
# Compatible with Hugging Face Docker Spaces (port 7860, non-root user).
FROM python:3.12-slim

RUN useradd -m -u 1000 app
WORKDIR /home/app/graphrag-service
ENV PATH="/home/app/.local/bin:${PATH}" PYTHONUNBUFFERED=1

COPY --chown=app requirements.txt ./
USER app
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app pyproject.toml README.md LICENSE ./
COPY --chown=app src ./src
RUN pip install --no-cache-dir --no-deps .

EXPOSE 7860
# Gemini mode stays off unless ENABLE_GEMINI_DEMO and GEMINI_API_KEY are set as server secrets.
CMD ["graphrag", "serve", "--host", "0.0.0.0", "--port", "7860"]
