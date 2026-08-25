# syntax=docker/dockerfile:1
FROM node:20-alpine AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./
# sentence-transformers only needs CPU inference in the standard web image.
# Installing torch from PyPI can pull multiple CUDA runtime wheels (several GB)
# even though this image has no GPU runtime; install the CPU wheel first.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt
COPY . ./
COPY --from=frontend /build/static/dist ./static/dist
RUN useradd --create-home --uid 10001 artagent && chown -R artagent:artagent /app
USER artagent
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3)"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
