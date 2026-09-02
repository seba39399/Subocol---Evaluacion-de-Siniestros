# Usa una imagen oficial y ligera de Python
FROM python:3.12-slim

# Evita la creación de archivos .pyc y fuerza el stdout/stderr sin buffer
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Instala uv para instalaciones ultrarrápidas y curl para el HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copia los archivos de configuración de dependencias
COPY pyproject.toml uv.lock ./

# Instala las dependencias del proyecto omitiendo las de desarrollo (--no-dev)
RUN uv pip install --system --no-cache -r pyproject.toml

# Copia el código de la aplicación
COPY src/ /app/src/
COPY ui/ /app/ui/

# Crea un usuario sin privilegios por seguridad
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expone el puerto por defecto de Streamlit
EXPOSE 8501

# Healthcheck para verificar la disponibilidad del servicio
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando por defecto para iniciar Streamlit
CMD ["streamlit", "run", "ui/app.py"]