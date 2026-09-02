# Usa una imagen oficial y ligera de Python 3.12
FROM python:3.12-slim

# Variables de entorno para evitar buffering de logs y optimizar Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Instala dependencias del sistema necesarias para ReportLab, OpenCV y curl para HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copia manifiestos de dependencias desde la raíz
COPY pyproject.toml uv.lock ./

# Instala dependencias del proyecto
RUN uv pip install --system --no-cache -r pyproject.toml

# Copia la carpeta de modelos y todo el código fuente de la aplicación
COPY models/ /app/models/
COPY src/ /app/src/

# Seguridad: Crear y usar usuario sin privilegios
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expone el puerto del backend
EXPOSE 8000

# Healthcheck que verifica el endpoint de salud de la API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Servidor de producción Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]