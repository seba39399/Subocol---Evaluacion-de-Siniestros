from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint():
    """
    Verifica que el endpoint '/health' responda correctamente y el servicio esté activo.

    Comprueba que la petición GET retorne un código de estado HTTP 200 (OK)
    y que la respuesta JSON contenga el campo 'status' con valor 'ok'.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"


def test_predict_endpoint_schema_valid():
    """
    Verifica que el endpoint '/predict' procese correctamente un payload válido y retorne el esquema esperado.

    Comprueba que la petición POST responda con código HTTP 200 (OK) al recibir una estructura
    de datos correcta, y valida que los campos de la respuesta cumplan con las restricciones del modelo:
    - Campo 'prediccion' presente y dentro de los valores permitidos ("ENTREGADO" o "OBJETADO").
    - Campo 'probabilidad_objetado' en el rango decimal de [0.0, 1.0].
    - Campo 'flag_inconsistencia_posicion' de tipo entero.
    """
    payload = {
        "version_hechos": "Vehiculo pierde adherencia en piso mojado e impacta a baja velocidad contra un separador de via con la zona delantera.",
        "piezas_afectadas": "parachoques delantero, farola derecha",
        "piezas_totales": 4,
        "piezas_cambio": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediccion" in data
    assert data["prediccion"] in ["ENTREGADO", "OBJETADO"]
    assert 0.0 <= data["probabilidad_objetado"] <= 1.0
    assert isinstance(data["flag_inconsistencia_posicion"], int)


def test_generate_pdf_endpoint():
    """
    Verifica que el endpoint '/generate-pdf' procese correctamente un payload válido y retorne un archivo PDF.

    Comprueba que la petición POST responda con código HTTP 200 (OK) al recibir una estructura
    de datos correcta, y valida que la respuesta sea un archivo PDF con contenido.
    """
    payload = {
        "version_hechos": "Impacto frontal",
        "piezas_afectadas": "parachoques delantero",
        "piezas_totales": 2,
        "piezas_cambio": 1,
        "prediccion": "ENTREGADO",
        "probabilidad_objetado": 0.15,
        "flag_inconsistencia_posicion": 0,
        "justificacion_llm": "Coherencia espacial confirmada.",
    }
    response = client.post("/generate-pdf", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert len(response.content) > 0


def test_predict_endpoint_missing_required_fields():
    """
    Verifica que el endpoint '/predict' retorne un error HTTP 422 (Unprocessable Entity)
    cuando faltan campos obligatorios en la petición JSON.
    """
    payload_incompleto = {
        "version_hechos": "Impacto frontal",
        # Falta piezas_afectadas, piezas_totales y piezas_cambio
    }
    response = client.post("/predict", json=payload_incompleto)
    assert response.status_code == 422  # Error de validación de Pydantic


def test_predict_endpoint_invalid_data_types():
    """
    Verifica que el endpoint '/predict' rechazé peticiones con tipos de datos incorrectos,
    por ejemplo, enviar texto en campos numéricos enteros.
    """
    payload_tipo_erroneo = {
        "version_hechos": "Impacto lateral",
        "piezas_afectadas": "puerta izquierda",
        "piezas_totales": "cuatro",  # Debería ser entero
        "piezas_cambio": "dos",  # Debería ser entero
    }
    response = client.post("/predict", json=payload_tipo_erroneo)
    assert response.status_code == 422


def test_predict_endpoint_empty_payload():
    """
    Verifica que la API maneje de forma segura peticiones con cuerpo JSON vacío.
    """
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_endpoint_inconsistent_pieces():
    """
    Verifica cómo reacciona el endpoint cuando el número de 'piezas_cambio'
    es mayor que 'piezas_totales' (inconsistencia lógica de negocio).
    """
    payload_inconsistente = {
        "version_hechos": "Choque posterior contra poste.",
        "piezas_afectadas": "stop izquierdo",
        "piezas_totales": 2,
        "piezas_cambio": 10,  # Inconsistencia lógica
    }
    response = client.post("/predict", json=payload_inconsistente)
    # Debe responder con éxito procesando el límite o retornar un error de validación según las reglas
    assert response.status_code in [200, 400, 422]


def test_generate_pdf_invalid_probability_range():
    """
    Verifica la robustez al generar PDF con valores fuera del rango esperado
    (por ejemplo, una probabilidad mayor a 1.0).
    """
    payload_limite = {
        "version_hechos": "Impacto frontal",
        "piezas_afectadas": "parachoques",
        "piezas_totales": 1,
        "piezas_cambio": 1,
        "prediccion": "OBJETADO",
        "probabilidad_objetado": 1.5,  # Fuera de rango normal [0.0, 1.0]
        "flag_inconsistencia_posicion": 1,
        "justificacion_llm": "Prueba límite",
    }
    response = client.post("/generate-pdf", json=payload_limite)
    # Comprueba que no genere un error 500 no controlado en el servidor
    assert response.status_code in [200, 422]


def test_predict_endpoint_method_not_allowed():
    """
    Verifica que solicitar '/predict' mediante un método no soportado (GET en vez de POST)
    devuelva un código HTTP 405 (Method Not Allowed).
    """
    response = client.get("/predict")
    assert response.status_code == 405


def test_predict_endpoint_extreme_large_text_input():
    """
    Verifica que el endpoint '/predict' procese de forma segura y sin fallar (HTTP 200)
    peticiones con textos extremadamente largos y caracteres especiales en la descripción.
    """
    payload_texto_largo = {
        "version_hechos": "Impacto frontal " * 500 + "!!!@#$%^&*()_+-=[]{}|;:',.<>?/~`",
        "piezas_afectadas": "parachoques delantero, farola izquierda, capó",
        "piezas_totales": 10,
        "piezas_cambio": 5,
    }
    response = client.post("/predict", json=payload_texto_largo)
    assert response.status_code == 200

    data = response.json()
    assert "prediccion" in data
    assert data["prediccion"] in ["ENTREGADO", "OBJETADO"]
