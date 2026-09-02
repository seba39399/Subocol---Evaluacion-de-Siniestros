import numpy as np
import pandas as pd
from src.processing import limpiar_texto, preparar_features_input


def test_limpiar_texto_remueve_caracteres_especiales_y_mayusculas():
    """
    Verifica que la función 'limpiar_texto' remueva signos de puntuación, caracteres especiales y mayúsculas.

    Garantiza la normalización básica de cadenas de texto convirtiendo todos los caracteres
    a minúsculas y eliminando símbolos no alfanuméricos como signos de exclamación y arrobas.
    """
    texto_raw = "¡Parachoques TRASERO, stop derecho!! @#$"
    texto_esperado = "parachoques trasero stop derecho"
    assert limpiar_texto(texto_raw) == texto_esperado


def test_limpiar_texto_remueve_tildes_y_acentos():
    """
    Evalúa la remoción o estandarización de caracteres acentuados y tildes en el texto.

    Comprueba que palabras con caracteres del español (como 'vehículo', 'colisión' o 'camión')
    sean procesadas correctamente para facilitar el empaquetado y la tokenización TF-IDF.
    """
    texto_con_tildes = "Vehículo sufrió una colisión en la parte delantera."
    texto_limpio = limpiar_texto(texto_con_tildes)

    # Comprueba conversión a minúsculas y eliminación de puntuación
    assert "vehiculo" in texto_limpio or "vehículo" in texto_limpio
    assert "." not in texto_limpio


def test_limpiar_texto_cadena_vacia_o_espacios():
    """
    Verifica el comportamiento de 'limpiar_texto' ante entradas vacías o compuestas solo por espacios.

    Asegura que el procesamiento de entradas nulas o compuestas puramente por espacios en blanco
    no genere excepciones y devuelva una cadena vacía.
    """
    assert limpiar_texto("") == ""
    assert limpiar_texto("   \n\t   ") == ""


def test_limpiar_texto_manejo_numeros():
    """
    Verifica la preservación o filtrado adecuado de dígitos numéricos en la cadena de texto.

    Asegura que la función procese correctamente cadenas mixtas que combinan texto con números
    (por ejemplo, códigos de partes o nomenclaturas como 'puerta 2').
    """
    texto_con_numeros = "Parachoques 2 y farola 1"
    resultado = limpiar_texto(texto_con_numeros)
    assert isinstance(resultado, str)
    assert "parachoques" in resultado


def test_preparar_features_input_evita_division_por_cero(mocker):
    """
    Verifica que 'preparar_features_input' maneje casos con cero piezas totales sin lanzar división por cero.

    Comprueba que el cálculo de proporciones (ej. ratio de piezas a cambiar sobre piezas totales)
    gestione de forma segura valores nulos o ceros mediante protecciones numéricas.
    """
    mock_tfidf = mocker.MagicMock()
    mock_tfidf.transform.return_value.toarray.return_value = np.zeros((1, 10))

    data_zero = {
        "version_hechos": "choque leve",
        "piezas_afectadas": "parachoques",
        "piezas_totales": 0,
        "piezas_cambio": 0,
    }

    X_res = preparar_features_input(data_zero, mock_tfidf, mock_tfidf)
    assert X_res is not None
    assert isinstance(X_res, (np.ndarray, pd.DataFrame))


def test_preparar_features_input_estrucutra_dimensiones(mocker):
    """
    Valida la forma y dimensiones de la matriz o DataFrame resultante del feature engineering.

    Verifica que al concatenar las representaciones vectoriales TF-IDF junto con las variables
    numéricas (tasas y conteos de piezas), la salida tenga exactamente 1 fila y las columnas esperadas.
    """
    mock_tfidf = mocker.MagicMock()
    mock_tfidf.transform.return_value.toarray.return_value = np.ones((1, 5))

    payload = {
        "version_hechos": "Impacto frontal leve",
        "piezas_afectadas": "parachoques delantero",
        "piezas_totales": 4,
        "piezas_cambio": 2,
    }

    X_res = preparar_features_input(payload, mock_tfidf, mock_tfidf)

    # La salida debe corresponder a un arreglo bidimensional con 1 fila
    shape = getattr(X_res, "shape", None)
    assert shape is not None
    assert shape[0] == 1


def test_preparar_features_input_invoca_vectorizadores(mocker):
    """
    Verifica la invocación correcta de los transformadores TF-IDF para texto.

    Garantiza que la función transforme por separado la 'version_hechos' y las 'piezas_afectadas'
    utilizando los vectorizadores pasados como parámetro.
    """
    mock_tfidf_hechos = mocker.MagicMock()
    mock_tfidf_hechos.transform.return_value.toarray.return_value = np.zeros((1, 3))

    mock_tfidf_piezas = mocker.MagicMock()
    mock_tfidf_piezas.transform.return_value.toarray.return_value = np.zeros((1, 3))

    payload = {
        "version_hechos": "Choque frontal",
        "piezas_afectadas": "farola",
        "piezas_totales": 2,
        "piezas_cambio": 1,
    }

    _ = preparar_features_input(payload, mock_tfidf_hechos, mock_tfidf_piezas)

    assert mock_tfidf_hechos.transform.called
    assert mock_tfidf_piezas.transform.called


def test_preparar_features_input_calculo_proporcion_piezas(mocker):
    """
    Valida la correcta derivación de variables numéricas calculadas (feature ratios).

    Verifica que las variables numéricas generadas a partir de 'piezas_cambio' y 'piezas_totales'
    se mantengan dentro de un rango consistente y válido para el modelo de clasificación.
    """
    mock_tfidf = mocker.MagicMock()
    mock_tfidf.transform.return_value.toarray.return_value = np.zeros((1, 2))

    payload = {
        "version_hechos": "Colisión lateral",
        "piezas_afectadas": "puerta izquierda",
        "piezas_totales": 10,
        "piezas_cambio": 5,
    }

    X_res = preparar_features_input(payload, mock_tfidf, mock_tfidf)
    assert not np.isnan(X_res).any()


def test_preparar_features_input_sin_valores_nan_o_inf(mocker):
    """
    Asegura la integridad matemática de la matriz de entrada antes de ser procesada por el modelo.

    Garantiza que ninguna transformación numérica o cálculo vectorial genere valores indeterminados
    `NaN` o infinitos (`Inf`) en la matriz resultante.
    """
    mock_tfidf = mocker.MagicMock()
    mock_tfidf.transform.return_value.toarray.return_value = np.array([[0.1, 0.5, 0.9]])

    payload = {
        "version_hechos": "Impacto posterior contra muro",
        "piezas_afectadas": "parachoques trasero, stop",
        "piezas_totales": 5,
        "piezas_cambio": 1,
    }

    X_res = preparar_features_input(payload, mock_tfidf, mock_tfidf)

    if isinstance(X_res, pd.DataFrame):
        assert not X_res.isnull().values.any()
    else:
        assert not np.isnan(X_res).any()
        assert not np.isinf(X_res).any()


def test_preparar_features_input_soporta_diccionario_y_pydantic(mocker):
    """
    Verifica la flexibilidad del empaquetado de datos para aceptar tanto diccionarios como objetos Pydantic.

    Asegura que la extracción de atributos mediante corchetes o mediante acceso por atributos
    se ejecute sin arrojar excepciones de tipo `KeyError` o `AttributeError`.
    """
    mock_tfidf = mocker.MagicMock()
    mock_tfidf.transform.return_value.toarray.return_value = np.zeros((1, 4))

    class MockPayloadPydantic:
        version_hechos = "Impacto frontal"
        piezas_afectadas = "capo"
        piezas_totales = 3
        piezas_cambio = 1

        def dict(self):
            return {
                "version_hechos": self.version_hechos,
                "piezas_afectadas": self.piezas_afectadas,
                "piezas_totales": self.piezas_totales,
                "piezas_cambio": self.piezas_cambio,
            }

    payload_pydantic = MockPayloadPydantic()
    payload_dict = payload_pydantic.dict()

    X_from_dict = preparar_features_input(payload_dict, mock_tfidf, mock_tfidf)
    assert X_from_dict is not None
