from src.processing import detectar_inconsistencia_posicion


def test_inconsistencia_posicion_detectada():
    """
    Verifica la detección de incoherencia espacial clara entre la versión de hechos y las piezas.

    Comprueba que cuando la narración describe un choque en la parte delantera pero las piezas
    reportadas corresponden a la parte trasera, la regla active la alerta retornando 1.
    """
    hechos = "vehiculo colisiona por alcance parte delantera"
    piezas = "parachoques trasero stop derecho"
    assert detectar_inconsistencia_posicion(hechos, piezas) == 1


def test_inconsistencia_posicion_no_detectada():
    """
    Valida la concordancia espacial cuando el relato y las piezas afectadas coinciden.

    Comprueba que al coincidir la descripción de un impacto en la zona trasera con el listado
    de piezas traseras, el motor de reglas no active la alerta y retorne 0.
    """
    hechos = "vehiculo impacta por la parte trasera en semaforo"
    piezas = "parachoques trasero stop izquierdo compuerta baul"
    assert detectar_inconsistencia_posicion(hechos, piezas) == 0


def test_inconsistencia_posicion_trasera_relato_vs_piezas_delanteras():
    """
    Evalúa la detección de inconsistencia inversa (relato trasero vs. piezas delanteras).

    Asegura que si la versión de hechos indica un choque posterior/trasero pero las piezas afectadas
    son de la parte frontal (ej. capó, farolas), la regla retorne 1.
    """
    hechos = "impacto por la parte trasera al frenar intempestivamente"
    piezas = "capo farola izquierda parachoques delantero"
    assert detectar_inconsistencia_posicion(hechos, piezas) == 1


def test_coherencia_posicion_lateral_izquierda():
    """
    Verifica la concordancia espacial para impactos laterales en el costado izquierdo.

    Valida que un evento descrito en el lateral o costado izquierdo en combinación con piezas
    del mismo costado genere un flag de inconsistencia igual a 0.
    """
    hechos = "rozadura lateral en el costado izquierdo al salir del parqueadero"
    piezas = "puerta delantera izquierda espejo izquierdo"
    assert detectar_inconsistencia_posicion(hechos, piezas) == 0


def test_coherencia_posicion_volcamiento_superior():
    """
    Valida la compatibilidad entre una descripción de volcamiento y piezas superiores/techo.

    Comprueba que un volcamiento o impacto en el techo asociado a piezas de la parte superior
    del vehículo sea evaluado como consistente (flag = 0).
    """
    hechos = "vehiculo sufre volcamiento sobre el techo"
    piezas = "techo panoramico parantes superiores"
    assert detectar_inconsistencia_posicion(hechos, piezas) == 0


def test_inconsistencia_posicion_insensible_a_mayusculas():
    """
    Garantiza que la regla detecte la inconsistencia independientemente del uso de mayúsculas.

    Se pasa el texto formateado en mayúsculas sostenidas para verificar que la limpieza
    interna de texto (limpiar_texto) normalice las cadenas a minúsculas antes de aplicar
    la validación de posición espacial.
    """
    hechos = "IMPACTO POR ALCANCE EN LA PARTE DELANTERA"
    piezas = "PARACHOQUES TRASERO STOP DERECHO"
    assert detectar_inconsistencia_posicion(hechos, piezas) == 1


def test_inconsistencia_posicion_con_puntuacion_y_simbolos():
    """
    Evalúa la solidez del motor de reglas ante entradas cargadas de signos de puntuación y símbolos.

    Asegura que la inclusión de guiones, comas, puntos o signos de admiración no impida
    la extracción correcta de las palabras clave de posición.
    """
    hechos = "¡¡¡Choque de frente!!! - zona delantera afectada."
    piezas = "stop trasero / parachoques trasero"
    assert detectar_inconsistencia_posicion(hechos, piezas) == 1


def test_inconsistencia_posicion_cadenas_vacias():
    """
    Verifica el comportamiento seguro del motor de reglas ante entradas vacías o nulas.

    Comprueba que al recibir textos vacíos para hechos o piezas, la regla no falle por excepción
    y asuma por defecto que no hay inconsistencia detectable (retornando 0).
    """
    assert detectar_inconsistencia_posicion("", "") == 0
    assert detectar_inconsistencia_posicion("choque frontal", "") == 0


def test_inconsistencia_posicion_relato_generico_sin_palabras_posicionales():
    """
    Evalúa el comportamiento cuando el relato narrativo no especifica ninguna zona anatómica.

    Verifica que si la versión de hechos no contiene descriptores de ubicación (ej. "falla mecánica"),
    la regla evite falsos positivos y retorne 0.
    """
    hechos = "el vehiculo se apago y sufrio un golpe con un elemento desconocido"
    piezas = "puerta derecha"
    assert detectar_inconsistencia_posicion(hechos, piezas) == 0


def test_inconsistencia_posicion_multiples_zonas_relatadas():
    """
    Verifica el comportamiento cuando la versión de hechos menciona múltiples zonas de impacto.

    Garantiza que en choques múltiples o colisiones en cadena el motor de reglas
    procese la entrada y retorne una bandera de control válida (0 o 1) sin arrojar excepciones.
    """
    hechos = "vehiculo colisiona de frente y posteriormente es impactado por la parte trasera"
    piezas = "parachoques delantero stop trasero"
    flag = detectar_inconsistencia_posicion(hechos, piezas)
    assert flag in (0, 1)
