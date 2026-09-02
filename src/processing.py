import io
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from groq import Groq
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Carga inicial intentando ubicar el archivo .env en la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def obtener_cliente_groq() -> Groq | None:
    """
    Inicializa el cliente de Groq utilizando la variable de entorno GROQ_API_KEY.

    Asegura la lectura directa desde os.getenv y realiza una verificación previa
    de la clave de autenticación antes de instanciar el cliente oficial de la SDK.

    Returns:
        Groq | None: Instancia del cliente de Groq si la API key está configurada, de lo contrario None.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def limpiar_texto(texto: str) -> str:
    """
    Limpia y normaliza un texto eliminando caracteres especiales y espacios redundantes.

    Args:
        texto (str): Cadena de texto a procesar.

    Returns:
        str: Texto normalizado en minúsculas, sin tildes/caracteres especiales y con espacios simples.
    """
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñ0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def detectar_inconsistencia_posicion(hechos_clean: str, piezas_clean: str) -> int:
    """
    Evalúa contradicciones espaciales entre la narración del evento y los repuestos solicitados.

    Args:
        hechos_clean (str): Narración de los hechos.
        piezas_clean (str): Listado de piezas afectadas.

    Returns:
        int: 1 si detecta una contradicción espacial (frente vs. atrás), 0 en caso contrario.
    """

    hechos_clean = (hechos_clean or "").lower()
    piezas_clean = (piezas_clean or "").lower()

    frente_en_hechos = any(
        w in hechos_clean
        for w in [
            "frente",
            "frontal",
            "delante",
            "delantera",
            "delantero",
            "trompa",
            "capo",
        ]
    )
    atras_en_piezas = any(
        w in piezas_clean
        for w in [
            "trasero",
            "trasera",
            "atras",
            "atrás",
            "baul",
            "baúl",
            "stop",
            "compuerta",
        ]
    )

    atras_en_hechos = any(
        w in hechos_clean
        for w in ["atras", "atrás", "trasera", "trasero", "reversa", "cola"]
    )
    frente_en_piezas = any(
        w in piezas_clean
        for w in [
            "delantero",
            "delantera",
            "frente",
            "frontal",
            "farola",
            "capo",
            "parachoque del",
        ]
    )

    if (frente_en_hechos and atras_en_piezas) or (atras_en_hechos and frente_en_piezas):
        return 1
    return 0


def analizar_siniestro_con_llm(
    version_hechos: str, piezas_afectadas: str, probabilidad_ml: float, flag_rule: int
) -> tuple[int, str]:
    """
    Utiliza un LLM (Llama 3 en Groq) para evaluar la coherencia semántico-espacial del siniestro
    y generar una justificación técnica en lenguaje natural para el auditor.

    Realiza una recarga de la variable de entorno GROQ_API_KEY, construye la interacción
    con respuesta estructurada JSON, captura excepciones detalladas de la API y garantiza
    mensajes explícitos en caso de errores de autenticación o configuración.

    Args:
        version_hechos (str): Descripción textual del accidente narrada por el usuario.
        piezas_afectadas (str): Listado de repuestos o piezas reportadas como dañadas.
        probabilidad_ml (float): Probabilidad calculada por el modelo de machine learning.
        flag_rule (int): Flag determinista de inconsistencia detectado por reglas de palabras clave.

    Returns:
        Tuple[int, str]: Una tupla con (flag_inconsistencia_llm, justificacion_tecnica).
    """
    # Forzar recarga del archivo .env si no se ha detectado la variable previamente
    if not os.getenv("GROQ_API_KEY"):
        load_dotenv(dotenv_path=ENV_PATH)

    client = obtener_cliente_groq()

    # Manejo de error claro si la clave no está configurada en el entorno
    if not client:
        return (
            flag_rule,
            f"[ERROR CONFIG]: No se encontró la variable GROQ_API_KEY en {ENV_PATH}",
        )

    prompt = f"""
    Eres un perito experto en auditoría de siniestros de automóviles para una aseguradora.
    Analiza la coherencia lógica y la física del impacto del siguiente reclamo:

    DATOS DEL RECLAMO:
    - Versión de los hechos: "{version_hechos}"
    - Piezas afectadas reclamadas: "{piezas_afectadas}"
    - Probabilidad de objeción calculada por ML: {probabilidad_ml:.1%}
    - Inconsistencia espacial detectada por reglas: {"SÍ" if flag_rule == 1 else "NO"}

    TAREAS:
    1. Evalúa si existe una incompatibilidad física o semántica real entre la zona del impacto narrada y la ubicación de los repuestos exigidos.
    2. Redacta una justificación técnica concisa (máximo 2 oraciones) orientada al auditor pericial.

    RESPONDE EXACTAMENTE EN EL SIGUIENTE FORMATO JSON:
    {{
        "inconsistencia_detectada": 1 o 0,
        "justificacion": "Explicación breve aquí."
    }}
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Responde únicamente en formato JSON válido.",
                },
                {"role": "user", "content": prompt},
            ],
            model="qwen/qwen3.6-27b",
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("La respuesta de Groq no devolvió contenido JSON.")

        res_json = json.loads(content)
        flag_llm = int(res_json.get("inconsistencia_detectada", flag_rule))
        justificacion = res_json.get(
            "justificacion", "Análisis completado exitosamente."
        )

        return flag_llm, justificacion

    except Exception as e:
        # Reportar el detalle real de la excepción de Groq o de la red en lugar de un mensaje genérico
        return flag_rule, f"[ERROR GROQ API]: {str(e)}"


def preparar_features_input(
    data: dict[str, Any], tfidf_hechos: Any, tfidf_piezas: Any
) -> np.ndarray:
    """
    Transforma el diccionario o petición de entrada en la matriz de características para el modelo.

    Extrae características numéricas, calcula la tasa de reemplazo de piezas, identifica
    inconsistencias espaciales y genera los vectores TF-IDF para la narración y los repuestos.

    Args:
        data (Dict[str, Any]): Diccionario con las llaves 'version_hechos', 'piezas_afectadas',
            'piezas_totales' y 'piezas_cambio'.
        tfidf_hechos (TfidfVectorizer): Vectorizador TF-IDF ajustado para la versión de los hechos.
        tfidf_piezas (TfidfVectorizer): Vectorizador TF-IDF ajustado para el listado de piezas.

    Returns:
        np.ndarray: Vector 2D (1, N_features) con el formato exacto requerido por el modelo.
    """
    version_hechos = data.get("version_hechos", "")
    raw_piezas = data.get("piezas_afectadas", "")

    # Normalizar piezas_afectadas a string y lista según el tipo recibido
    if isinstance(raw_piezas, list):
        piezas_afectadas_list: list[str] = [
            str(p)
            for p in raw_piezas
            if isinstance(p, (str, int, float)) and p is not None
        ]
        piezas_afectadas_str = " | ".join(piezas_afectadas_list)
        cant_piezas_distintas = len(piezas_afectadas_list)
    else:
        piezas_afectadas_str = str(raw_piezas)
        cant_piezas_distintas = len(
            [p for p in piezas_afectadas_str.split(",") if p.strip()]
        )

    piezas_totales = data.get("piezas_totales", cant_piezas_distintas)
    piezas_cambio = data.get("piezas_cambio", cant_piezas_distintas)

    hechos_clean = limpiar_texto(version_hechos)
    piezas_clean = limpiar_texto(piezas_afectadas_str)

    ratio_piezas_cambio = float(piezas_cambio) / (float(piezas_totales) + 1e-5)
    len_hechos = len(hechos_clean.split())
    inconsistencia = detectar_inconsistencia_posicion(hechos_clean, piezas_clean)

    vec_num: np.ndarray = np.asarray(
        [
            [
                float(piezas_totales),
                float(piezas_cambio),
                float(ratio_piezas_cambio),
                float(len_hechos),
                float(cant_piezas_distintas),
                float(inconsistencia),
            ]
        ],
        dtype=float,
    )

    vec_tfidf_h = tfidf_hechos.transform([hechos_clean]).toarray()
    vec_tfidf_p = tfidf_piezas.transform([piezas_clean]).toarray()

    return np.hstack([vec_num, vec_tfidf_h, vec_tfidf_p])


def generar_pdf_dictamen(datos: dict[str, Any]) -> bytes:
    """Genera en memoria un documento PDF pericial estructurado con ReportLab.

    Construye un reporte ejecutivo en PDF que resume la versión de los hechos, el dictamen
    del modelo (OBJETADO / ENTREGADO), las métricas cuantitativas clave y la justificación
    técnica generada por el agente GenAI.

    Args:
        datos (Dict[str, Any]): Diccionario con la información del siniestro y la predicción:
            - version_hechos (str): Descripción del accidente.
            - piezas_afectadas (str): Lista de componentes reclamados.
            - piezas_totales (int): Conteo total de piezas involucradas.
            - piezas_cambio (int): Piezas solicitadas para cambio.
            - prediccion (str): Dictamen pericial ('OBJETADO' o 'ENTREGADO').
            - probabilidad_objetado (float): Probabilidad calculada (0.0 a 1.0).
            - flag_inconsistencia_posicion (int): Indicador de contradicción espacial.
            - justificacion_llm (str): Argumentación técnica en lenguaje natural.

    Returns:
        bytes: Flujo binario buffer con el contenido completo del archivo PDF.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Definición de estilos tipográficos ejecutivos
    title_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "HeaderSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=15,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )

    bold_label_style = ParagraphStyle(
        "BoldLabel",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0f172a"),
    )

    story: list[object] = []

    # Encabezado Principal
    story.append(
        Paragraph("SUBOCOL — INFORME TÉCNICO DE AUDITORÍA DE SINIESTROS", title_style)
    )
    story.append(
        Paragraph(
            "Sistema Pericial Automático - Evaluación Híbrida ML + GenAI",
            subtitle_style,
        )
    )
    story.append(
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"))
    )
    story.append(Spacer(1, 10))

    # Extracción y formateo de datos del dictamen
    prediccion = datos.get("prediccion", "N/A")
    prob = datos.get("probabilidad_objetado", 0.0)
    flag_espacial = datos.get("flag_inconsistencia_posicion", 0)
    es_objetado = prediccion == "OBJETADO"

    color_dictamen = (
        colors.HexColor("#dc2626") if es_objetado else colors.HexColor("#16a34a")
    )
    bg_dictamen = (
        colors.HexColor("#fef2f2") if es_objetado else colors.HexColor("#f0fdf4")
    )

    # Cuadro Resumen del Dictamen Pericial
    dictamen_text = f"<b>DICTAMEN FINAL:</b> <font color='{color_dictamen.hexval()}'>{prediccion}</font>"
    dictamen_style = ParagraphStyle(
        "DictamenBox",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=color_dictamen,
    )

    summary_table_data = [[Paragraph(dictamen_text, dictamen_style)]]
    summary_table = Table(summary_table_data, colWidths=[540])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg_dictamen),
                ("BOX", (0, 0), (-1, -1), 1.5, color_dictamen),
                ("PADDING", (0, 0), (-1, -1), 10),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 15))

    # Sección 1: Datos de Entrada del Reclamo
    story.append(Paragraph("1. Información General del Reclamo", section_heading))

    table_data = [
        [
            Paragraph("Versión de los Hechos:", bold_label_style),
            Paragraph(datos.get("version_hechos", "N/A"), body_style),
        ],
        [
            Paragraph("Piezas Afectadas:", bold_label_style),
            Paragraph(str(datos.get("piezas_afectadas", "N/A")), body_style),
        ],
        [
            Paragraph("Piezas Totales Involucradas:", bold_label_style),
            Paragraph(str(datos.get("piezas_totales", "N/A")), body_style),
        ],
        [
            Paragraph("Piezas Solicitadas para Cambio:", bold_label_style),
            Paragraph(str(datos.get("piezas_cambio", "N/A")), body_style),
        ],
    ]

    claim_table = Table(table_data, colWidths=[160, 380])
    claim_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(claim_table)
    story.append(Spacer(1, 15))

    # Sección 2: Métricas de Modelado y Reglas Espaciales
    story.append(
        Paragraph("2. Métricas del Motor Pericial (ML + Reglas)", section_heading)
    )

    metrics_data = [
        [
            Paragraph("<b>Métrica / Indicador</b>", bold_label_style),
            Paragraph("<b>Valor Calculado</b>", bold_label_style),
        ],
        [
            Paragraph("Probabilidad Continuo de Objeción", body_style),
            Paragraph(f"{prob * 100:.1f}%", body_style),
        ],
        [
            Paragraph("Inconsistencia Espacial (Regla Física)", body_style),
            Paragraph(
                "SÍ (1) — Incompatibilidad detectada"
                if flag_espacial == 1
                else "NO (0) — Coherencia espacial",
                body_style,
            ),
        ],
    ]

    metrics_table = Table(metrics_data, colWidths=[270, 270])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#cbd5e1")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(metrics_table)
    story.append(Spacer(1, 15))

    # Sección 3: Explicabilidad GenAI
    story.append(
        Paragraph("3. Auditoría Semántica Generativa (Groq LLM)", section_heading)
    )
    justificacion_txt = datos.get("justificacion_llm", "Sin justificación disponible.")

    llm_box_data = [
        [
            Paragraph(
                f"<b>Justificación Técnica del Agente:</b><br/>{justificacion_txt}",
                body_style,
            )
        ]
    ]
    llm_table = Table(llm_box_data, colWidths=[540])
    llm_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(llm_table)

    # Construcción final del documento
    doc.build(story)  # type: ignore
    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes
