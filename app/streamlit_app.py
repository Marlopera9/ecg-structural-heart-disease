import streamlit as st
import numpy as np
import requests
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000"

# Colores del tema, reutilizados también en las gráficas de matplotlib
# para que combinen con el resto de la interfaz
COLOR_PRINCIPAL = "#0F766E"
COLOR_TRAZO = "#1E293B"

CLASES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
UMBRALES = {'NORM': 0.5, 'MI': 0.5, 'STTC': 0.5, 'CD': 0.5, 'HYP': 0.263}
NOMBRES_DERIVACIONES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ORDEN_GRID_CLINICO = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

# ============================================================
# DICCIONARIO DE TEXTOS (español / inglés)
# ============================================================
TEXTOS = {
    'es': {
        'titulo_principal': "Detección de Enfermedad Cardíaca Estructural mediante ECG",
        'aviso_demo': "⚠️ Esta es una demostración con fines educativos y de portfolio técnico. No debe usarse como herramienta de diagnóstico real.",
        'tab_diagnostico': "🔍 Diagnóstico",
        'tab_guia': "📖 Guía: derivaciones y diagnósticos",
        'sidebar_header': "Elige un ECG de ejemplo",
        'ejemplo_formato': "Ejemplo {n} — diagnóstico real: {clases}",
        'boton_azar': "🎲 Elegir uno al azar",
        'label_edad': "**Edad:** {edad}",
        'label_sexo': "**Sexo:** {sexo}",
        'texto_hombre': "Hombre",
        'texto_mujer': "Mujer",
        'texto_90_mas': "90+ años",
        'subheader_señal': "Señal ECG",
        'checkbox_12_derivaciones': "Ver las 12 derivaciones a la vez",
        'caption_grid_clinico': "Disposición idéntica a la de un ECG clínico impreso: 3 filas × 4 columnas. ¿Qué significa cada una? Consulta la pestaña 'Guía'.",
        'label_derivacion_mostrar': "Derivación a mostrar:",
        'formato_derivacion': "Derivación {d}",
        'subheader_diagnostico': "Diagnóstico",
        'boton_analizar': "🔍 Analizar este ECG",
        'caption_leyenda': "✅ = coincide con el diagnóstico real de este ejemplo · ❌ = no coincide · Umbral de Hipertrofia optimizado a 0,263 (ver README para el porqué).",
        'subheader_explicacion': "¿En qué se basó el modelo?",
        'label_explicar_para': "Explicar la predicción para:",
        'boton_gradcam': "🧠 Generar Grad-CAM",
        'titulo_gradcam_formato': "Grad-CAM — {clase} (probabilidad: {prob}%)",
        'caption_gradcam': "Las zonas más rojas son los instantes de la señal donde el modelo se fijó más para tomar esta decisión.",
        'info_pulsa_analizar': "Pulsa \"Analizar este ECG\" primero para poder generar la explicación.",
        'footer_caption': "Dataset: PTB-XL (PhysioNet). Proyecto educativo — no es una herramienta de diagnóstico médico real. Código completo en [GitHub](https://github.com/Marlopera9/ecg-structural-heart-disease).",
        'error_conexion': "No se pudo conectar con la API. Verifica que esté corriendo en otra terminal con `uvicorn src.api:app --reload` antes de usar la interfaz.",
        'error_timeout': "La API tardó demasiado en responder. Inténtalo de nuevo.",
        'spinner_analizando': "Analizando ECG...",
        'spinner_explicando': "Generando explicación visual...",

        'nombres_clases': {
            'NORM': 'Normal', 'MI': 'Infarto de miocardio', 'STTC': 'Cambios ST/T',
            'CD': 'Trastorno de conducción', 'HYP': 'Hipertrofia',
        },
        'descripciones_clases': {
            'NORM': 'Sin anomalías detectadas.',
            'MI': 'Daño al músculo del corazón por falta de riego sanguíneo.',
            'STTC': 'Cambios en el ritmo eléctrico que pueden indicar falta de riego o inflamación.',
            'CD': 'El impulso eléctrico no viaja correctamente por el corazón.',
            'HYP': 'Engrosamiento anormal del músculo del corazón.',
        },

        'guia_header_derivaciones': "¿Qué es un ECG de 12 derivaciones?",
        'guia_texto_derivaciones': (
            "Imagina 12 cámaras colocadas en distintos ángulos alrededor del corazón: "
            "todas están grabando el mismo latido, pero cada una lo ve desde un punto "
            "de vista distinto. Eso son las **derivaciones**: cada una detecta mejor "
            "unas cosas que otras, dependiendo de qué parte del corazón \"mira\" más de "
            "cerca.\n\nSe agrupan en 3 familias, según dónde se colocan los electrodos "
            "en el cuerpo:"
        ),
        'guia_col1_titulo': "**Derivaciones de extremidades**",
        'guia_col1_lista': "I, II, III",
        'guia_col1_caption': "Electrodos en brazos y piernas. Forman un triángulo alrededor del corazón.",
        'guia_col2_titulo': "**Derivaciones aumentadas**",
        'guia_col2_lista': "aVR, aVL, aVF",
        'guia_col2_caption': "Mismos electrodos que las anteriores, combinados matemáticamente para dar ángulos adicionales.",
        'guia_col3_titulo': "**Derivaciones precordiales**",
        'guia_col3_lista': "V1 – V6",
        'guia_col3_caption': "Electrodos colocados directamente sobre el pecho, muy cerca del corazón.",
        'guia_tabla_titulo': "#### ¿Qué zona del corazón ve cada una?",
        'guia_tabla': (
            "| Derivaciones | Zona del corazón que observan |\n"
            "|---|---|\n"
            "| II, III, aVF | Cara inferior |\n"
            "| I, aVL, V5, V6 | Cara lateral |\n"
            "| V1, V2 | Tabique (entre los dos ventrículos) |\n"
            "| V3, V4 | Cara anterior (frontal) |\n"
        ),
        'guia_header_diagnosticos': "¿Qué diagnostica este modelo?",
        'guia_intro_diagnosticos': "El modelo evalúa 5 categorías. Un mismo ECG puede pertenecer a varias a la vez.",
        'guia_diagnosticos': {
            'NORM': ("**Normal (NORM)**", "El ECG no muestra ninguna de las 4 anomalías siguientes."),
            'MI': ("**Infarto de miocardio (MI)**",
                   "Comúnmente conocido como 'ataque al corazón': una parte del músculo "
                   "cardíaco no recibe suficiente sangre (y por tanto oxígeno), lo que "
                   "puede dañarlo si no se trata a tiempo."),
            'STTC': ("**Cambios ST/T (STTC)**",
                     "El ECG tiene una forma característica, como una montaña con varios "
                     "picos y valles. El 'segmento ST' y la 'onda T' son dos de esas "
                     "partes concretas; cuando cambian de forma de lo esperado, puede "
                     "indicar falta de riego sanguíneo, inflamación, u otros "
                     "desequilibrios."),
            'CD': ("**Trastorno de conducción (CD)**",
                   "El corazón late gracias a un impulso eléctrico que viaja por unos "
                   "'cables' internos en un orden concreto. Cuando ese impulso se "
                   "retrasa o se bloquea en algún punto del camino, se produce este "
                   "tipo de trastorno (por ejemplo, los 'bloqueos de rama')."),
            'HYP': ("**Hipertrofia (HYP)**",
                    "El músculo del corazón se ha engrosado más de lo normal, a menudo "
                    "como consecuencia de tener que trabajar contra una resistencia "
                    "elevada durante mucho tiempo (por ejemplo, presión arterial alta "
                    "no controlada)."),
        },
        'guia_info_final': "💡 Esta guía es una simplificación con fines educativos. Para cualquier duda sobre un ECG real, consulta siempre a un profesional sanitario.",
    },

    'en': {
        'titulo_principal': "Structural Heart Disease Detection from ECG",
        'aviso_demo': "⚠️ This is a demonstration for educational and technical portfolio purposes. It should not be used as a real diagnostic tool.",
        'tab_diagnostico': "🔍 Diagnosis",
        'tab_guia': "📖 Guide: leads & diagnoses",
        'sidebar_header': "Choose a sample ECG",
        'ejemplo_formato': "Example {n} — actual diagnosis: {clases}",
        'boton_azar': "🎲 Pick one at random",
        'label_edad': "**Age:** {edad}",
        'label_sexo': "**Sex:** {sexo}",
        'texto_hombre': "Male",
        'texto_mujer': "Female",
        'texto_90_mas': "90+ years",
        'subheader_señal': "ECG Signal",
        'checkbox_12_derivaciones': "View all 12 leads at once",
        'caption_grid_clinico': "Same layout as a printed clinical ECG: 3 rows × 4 columns. Not sure what each one means? Check the 'Guide' tab.",
        'label_derivacion_mostrar': "Lead to display:",
        'formato_derivacion': "Lead {d}",
        'subheader_diagnostico': "Diagnosis",
        'boton_analizar': "🔍 Analyze this ECG",
        'caption_leyenda': "✅ = matches the actual diagnosis for this example · ❌ = doesn't match · Hypertrophy threshold optimized to 0.263 (see README for why).",
        'subheader_explicacion': "What did the model base this on?",
        'label_explicar_para': "Explain the prediction for:",
        'boton_gradcam': "🧠 Generate Grad-CAM",
        'titulo_gradcam_formato': "Grad-CAM — {clase} (probability: {prob}%)",
        'caption_gradcam': "The reddest areas are the moments of the signal the model focused on most to make this decision.",
        'info_pulsa_analizar': "Click \"Analyze this ECG\" first to generate the explanation.",
        'footer_caption': "Dataset: PTB-XL (PhysioNet). Educational project — not a real medical diagnostic tool. Full code on [GitHub](https://github.com/Marlopera9/ecg-structural-heart-disease).",
        'error_conexion': "Could not connect to the API. Make sure it's running in another terminal with `uvicorn src.api:app --reload` before using the interface.",
        'error_timeout': "The API took too long to respond. Please try again.",
        'spinner_analizando': "Analyzing ECG...",
        'spinner_explicando': "Generating visual explanation...",

        'nombres_clases': {
            'NORM': 'Normal', 'MI': 'Myocardial infarction', 'STTC': 'ST/T changes',
            'CD': 'Conduction disturbance', 'HYP': 'Hypertrophy',
        },
        'descripciones_clases': {
            'NORM': 'No abnormalities detected.',
            'MI': 'Damage to the heart muscle from insufficient blood supply.',
            'STTC': 'Changes in the electrical rhythm that may indicate poor blood flow or inflammation.',
            'CD': "The electrical impulse doesn't travel through the heart correctly.",
            'HYP': 'Abnormal thickening of the heart muscle.',
        },

        'guia_header_derivaciones': "What is a 12-lead ECG?",
        'guia_texto_derivaciones': (
            "Imagine 12 cameras placed at different angles around the heart: they're "
            "all recording the same heartbeat, but each one sees it from a different "
            "point of view. That's what **leads** are: each one picks up certain "
            "things better than others, depending on which part of the heart it's "
            "\"looking at\" most closely.\n\nThey're grouped into 3 families, based on "
            "where the electrodes are placed on the body:"
        ),
        'guia_col1_titulo': "**Limb leads**",
        'guia_col1_lista': "I, II, III",
        'guia_col1_caption': "Electrodes on the arms and legs. They form a triangle around the heart.",
        'guia_col2_titulo': "**Augmented leads**",
        'guia_col2_lista': "aVR, aVL, aVF",
        'guia_col2_caption': "Same electrodes as above, mathematically combined to give additional angles.",
        'guia_col3_titulo': "**Precordial (chest) leads**",
        'guia_col3_lista': "V1 – V6",
        'guia_col3_caption': "Electrodes placed directly on the chest, very close to the heart.",
        'guia_tabla_titulo': "#### Which area of the heart does each one see?",
        'guia_tabla': (
            "| Leads | Heart region observed |\n"
            "|---|---|\n"
            "| II, III, aVF | Inferior wall |\n"
            "| I, aVL, V5, V6 | Lateral wall |\n"
            "| V1, V2 | Septum (between the two ventricles) |\n"
            "| V3, V4 | Anterior wall (front) |\n"
        ),
        'guia_header_diagnosticos': "What does this model diagnose?",
        'guia_intro_diagnosticos': "The model evaluates 5 categories. A single ECG can belong to more than one at the same time.",
        'guia_diagnosticos': {
            'NORM': ("**Normal (NORM)**", "The ECG doesn't show any of the following 4 abnormalities."),
            'MI': ("**Myocardial infarction (MI)**",
                   "Commonly known as a 'heart attack': part of the heart muscle "
                   "doesn't receive enough blood (and therefore oxygen), which can "
                   "damage it if not treated in time."),
            'STTC': ("**ST/T changes (STTC)**",
                     "The ECG has a characteristic shape, like a mountain with several "
                     "peaks and valleys. The 'ST segment' and the 'T wave' are two "
                     "specific parts of that shape; when they change from what's "
                     "expected, it can indicate poor blood flow, inflammation, or "
                     "other imbalances."),
            'CD': ("**Conduction disturbance (CD)**",
                   "The heart beats thanks to an electrical impulse that travels "
                   "through internal 'wiring' in a specific order. When that impulse "
                   "gets delayed or blocked somewhere along the way, this type of "
                   "disturbance occurs (for example, 'bundle branch blocks')."),
            'HYP': ("**Hypertrophy (HYP)**",
                    "The heart muscle has thickened more than normal, often as a "
                    "result of having to work against elevated resistance for a long "
                    "time (for example, uncontrolled high blood pressure)."),
        },
        'guia_info_final': "💡 This guide is a simplification for educational purposes. For any questions about a real ECG, always consult a healthcare professional.",
    },
}


def t(clave):
    """Devuelve el texto correspondiente a la clave, en el idioma actualmente
    seleccionado. Es el único punto del código que sabe 'en qué idioma estamos'."""
    return TEXTOS[st.session_state.idioma][clave]


# ============================================================
# CONFIGURACIÓN DE PÁGINA E IDIOMA
# ============================================================
st.set_page_config(page_title="ECG Structural Heart Disease Detection", page_icon="🫀", layout="wide")

if 'idioma' not in st.session_state:
    st.session_state.idioma = 'es'

_idioma_visible = st.sidebar.radio(
    "idioma", ["Español", "English"],
    index=0 if st.session_state.idioma == 'es' else 1,
    horizontal=True, label_visibility="collapsed",
)
st.session_state.idioma = 'es' if _idioma_visible == "Español" else 'en'


@st.cache_data
def cargar_ejemplos():
    datos = np.load('app/ejemplos_demo.npz')
    return datos['señales'], datos['etiquetas'], datos['edades'], datos['sexos']


def llamar_api(endpoint, payload, mensaje_carga):
    try:
        with st.spinner(mensaje_carga):
            respuesta = requests.post(f"{API_URL}/{endpoint}", json=payload, timeout=10)
        respuesta.raise_for_status()
        return respuesta.json()
    except requests.exceptions.ConnectionError:
        st.error(t('error_conexion'))
        st.stop()
    except requests.exceptions.Timeout:
        st.error(t('error_timeout'))
        st.stop()


señales, etiquetas, edades, sexos = cargar_ejemplos()

st.title(t('titulo_principal'))
st.warning(t('aviso_demo'))

tab_diagnostico, tab_guia = st.tabs([t('tab_diagnostico'), t('tab_guia')])

# ============================================================
# PESTAÑA 1: DIAGNÓSTICO
# ============================================================
with tab_diagnostico:

    st.sidebar.header(t('sidebar_header'))

    etiquetas_texto = []
    for i in range(len(señales)):
        clases_positivas = [t('nombres_clases')[CLASES[j]] for j in range(5) if etiquetas[i, j] == 1]
        etiquetas_texto.append(t('ejemplo_formato').format(n=i + 1, clases=", ".join(clases_positivas)))

    indice_seleccionado = st.sidebar.selectbox(
        "casos", range(len(señales)), format_func=lambda i: etiquetas_texto[i], label_visibility="collapsed"
    )

    if st.sidebar.button(t('boton_azar')):
        indice_seleccionado = np.random.randint(0, len(señales))
        st.rerun()

    señal_elegida = señales[indice_seleccionado]
    edad_paciente = edades[indice_seleccionado]
    sexo_paciente = sexos[indice_seleccionado]
    etiqueta_real = etiquetas[indice_seleccionado]

    st.sidebar.markdown("---")
    edad_texto = t('texto_90_mas') if edad_paciente == 300 else f"{int(edad_paciente)}"
    sexo_texto = t('texto_hombre') if sexo_paciente == 0 else t('texto_mujer')
    st.sidebar.markdown(t('label_edad').format(edad=edad_texto))
    st.sidebar.markdown(t('label_sexo').format(sexo=sexo_texto))

    col_izquierda, col_derecha = st.columns([1.3, 1])

    with col_izquierda:
        st.subheader(t('subheader_señal'))
        ver_12_derivaciones = st.checkbox(t('checkbox_12_derivaciones'), value=True)

        if ver_12_derivaciones:
            fig, axes = plt.subplots(3, 4, figsize=(11, 5.5), sharex=True)
            for pos, idx_derivacion in enumerate(ORDEN_GRID_CLINICO):
                fila, columna = divmod(pos, 4)
                ax = axes[fila, columna]
                ax.plot(señal_elegida[:, idx_derivacion], color=COLOR_TRAZO, linewidth=0.8)
                ax.set_title(NOMBRES_DERIVACIONES[idx_derivacion], fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            st.pyplot(fig)
            st.caption(t('caption_grid_clinico'))
        else:
            derivacion = st.selectbox(
                t('label_derivacion_mostrar'), range(12), index=1,
                format_func=lambda d: t('formato_derivacion').format(d=NOMBRES_DERIVACIONES[d])
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(señal_elegida[:, derivacion], color=COLOR_TRAZO, linewidth=1.2)
            st.pyplot(fig)

    with col_derecha:
        st.subheader(t('subheader_diagnostico'))

        if st.button(t('boton_analizar'), type="primary"):
            probabilidades = llamar_api(
                "predecir_preprocesado", {"señal": señal_elegida.tolist()}, t('spinner_analizando')
            )
            st.session_state['probabilidades'] = probabilidades
            st.session_state['señal_analizada'] = señal_elegida.tolist()

        if 'probabilidades' in st.session_state:
            probabilidades = st.session_state['probabilidades']

            for clase in CLASES:
                prob = probabilidades[clase]
                umbral = UMBRALES[clase]
                es_positivo = prob >= umbral
                real_positivo = etiqueta_real[CLASES.index(clase)] == 1

                color = "🔴" if es_positivo else "⚪"
                acierto = "✅" if es_positivo == real_positivo else "❌"

                st.markdown(f"{color} **{t('nombres_clases')[clase]}**: {prob*100:.1f}% {acierto}")
                st.progress(min(prob, 1.0))
                st.caption(t('descripciones_clases')[clase])

            st.caption(t('caption_leyenda'))

    st.markdown("---")
    st.subheader(t('subheader_explicacion'))

    if 'probabilidades' in st.session_state:
        clase_a_explicar = st.selectbox(
            t('label_explicar_para'), CLASES,
            index=int(np.argmax([st.session_state['probabilidades'][c] for c in CLASES])),
            format_func=lambda c: t('nombres_clases')[c],
        )

        if st.button(t('boton_gradcam')):
            resultado = llamar_api(
                "explicar_preprocesado",
                {"señal": st.session_state['señal_analizada'], "indice_clase": CLASES.index(clase_a_explicar)},
                t('spinner_explicando')
            )
            cam = np.array(resultado['cam'])

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.imshow(
                cam[np.newaxis, :], cmap='Reds', aspect='auto', alpha=0.5,
                extent=[0, len(cam), señal_elegida[:, 1].min(), señal_elegida[:, 1].max()]
            )
            ax2.plot(señal_elegida[:, 1], color=COLOR_TRAZO, linewidth=1.2)
            ax2.set_title(t('titulo_gradcam_formato').format(
                clase=t('nombres_clases')[clase_a_explicar], prob=f"{resultado['probabilidad']*100:.1f}"
            ))
            st.pyplot(fig2)
            st.caption(t('caption_gradcam'))
    else:
        st.info(t('info_pulsa_analizar'))

    st.markdown("---")
    st.caption(t('footer_caption'))

# ============================================================
# PESTAÑA 2: GUÍA
# ============================================================
with tab_guia:

    st.header(t('guia_header_derivaciones'))
    st.markdown(t('guia_texto_derivaciones'))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(t('guia_col1_titulo'))
        st.markdown(t('guia_col1_lista'))
        st.caption(t('guia_col1_caption'))
    with col2:
        st.markdown(t('guia_col2_titulo'))
        st.markdown(t('guia_col2_lista'))
        st.caption(t('guia_col2_caption'))
    with col3:
        st.markdown(t('guia_col3_titulo'))
        st.markdown(t('guia_col3_lista'))
        st.caption(t('guia_col3_caption'))

    st.markdown(t('guia_tabla_titulo'))
    st.markdown(t('guia_tabla'))

    st.markdown("---")
    st.header(t('guia_header_diagnosticos'))
    st.markdown(t('guia_intro_diagnosticos'))

    for clase in CLASES:
        titulo, texto = t('guia_diagnosticos')[clase]
        st.markdown(titulo)
        st.write(texto)

    st.markdown("---")
    st.info(t('guia_info_final'))