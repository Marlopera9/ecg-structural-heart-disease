# Detección de Enfermedad Cardíaca Estructural mediante ECG

Proyecto de deep learning para predecir anomalías cardíacas estructurales 
usando únicamente señales de electrocardiograma (ECG) de 12 derivaciones, 
sin necesidad de un ecocardiograma.

## Estado del proyecto: Exploración de datos completada (Fase 3/4)

## Motivación

El diagnóstico de enfermedades cardíacas estructurales normalmente requiere 
un ecocardiograma: una prueba costosa y que depende de la disponibilidad de 
un especialista. Este proyecto explora si un ECG estándar —mucho más barato, 
rápido y accesible— contiene suficiente información para que un modelo de 
deep learning prediga estas anomalías, actuando como una primera herramienta 
de cribado.

## Dataset

Se utiliza [PTB-XL](https://physionet.org/content/ptb-xl/), un dataset público 
que contiene 21.799 registros de ECG de 12 derivaciones, pertenecientes a 
18.869 pacientes distintos.

## Hallazgos de la exploración inicial (EDA)

Antes de tocar cualquier modelo, se realizó un análisis exploratorio para 
entender bien la naturaleza de los datos. Estos son los hallazgos más 
relevantes y cómo afectan a las decisiones técnicas del proyecto:

### 1. Es un problema multi-etiqueta, no de una sola clase
Un mismo ECG puede tener varios diagnósticos a la vez (por ejemplo, un 
paciente puede presentar simultáneamente una arritmia y un problema de 
conducción). En este dataset, cada ECG tiene en promedio 1,27 
diagnósticos, y hay registros con hasta 4 diagnósticos simultáneos. Esto 
significa que el modelo no puede diseñarse como "elige una opción entre 
varias" (multi-clase), sino como "decide, para cada posible enfermedad, si 
está presente o no" (multi-etiqueta) — esto afectará directamente a la 
función de pérdida usada al entrenar el modelo.

### 2. Se excluyen los registros sin diagnóstico claro
Se detectaron **411 registros (≈1,9%)** que no encajan en ninguna de las 5 
categorías diagnósticas principales del dataset. Al ser un porcentaje muy 
pequeño y no aportar una etiqueta útil para el aprendizaje, se descartarán 
del conjunto de entrenamiento.

### 3. El dataset está desbalanceado entre categorías
La distribución de las 5 categorías diagnósticas principales es la siguiente:

| Categoría                     | Registros | Porcentaje |
|-------------------------------|-----------|------------|
| NORM (normal)                 | 9.514     | 43,6%      |
| MI (infarto de miocardio)     | 5.469     | 25,1%      |
| STTC (cambios ST/T)           | 5.235     | 24,0%      |
| CD (trastornos de conducción) | 4.898     | 22,5%      |
| HYP (hipertrofia)             | 2.649     | 12,2%      |

*(Los porcentajes suman más de 100% porque, como se explicó arriba, un mismo 
ECG puede pertenecer a varias categorías.)* La categoría `HYP` es 
significativamente minoritaria frente a `NORM`. Esto se tendrá en cuenta en 
la fase de entrenamiento mediante funciones de pérdida especializadas 
(Focal Loss) para evitar que el modelo "ignore" las clases minoritarias.

### 4. Evitar fuga de datos entre pacientes (data leakage)
Aunque hay 21.799 ECGs, solo corresponden a 18.869 pacientes únicos: 2.930 
pacientes tienen más de un ECG registrado. Dividir los datos de forma 
aleatoria por ECG sería un error grave, ya que el mismo paciente podría 
aparecer tanto en entrenamiento como en test, haciendo que el modelo 
memorice la "firma" particular de ese paciente en vez de aprender patrones 
generales de la enfermedad. Para evitarlo, se utiliza la columna oficial 
`strat_fold` (incluida por los creadores del dataset), que garantiza una 
división correcta sin solapamiento de pacientes entre grupos:
- **Folds 1-8** → entrenamiento
- **Fold 9** → validación
- **Fold 10** → test

Este split es además el estándar usado en la literatura científica sobre 
PTB-XL, lo que permite comparar los resultados de este proyecto con 
benchmarks publicados.

### 5. Variables descartadas por exceso de datos faltantes
Las columnas `height` (altura) y `weight` (peso) presentan un 67,9% y 56,7% 
de valores faltantes respectivamente. Al ser una proporción demasiado alta 
para imputar de forma fiable, se descartan como variables de entrada del 
modelo.

### 6. Anonimización de edad 
Se detectó que las edades de pacientes de 90 años o más aparecen codificadas 
como el valor **300** en el dataset. Tras revisar la documentación oficial de 
PTB-XL, se confirmó que esto es intencional: es una técnica de anonimización 
para cumplir con el estándar HIPAA de protección de datos sanitarios, ya que 
las edades muy avanzadas son estadísticamente más fáciles de usar para 
identificar a una persona real. Estos valores se tratarán como una categoría 
especial ("90 años o más"), no como una edad numérica real.

### 7. Marcapasos: exclusión en esta primera versión
285 registros corresponden a pacientes con marcapasos. Dado que un 
marcapasos altera significativamente la morfología de la señal ECG (y 
podría hacer que el modelo aprenda a detectar el marcapasos en lugar de la 
patología real), estos registros se excluyen del entrenamiento inicial. Se 
propone como línea de trabajo futuro entrenar un modelo específico para 
esta población.
### 8. Ruido técnico de la señal
El dataset incluye anotaciones de ruido eléctrico (`static_noise`, 
`burst_noise`) y problemas de electrodos, registradas como texto libre (en 
alemán) indicando qué derivaciones se vieron afectadas. Dada su baja 
frecuencia y formato poco estructurado, se creará una variable binaria 
simple (`tiene_ruido_o_problema`) para poder analizar más adelante si el 
rendimiento del modelo empeora en estos casos, sin necesidad de interpretar 
cada código individualmente.

### 9. Distribución por sexo
El dataset está bien balanceado por sexo: 52,1% (sexo 0) frente a 47,9% 
(sexo 1). No se requieren ajustes adicionales en este aspecto.
## Estructura del repositorio

```
├── data/              # Dataset (excluido de git, ver .gitignore)
├── notebooks/         # Notebooks de exploración y experimentación
│   └── 01_exploracion_datos.ipynb
├── src/                # Código de producción (preprocesamiento, modelo, API)
├── requirements.txt   # Dependencias del proyecto
└── README.md
```
## Cómo reproducir el entorno

```bash
git clone https://github.com/tuusuario/ecg-structural-heart-disease.git
cd ecg-structural-heart-disease
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

El dataset debe descargarse manualmente desde 
[PhysioNet](https://physionet.org/content/ptb-xl/) y colocarse en la carpeta 
`data/`.

## Próximos pasos

- [x] Exploración de metadatos y calidad de datos (EDA)
- [x] Preprocesamiento de señal: filtrado, normalización, exclusión de registros 
- [x] Arquitectura CNN 1D en PyTorch
- [x] Entrenamiento con manejo de desbalanceo de clases (Focal Loss)
- [x] Evaluación con AUROC / AUPRC por categoría diagnóstica
- [ ] Explicabilidad con Grad-CAM 1D
- [ ] Despliegue como API con FastAPI

## Consideraciones éticas y de sesgos

Este proyecto tiene fines exclusivamente educativos y de demostración 
técnica. No debe utilizarse como herramienta de diagnóstico real. Cualquier 
modelo entrenado hereda las características demográficas y clínicas del 
dataset PTB-XL (proveniente de un centro médico en Alemania), por lo que su 
rendimiento podría no generalizar a poblaciones con características 
demográficas distintas.

## Preprocesamiento de la señal

Cada ECG pasa por dos transformaciones antes de entrar al modelo:

1. **Filtrado paso-banda (0.5-40 Hz)**: elimina la deriva de línea base 
   (causada por la respiración, frecuencias muy bajas) y el ruido de alta 
   frecuencia (interferencias eléctricas). Se validó su correcto 
   funcionamiento con una señal sintética de control antes de aplicarlo al 
   dataset real. Sobre las señales de PTB-XL, el filtro elimina en promedio 
   un ~15% de la "energía" de la señal — un efecto sutil pero real, 
   consistente con el uso de equipos de grabación clínicos de buena calidad.
2. **Normalización Z-score** por derivación: cada una de las 12 derivaciones 
   se normaliza independientemente a media 0 y desviación estándar 1, 
   evitando que derivaciones con amplitudes naturalmente mayores dominen 
   el aprendizaje.

Los datos procesados se dividen en train/val/test usando el `strat_fold` 
oficial de PTB-XL (documentado en la Fase 1), y se guardan en formato 
`.npy` para una carga eficiente durante el entrenamiento.

## Arquitectura del modelo

Se implementó una red neuronal convolucional 1D (CNN 1D) en PyTorch, 
diseñada específicamente para series temporales biomédicas:

- 3 bloques convolucionales (32 → 64 → 128 canales), cada uno con 
  Batch Normalization, activación ReLU y MaxPooling, para detectar 
  patrones morfológicos a distintas escalas temporales.
- Un clasificador final con Dropout (0.3) para reducir el riesgo de 
  sobreajuste.
- Salida de 5 valores (uno por categoría diagnóstica: NORM, MI, STTC, CD, 
  HYP), tratando el problema como clasificación multi-etiqueta.

Se verificó el correcto funcionamiento de principio a fin (carga de datos → 
Dataset → DataLoader → modelo) mediante un smoke test con un batch de 
ejemplo, antes de proceder a la fase de entrenamiento.

## Problemas encontrados y soluciones en la Fase 2

Durante esta fase surgieron varios problemas técnicos reales. Se documentan 
aquí porque forman parte del proceso de aprendizaje y pueden ahorrar tiempo 
a cualquiera que reproduzca este proyecto en Windows.

**1. Aviso falso de "módulo no encontrado" en el editor**
El editor marcaba `wfdb` como no encontrado en los archivos `.py`, aunque el 
notebook sí lo reconocía sin problema. Causa: VS Code usa dos configuraciones 
de Python distintas — una para los notebooks (el "kernel") y otra para los 
archivos `.py` sueltos (el "intérprete del editor"). Bastaba con fijar 
manualmente el intérprete del editor al entorno `venv` del proyecto 
(`Python: Select Interpreter`) para que ambas configuraciones coincidieran.

**2. El filtro de la señal parecía no hacer nada**
Al comparar visualmente la señal ECG antes y después del filtrado, las 
gráficas parecían casi idénticas. Para comprobar si era un error de código o 
un rasgo real del dataset, se validó el filtro con una señal sintética 
generada a propósito con ruido conocido, y se midió numéricamente la 
diferencia (residuo) entre la señal cruda y la filtrada. Conclusión: el 
filtro funcionaba correctamente; el efecto era sutil porque las señales de 
PTB-XL ya vienen relativamente limpias, al proceder de equipos clínicos.

**3. Error al guardar los datos procesados (`FileNotFoundError`)**
Tras procesar los más de 21.000 ECGs (proceso de varios minutos), el guardado 
final falló porque la carpeta de destino se había creado con un error 
tipográfico (`proccesed` en vez de `processed`). Se corrigió el nombre de la 
carpeta y se repitió el procesamiento.

**4. Error de carga de PyTorch (`OSError: WinError 1114`)**
Al importar `torch` por primera vez, Windows no conseguía cargar uno de sus 
archivos internos (`c10.dll`). Se descartaron como causa tanto la falta del 
Visual C++ Redistributable como un posible conflicto con la sincronización de 
OneDrive (ambos verificados explícitamente). La solución fue reinstalar 
PyTorch fijando una versión concreta y estable (`torch==2.4.1`) en lugar de 
la última versión disponible, que presentaba el conflicto.

**5. `ModuleNotFoundError: No module named 'src'` tras reiniciar el kernel**
Después de reiniciar el kernel del notebook (necesario para solucionar el 
problema anterior), las importaciones desde `src/` dejaron de funcionar. 
Causa: la configuración que añade la carpeta raíz del proyecto a las rutas de 
búsqueda de Python (`sys.path.append('..')`) solo persiste mientras el kernel 
está activo, y debe volver a ejecutarse tras cada reinicio.

## 🏋️ Entrenamiento

Se entrenó el modelo `ECG_CNN1D` usando **Focal Loss** (α=0.25, γ=2.0) en 
lugar de una función de pérdida estándar, precisamente para compensar el 
desbalanceo de clases detectado en la Fase 1 (la clase `HYP` representa solo 
un 12,2% de los registros frente al 43,6% de `NORM`). Esta función reduce 
el peso de aprendizaje de los ejemplos que el modelo ya clasifica bien, 
obligándolo a prestar más atención relativa a los casos difíciles o 
minoritarios.

El modelo se entrenó con el optimizador Adam (tasa de aprendizaje 1e-3), 
guardando en cada época únicamente si mejoraba el AUROC macro en el 
conjunto de **validación** (nunca en entrenamiento, para evitar decisiones 
sesgadas).

![Curvas de entrenamiento](assets/curvas_entrenamiento.png)

### Experimento: ¿merece la pena entrenar más épocas?

Se realizó una prueba entrenando 40 épocas (el doble de lo inicialmente 
planeado) para comprobar si el modelo seguía mejorando. El resultado fue 
claro: a partir de aproximadamente la época 10, la pérdida de validación 
empezó a subir de forma sostenida mientras la de entrenamiento seguía 
bajando — la señal clásica de **sobreajuste** (overfitting), es decir, el 
modelo memorizando detalles del set de entrenamiento en vez de aprender 
patrones generalizables. En consonancia, el AUROC de validación alcanzó su 
mejor valor en la **época 8** (0,9106) y fue empeorando gradualmente después, 
pese al ruido natural entre épocas.

Gracias a la estrategia de guardar solo el mejor modelo según AUROC de 
validación, el modelo final utilizado en este proyecto corresponde a esa 
época temprana y no se ve afectado por el sobreajuste posterior — 
demostrando en la práctica por qué esta estrategia de checkpointing es 
una buena práctica estándar en Machine Learning.

## Resultados en el conjunto de test

Evaluación final sobre el conjunto de test (fold 10), datos que el modelo 
no vio en ningún momento durante el entrenamiento ni la validación:

| Clase | AUROC | AUPRC | Prevalencia en el dataset |
|-------|-------|-------|---------------------------|
| NORM  | 0.940 | 0.919 |         43,6%             |
| MI    | 0.926 | 0.826 |         25,1%             |
| STTC  | 0.931 | 0.821 |         24,0%             |
| CD    | 0.914 | 0.832 |         22,5%             |
| HYP   | 0.829 | 0.481 |         12,2%             |

**AUROC macro: 0,908** &nbsp;|&nbsp; **AUPRC macro: 0,776**

![Curvas ROC](assets/curvas_roc.png)
![Curvas Precision-Recall](assets/curvas_precision_recall.png)

### Cómo interpretar estos resultados

- **AUROC** mide qué tan bien distingue el modelo entre un ECG con la 
  patología y uno sin ella, en general. Un valor de 0,908 significa que, en 
  aproximadamente 9 de cada 10 comparaciones, el modelo puntúa más alto al 
  ECG que realmente tiene la anomalía.
- **AUPRC** es más exigente en datasets desbalanceados: mide cuánto acierta 
  el modelo entre todo lo que marca como positivo, y cuántos casos reales 
  consigue detectar. Por eso se reporta junto al AUROC, en vez de solo este 
  último.
- **La clase `HYP` es, con diferencia, la más difícil para el modelo** 
  (AUROC 0,829, AUPRC 0,481), consistente con ser la clase minoritaria del 
  dataset. Aun así, su AUPRC es casi 4 veces superior al que se obtendría 
  adivinando al azar (≈0,122, su prevalencia), lo que indica que el modelo 
  sí aprendió señal real sobre esta clase, aunque con menos confianza que 
  en las demás.

### Nota importante sobre el alcance de estos resultados

Los diagnósticos de PTB-XL (`NORM`, `MI`, `STTC`, `CD`, `HYP`) son hallazgos 
que un cardiólogo puede identificar leyendo directamente el ECG — de hecho, 
así se etiquetó originalmente el dataset. Este proyecto demuestra que un 
modelo puede automatizar esa lectura con alta fiabilidad, pero es un reto 
distinto y más accesible al de detectar patología estructural que **no** 
es evidente en una lectura convencional del ECG (el objetivo, por ejemplo, 
del dataset EchoNext, mencionado como posible extensión futura de este 
proyecto).