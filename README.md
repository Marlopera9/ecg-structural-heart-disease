# Detección de Enfermedad Cardíaca Estructural mediante ECG

Proyecto de deep learning para predecir anomalías cardíacas estructurales 
usando únicamente señales de electrocardiograma (ECG) de 12 derivaciones, 
sin necesidad de un ecocardiograma.

## Estado del proyecto: Exploración de datos completada (Fase 1/4)

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

## 🔍 Hallazgos de la exploración inicial (EDA)

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