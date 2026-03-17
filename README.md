# GUION DE CLASES - SEMANAS 7 Y 8
## Regresión Lineal Simple y Métricas de Error en Ingeniería de Software
### Curso: Herramientas Computacionales para Interpretación y Validación de Resultados
### Programa: Ingeniería de Software | Universidad Cooperativa de Colombia

---

## SEMANA 7: MODELADO Y REGRESIÓN LINEAL SIMPLE

### **Objetivo General de la Semana**

Que el estudiante comprenda los fundamentos de regresión lineal simple, implemente un modelo predictivo en Python usando scikit-learn y Google Colab, aplicado a un problema real de ingeniería de software, y visualice el ajuste del modelo.

### **Objetivos Específicos**

1. Identificar variables dependientes e independientes en contextos de ingeniería de software.
2. Comprender el concepto de mínimos cuadrados y recta de mejor ajuste.
3. Implementar regresión lineal simple con scikit-learn en Google Colab.
4. Visualizar de forma clara la relación entre variables y el modelo ajustado.
5. Interpretar de forma crítica si el modelo es útil para tomar decisiones.

### **Competencias a Desarrollar**

- **Saber:** Conceptos de variables en regresión, supuestos básicos, significado de la pendiente e intercepto.
- **Hacer:** Implementación en Python, manejo de datasets, visualización.
- **Ser:** Reflexión crítica sobre aplicabilidad del modelo, comunicación clara de resultados.

---

### **CONTENIDOS CLAVE**

#### **1. Repaso Rápido: Preparación de Datos con Pandas y NumPy** (10 min)

Aunque los estudiantes ya conocen pandas de semanas anteriores, recordar brevemente:
- Carga de datos: `pd.read_csv()`, `.head()`, `.info()`, `.describe()`
- Selección de columnas: `df[['col1', 'col2']]`
- Conversión a arrays NumPy: `.values`
- Manejo de valores faltantes: `.dropna()`, `.fillna()`

**Código de referencia:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar datos
df = pd.read_csv('datos_rendimiento_software.csv')
print(df.head())
print(df.info())
print(df.describe())

# Limpiar datos (si es necesario)
df = df.dropna()
```

#### **2. Conceptos Teóricos: Regresión Lineal Simple**

**Definición:** Modelar la relación lineal entre una variable dependiente (Y, respuesta) y una variable independiente (X, predictor) mediante la ecuación:

```
Y = a + b*X + ε
```

Donde:
- `a` = intercepto (valor de Y cuando X=0)
- `b` = pendiente (cambio en Y por cada unidad de cambio en X)
- `ε` = error aleatorio (residuos)

**Supuestos básicos (sin profundizar demasiado):**
1. Relación lineal entre X e Y.
2. Errores independientes y distribuidos normalmente.
3. Varianza constante de errores.

**Método de ajuste:** Mínimos cuadrados ordinarios (OLS): minimizar la suma de cuadrados de los residuos.

#### **3. Ejemplos en Ingeniería de Software**

**Ejemplo A: Predicción de Tiempo de Respuesta**
- **X (variable independiente):** número de peticiones simultáneas a un servidor.
- **Y (variable dependiente):** tiempo de respuesta promedio en milisegundos.
- **Interpretación:** ¿Cómo cambia el tiempo de respuesta al aumentar la carga?

**Ejemplo B: Predicción de Defectos por Módulo**
- **X:** líneas de código (LOC) de un módulo.
- **Y:** número de defectos reportados después de release.
- **Interpretación:** ¿Módulos más grandes tienen más defectos?

**Ejemplo C: Esfuerzo vs. Complejidad**
- **X:** complejidad ciclomática de una función.
- **Y:** tiempo de desarrollo en horas.
- **Interpretación:** ¿Funciones más complejas requieren más tiempo?

#### **4. Implementación en Google Colab con Scikit-learn**

**Paso 1: Preparar los datos**
```python
# Crear arrays X (independiente) e Y (dependiente)
X = df[['numero_peticiones']].values  # Matriz 2D
y = df['tiempo_respuesta'].values     # Vector 1D
```

**Paso 2: Crear e instanciar el modelo**
```python
# Instanciar el modelo
modelo = LinearRegression()
```

**Paso 3: Entrenar el modelo**
```python
# Ajustar el modelo a los datos
modelo.fit(X, y)
```

**Paso 4: Obtener parámetros**
```python
# Coeficientes del modelo
intercepto = modelo.intercept_
pendiente = modelo.coef_[0]

print(f"Ecuación del modelo: Y = {intercepto:.2f} + {pendiente:.2f}*X")
```

**Paso 5: Hacer predicciones**
```python
# Predicciones en los datos de entrenamiento
y_pred = modelo.predict(X)

# Predicción para nuevos valores
X_nuevo = np.array([[100], [150], [200]])
y_nuevo = modelo.predict(X_nuevo)
print(f"Predicciones: {y_nuevo}")
```

#### **5. Visualización del Ajuste del Modelo**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Gráfico de dispersión
plt.scatter(X, y, alpha=0.6, label='Datos observados', color='steelblue')

# Recta de regresión
X_linea = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_linea = modelo.predict(X_linea)
plt.plot(X_linea, y_linea, 'r-', linewidth=2, label='Recta de regresión')

# Etiquetas y títulos
plt.xlabel('Número de Peticiones', fontsize=12)
plt.ylabel('Tiempo de Respuesta (ms)', fontsize=12)
plt.title('Regresión Lineal: Tiempo de Respuesta vs. Carga', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Opcional: mostrar ecuación en el gráfico
ecuacion = f'Y = {intercepto:.2f} + {pendiente:.4f}*X'
plt.text(0.05, 0.95, ecuacion, transform=plt.gca().transAxes, 
         fontsize=11, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
```

---

### **SECUENCIA DE ACTIVIDADES - SEMANA 7**

#### **Bloque 1: Apertura y Motivación (15 min)**

**Actividad de Apertura - Mini Quiz Interactivo:**

1. **Pregunta disparadora (5 min):**
   - *"En una empresa de software, el equipo de operaciones observa que el tiempo de respuesta de una API aumenta cuando hay más usuarios conectados. ¿Cómo podrían predecir cuánto tiempo tardará la API si en las próximas horas hay 500 usuarios simultáneos? ¿Qué datos necesitarían?"*
   - Dejar que los estudiantes respondan brevemente (lluvia de ideas).

2. **Mini-quiz (5 min):**
   - ¿Cuál es la variable independiente (X) en este escenario?
   - ¿Cuál es la variable dependiente (Y)?
   - ¿Qué tipo de relación esperaríamos?

3. **Conexión con el curso (5 min):**
   - Explicar brevemente que hoy aprenderán a modelar relaciones como esta usando regresión lineal.
   - Vincular con Momento II del curso: "Hemos explorado datos, ahora los modelamos para hacer predicciones."

---

#### **Bloque 2: Exposición Teórica y Demostración en Vivo (40 min)**

**Estructura:**

1. **Conceptos teóricos (15 min):**
   - Proyectar diapositiva o notebook con la ecuación: `Y = a + b*X`
   - Explicar qué es pendiente (b): *"Por cada aumento unitario en X, Y aumenta b unidades."*
   - Explicar qué es intercepto (a): *"Valor de Y cuando X = 0."*
   - Dibujar en la pizarra o notebook 2–3 ejemplos visuales de rectas con diferentes pendientes (positiva, negativa, cercana a cero).
   - Mencionar brevemente el método de mínimos cuadrados sin entrar en demostraciones matemáticas.

2. **Ejemplos en ingeniería de software (10 min):**
   - Presentar los 3 ejemplos mencionados (tiempo respuesta, defectos, esfuerzo).
   - Para cada uno, preguntar: *¿Esperaría una relación lineal? ¿Positiva o negativa?*

3. **Demo en vivo con Google Colab (15 min):**
   - Abrir Google Colab en pantalla compartida.
   - Cargar un dataset pre-preparado (pequeño, 20–30 registros) de ejemplo.
   - Ejecutar línea por línea el código de preparación, entrenamiento y predicción.
   - Mostrar los parámetros del modelo: intercepto y pendiente.
   - Crear el gráfico de dispersión + recta.
   - **Puntos clave a enfatizar:**
     - Código bien estructurado: comentarios, nombres claros.
     - Salida interpretable: qué significan los números.

---

#### **Bloque 3: Actividad Práctica Central (80 min)**

**Contexto:** Los estudiantes reciben un dataset de ejemplo y replican el proceso en sus propios notebooks de Colab.

**Estructura propuesta:**

1. **Distribución de datasets (5 min):**
   - Proporcionar 2–3 opciones de datasets pequeños (15–30 registros) ya limpios.
   - Opción 1: Datos de rendimiento de un servidor (carga vs. latencia).
   - Opción 2: Datos de defectos por módulo (LOC vs. cantidad de bugs reportados).
   - Opción 3: Datos de esfuerzo (complejidad vs. horas de desarrollo).
   - Los estudiantes (individual o en parejas) eligen uno.

2. **Paso 1: Carga de datos (10 min):**
   - Crear un nuevo notebook en Google Colab.
   - Cargar el dataset usando pandas.
   - Exploración inicial: `.head()`, `.describe()`, verificar datos faltantes.
   - **Checkpoint:** profesor circula, revisa que todos tengan datos cargados.

3. **Paso 2: Visualización inicial (15 min):**
   - Crear un gráfico de dispersión básico de X vs. Y.
   - Discusión breve: *"¿Observan una relación lineal? ¿Hay outliers?"*
   - **Checkpoint:** visualizaciones correctas.

4. **Paso 3: Construcción del modelo (20 min):**
   - Crear el modelo `LinearRegression()`.
   - Ajustarlo con `.fit()`.
   - Extraer y documentar intercepto y pendiente.
   - **Checkpoint:** modelos entrenados, parámetros visibles en celdas.

5. **Paso 4: Visualización mejorada (15 min):**
   - Añadir la recta de regresión al gráfico.
   - Incluir ecuación, títulos, leyendas.
   - **Checkpoint:** gráficos profesionales.

6. **Paso 5: Reflexión escrita (15 min):**
   - Añadir una celda markdown con preguntas:
     - *¿Qué significa la pendiente que obtuviste? ¿Tiene sentido en el contexto del problema?*
     - *¿Hay puntos que se desvíen mucho de la recta? ¿Por qué podrían existir?*
     - *¿Usarías este modelo para tomar decisiones en un equipo de software real?*
   - Los estudiantes escriben 3–5 oraciones respondiendo.

**Apoyo del docente durante la actividad:**
- Circulación constante, revisión de código en vivo.
- Responder preguntas, mostrar errores comunes.
- Ejemplos de código proyectados si hay dudas generales.

---

#### **Bloque 4: Cierre y Reflexión Colectiva (15 min)**

1. **Revisión de resultados (7 min):**
   - Pedir a 2–3 estudiantes que compartan su modelo brevemente.
   - Proyectar sus gráficos en pantalla compartida.
   - Discusión: *"¿Las pendientes son similares? ¿Por qué o por qué no?"*

2. **Síntesis y preguntas (5 min):**
   - *"Hoy aprendieron a ajustar una recta a datos reales. La semana próxima calcularemos qué tan bien ajusta esa recta."*
   - *"¿Preguntas sobre lo visto?"*

3. **Asignación de tarea (3 min):**
   - Guardar el notebook de hoy y preparar la actividad de la semana 8.
   - Traer anotaciones sobre posibles mejoras del modelo (por ejemplo, ¿qué pasaría si filtro outliers?).

---

### **RECURSOS Y ARCHIVOS PARA SEMANA 7**

#### **Dataset Sugerido 1: Rendimiento de API**
```csv
peticiones_simultaneas,tiempo_respuesta_ms
10,45
15,62
20,78
25,85
30,98
35,112
40,125
45,138
50,152
```

#### **Dataset Sugerido 2: Defectos por Módulo**
```csv
lineas_codigo,defectos_reportados
150,2
320,5
480,7
200,3
1100,15
450,6
800,10
600,8
350,4
```

#### **Código Base a Proporcionar** (Notebook Template)

Los estudiantes pueden copiar esta estructura:

```python
# ============================================================
# SEMANA 7: REGRESIÓN LINEAL SIMPLE
# Curso: Herramientas Computacionales para Interpretación y Validación de Resultados
# Estudiante: [nombre]
# Fecha: [fecha]
# ============================================================

# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================
# PASO 1: CARGAR Y EXPLORAR DATOS
# ============================================================

# [Aquí cargan el CSV]
# df = pd.read_csv('...')
# print(df.head())
# print(df.describe())

# ============================================================
# PASO 2: PREPARAR DATOS PARA EL MODELO
# ============================================================

# [Aquí seleccionan X e Y]
# X = df[['variable_independiente']].values
# y = df['variable_dependiente'].values

# ============================================================
# PASO 3: CREAR Y ENTRENAR EL MODELO
# ============================================================

# [Aquí crean y ajustan el modelo]
# modelo = LinearRegression()
# modelo.fit(X, y)

# Extraer parámetros
# intercepto = modelo.intercept_
# pendiente = modelo.coef_[0]
# print(f"Ecuación: Y = {intercepto:.2f} + {pendiente:.4f}*X")

# ============================================================
# PASO 4: VISUALIZAR RESULTADOS
# ============================================================

# [Aquí crean gráfico con dispersión + recta]

# ============================================================
# PASO 5: REFLEXIÓN
# ============================================================

# [Aquí escriben respuestas a preguntas de análisis]
```

---

---

## SEMANA 8: MÉTRICAS DE ERROR Y CALIDAD DEL MODELO

### **Objetivo General de la Semana**

Que el estudiante calcule e interprete métricas de error (MSE, RMSE, MAE, R²) para evaluar la calidad de un modelo de regresión lineal en contextos de ingeniería de software, y compare modelos alternativos.

### **Objetivos Específicos**

1. Comprender qué es un residuo (error) y su importancia.
2. Calcular MSE, RMSE, MAE y R² usando Python y scikit-learn.
3. Interpretar cada métrica en el contexto de la ingeniería de software.
4. Comparar dos modelos alternativos (con diferentes variables o subconjuntos de datos) usando métricas.
5. Argumentar de forma crítica si el modelo es adecuado para tomar decisiones.

### **Competencias a Desarrollar**

- **Saber:** Definiciones de métricas, interpretación, ventajas/desventajas de cada una.
- **Hacer:** Cálculo de métricas en Python, comparación de modelos.
- **Ser:** Análisis crítico de limitaciones, justificación de decisiones.

---

### **CONTENIDOS CLAVE**

#### **1. Concepto de Residuos y Errores**

**Residuo:** diferencia entre el valor observado y el predicho por el modelo.

```
residuo_i = Y_observado_i - Y_predicho_i
```

**Visualización:**
```
En un gráfico de dispersión con recta:
- Los puntos son los datos observados.
- La recta es las predicciones.
- La distancia vertical entre punto y recta es el residuo.
```

**Código en Python:**
```python
# Calcular residuos
y_predicho = modelo.predict(X)
residuos = y - y_predicho

print(f"Primeros 5 residuos: {residuos[:5]}")
print(f"Residuo promedio: {np.mean(residuos):.4f}")
print(f"Desv. estándar de residuos: {np.std(residuos):.4f}")
```

#### **2. Métricas de Error - Definiciones y Fórmulas**

##### **2.1 MSE (Mean Squared Error)**

```
MSE = (1/n) * Σ(Y_observado_i - Y_predicho_i)²
```

**Interpretación:**
- Penaliza más los errores grandes (por el cuadrado).
- En unidades cuadráticas (ej: ms²).
- Rango: [0, ∞). Menor es mejor.

**Código:**
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y, y_predicho)
print(f"MSE: {mse:.4f}")
```

**Contexto de ingeniería de software:**
- Si modelamos tiempo de respuesta en ms, MSE está en ms².
- Un MSE de 100 significa que en promedio nos equivocamos en ±10 ms (aprox).

##### **2.2 RMSE (Root Mean Squared Error)**

```
RMSE = √MSE = √[(1/n) * Σ(Y_observado_i - Y_predicho_i)²]
```

**Interpretación:**
- Raíz cuadrada del MSE.
- **Ventaja:** en unidades originales del problema (ej: ms, no ms²).
- Rango: [0, ∞). Menor es mejor.
- **Desventaja:** sigue penalizando mucho errores grandes.

**Código:**
```python
rmse = np.sqrt(mse)
# O directamente:
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y, y_predicho))
print(f"RMSE: {rmse:.4f}")
```

**Contexto de ingeniería de software:**
- RMSE = 15 ms significa que el modelo se equivoca en promedio en ±15 ms.
- Comparar RMSE con la desviación estándar de Y: si RMSE es parecido a std(Y), el modelo no añade valor.

##### **2.3 MAE (Mean Absolute Error)**

```
MAE = (1/n) * Σ|Y_observado_i - Y_predicho_i|
```

**Interpretación:**
- Promedio de errores absolutos (sin cuadrar).
- En unidades originales del problema.
- Menos sensible a outliers que RMSE.
- Rango: [0, ∞). Menor es mejor.

**Código:**
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y, y_predicho)
print(f"MAE: {mae:.4f}")
```

**Contexto de ingeniería de software:**
- MAE = 12 ms significa error promedio de ±12 ms sin ponderar outliers.
- Útil si outliers no son tan relevantes para tu decisión.

##### **2.4 R² (Coeficiente de Determinación)**

```
R² = 1 - (SS_residual / SS_total)
SS_residual = Σ(Y_observado_i - Y_predicho_i)²
SS_total = Σ(Y_observado_i - Y_promedio)²
```

**Interpretación:**
- Proporción de varianza en Y explicada por el modelo.
- Rango: [0, 1] (también puede ser negativo si el modelo es muy malo).
- Rango: cuanto mayor, mejor.
- **R² = 0.8** significa que el modelo explica el 80% de la variabilidad en Y.

**Código:**
```python
from sklearn.metrics import r2_score

r2 = r2_score(y, y_predicho)
print(f"R²: {r2:.4f}")

# O:
r2_alt = modelo.score(X, y)
print(f"R² (desde modelo): {r2_alt:.4f}")
```

**Contexto de ingeniería de software:**
- R² = 0.65 en predicción de tiempos: el modelo captura el 65% de la variación. ¿Suficiente para tomar decisiones?
- R² = 0.95 en predicción de defectos: muy bueno, el modelo es confiable.

#### **3. Comparación de Métricas y Selección del Modelo**

**Tabla resumen:**

| Métrica | Unidades | Sensibilidad a Outliers | Rango Ideal | Usar Cuando |
|---------|----------|------------------------|-------------|-----------|
| **MSE** | Cuadráticas | Alta | [0, ∞) bajo | Comparar modelos (no interpretar valor absoluto) |
| **RMSE** | Originales | Alta | [0, ∞) bajo | Reportar error en unidades entendibles |
| **MAE** | Originales | Baja | [0, ∞) bajo | Hay muchos outliers; robustez importante |
| **R²** | Proporción (0–1) | Media | [0, 1] alto | Evaluar qué % de varianza explica el modelo |

**Decisión práctica en ingeniería de software:**
1. Calcular todas las métricas.
2. Si RMSE es muy grande comparado con `std(Y)`, el modelo es débil.
3. Si R² < 0.5, el modelo no es muy útil para predicciones.
4. Si hay outliers, observar MAE además de RMSE.

#### **4. Análisis de Residuos**

**Gráfico de residuos vs. predicciones:**
```python
plt.figure(figsize=(10, 6))
plt.scatter(y_predicho, residuos, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos')
plt.grid(True, alpha=0.3)
plt.show()

# Interpretación:
# - Si los residuos están distribuidos alrededor de 0: bien.
# - Si hay patrón (embudo, curvatura): supuestos violados.
```

**Histograma de residuos:**
```python
plt.figure(figsize=(10, 6))
plt.hist(residuos, bins=15, edgecolor='black', alpha=0.7)
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Distribución de Residuos')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)
plt.show()

# Interpretación:
# - Si es aproximadamente normal (campana): supuesto satisfecho.
# - Si hay sesgo o colas pesadas: revisar datos/modelo.
```

---

### **SECUENCIA DE ACTIVIDADES - SEMANA 8**

#### **Bloque 1: Apertura y Conexión con Semana 7 (15 min)**

**Actividad de Apertura - Preguntas Reflexivas:**

1. **Recordatorio (5 min):**
   - Proyectar el gráfico y modelo de la semana anterior.
   - *"La semana pasada ajustamos una recta a los datos. Pero, ¿qué tan buena es esa recta?"*

2. **Pregunta disparadora (5 min):**
   - *"Imagina que tu modelo predice tiempos de respuesta. Predice 100 ms para un caso y el valor real es 110 ms. ¿Error pequeño o grande? ¿Cómo saberlo?"*
   - *"¿Y si tenemos 20 predicciones? ¿Cómo resumir el desempeño en un número?"*

3. **Visión general (5 min):**
   - Explicar brevemente que hoy aprenderán 4 métricas para responder estas preguntas.

---

#### **Bloque 2: Exposición Teórica de Métricas (45 min)**

**Estructura:**

1. **Concepto de residuos (10 min):**
   - Dibujar en pizarra o diapositiva un gráfico de dispersión con recta.
   - Marcar un punto y dibujar la distancia vertical: "Esto es el residuo."
   - Mostrar ecuación: `residuo = Y_observado - Y_predicho`
   - Explicar: "Si el modelo predice bien, los residuos serán pequeños."

2. **Métricas MSE y RMSE (12 min):**
   - Escribir la fórmula de MSE en la pizarra (sin demostraciones, solo estructura).
   - Explicar: "Cuadramos los errores para penalizar mucho los grandes desviaciones."
   - Introducir RMSE como la raíz de MSE: "Para tener unidades originales."
   - Ejemplo numérico: "Si Y está en ms y RMSE = 20, significa error promedio de ±20 ms."
   - Ejemplo en ingeniería: "Un servidor donde RMSE = 50 ms en predicción de latencia es bastante bueno si el rango de latencias es 0–500 ms."

3. **Métrica MAE (10 min):**
   - Definición: "Promedio de errores sin elevar al cuadrado."
   - Ventaja vs. RMSE: "Si hay valores atípicos (outliers), MAE es más robusto."
   - Ejemplo: "Predicción de defectos con MAE = 2.3 significa error promedio de 2–3 defectos por módulo."

4. **Métrica R² (10 min):**
   - Concepto: "¿Qué porcentaje de la variabilidad en los datos explica el modelo?"
   - Rango 0–1: R² = 0.7 → "El modelo explica el 70% de la variación."
   - Ejemplo: "En predicción de esfuerzo, R² = 0.85 es muy bueno; R² = 0.4 es débil."
   - Interpretación práctica: "R² alto no garantiza predicciones exactas, pero sí que el modelo captura la tendencia."

3. **Tabla comparativa (3 min):**
   - Proyectar tabla de métricas (ver sección anterior).
   - Resaltar cuándo usar cada una.

---

#### **Bloque 3: Actividad Práctica Central (85 min)**

**Contexto:** Los estudiantes usan el modelo y datos de la semana 7, y ahora calculan métricas, analizan residuos y comparan con un modelo alternativo.

**Estructura propuesta:**

1. **Paso 1: Calcular Residuos y Métricas Básicas (20 min):**
   - Abrir el notebook de la semana 7.
   - Añadir código para calcular residuos:
     ```python
     y_predicho = modelo.predict(X)
     residuos = y - y_predicho
     ```
   - Calcular MSE, RMSE, MAE, R²:
     ```python
     from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
     
     mse = mean_squared_error(y, y_predicho)
     rmse = np.sqrt(mse)
     mae = mean_absolute_error(y, y_predicho)
     r2 = r2_score(y, y_predicho)
     
     print(f"MSE: {mse:.4f}")
     print(f"RMSE: {rmse:.4f}")
     print(f"MAE: {mae:.4f}")
     print(f"R²: {r2:.4f}")
     ```
   - **Checkpoint:** todos tienen métricas calculadas.

2. **Paso 2: Interpretación de Métricas (15 min):**
   - Añadir celda markdown con preguntas:
     - *"¿RMSE en unidades del problema es alto o bajo? Compara con `y.std()` o el rango de Y."*
     - *"¿Qué significa el R² que obtuviste? ¿El modelo explica mucha varianza?"*
     - *"¿MAE es similar a RMSE? ¿Hay evidencia de outliers?"*
   - Los estudiantes escriben 2–3 oraciones respondiendo en el notebook.
   - **Checkpoint:** interpretaciones presentes.

3. **Paso 3: Análisis de Residuos (20 min):**
   - Crear gráfico de residuos vs. predichos:
     ```python
     plt.figure(figsize=(10, 6))
     plt.scatter(y_predicho, residuos, alpha=0.6)
     plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
     plt.xlabel('Predicciones')
     plt.ylabel('Residuos')
     plt.title('Análisis de Residuos')
     plt.grid(True, alpha=0.3)
     plt.show()
     ```
   - Crear histograma de residuos:
     ```python
     plt.figure(figsize=(10, 6))
     plt.hist(residuos, bins=15, edgecolor='black', alpha=0.7)
     plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
     plt.xlabel('Residuos')
     plt.ylabel('Frecuencia')
     plt.title('Distribución de Residuos')
     plt.grid(True, alpha=0.3)
     plt.show()
     ```
   - Preguntas guiadas:
     - *"¿Los residuos están distribuidos alrededor de 0?"*
     - *"¿Hay algún patrón (embudo, curvatura) que sugiera violación de supuestos?"*
   - **Checkpoint:** gráficos visibles, observaciones anotadas.

4. **Paso 4: Modelo Alternativo - Comparación (20 min):**
   - Proponer modificación del modelo:
     - Opción A: Cambiar variable independiente (ej: si primera fue X1, ahora intentar X2).
     - Opción B: Filtrar outliers antes de entrenar (ej: eliminar registros donde Y > percentil 95).
     - Opción C: Usar subconjunto de datos (ej: solo primeros 80% de datos).
   - Entrenar modelo alternativo:
     ```python
     # Ejemplo: filtrar outliers
     Q1 = y.quantile(0.25)
     Q3 = y.quantile(0.75)
     IQR = Q3 - Q1
     mascara = (y >= Q1 - 1.5*IQR) & (y <= Q3 + 1.5*IQR)
     
     X_sin_outliers = X[mascara]
     y_sin_outliers = y[mascara]
     
     modelo_alt = LinearRegression()
     modelo_alt.fit(X_sin_outliers, y_sin_outliers)
     y_pred_alt = modelo_alt.predict(X_sin_outliers)
     
     # Métricas del modelo alternativo
     r2_alt = r2_score(y_sin_outliers, y_pred_alt)
     rmse_alt = np.sqrt(mean_squared_error(y_sin_outliers, y_pred_alt))
     ```
   - Crear tabla de comparación:
     ```python
     comparacion = pd.DataFrame({
         'Métrica': ['R²', 'RMSE', 'MAE'],
         'Modelo Original': [r2, rmse, mae],
         'Modelo Alternativo': [r2_alt, rmse_alt, mae_alt]
     })
     print(comparacion)
     ```
   - **Checkpoint:** tabla de comparación visible.

5. **Paso 5: Conclusiones y Recomendaciones (10 min):**
   - Añadir celda markdown con preguntas:
     - *"¿Cuál modelo es mejor según las métricas? ¿Por qué?"*
     - *"¿Qué limitaciones tiene el modelo elegido?"*
     - *"¿Lo usarías para tomar decisiones de software en tu empresa? ¿Con qué restricciones?"*
   - Los estudiantes escriben párrafo (5–7 oraciones) respondiendo.
   - **Checkpoint:** reflexión crítica documentada.

**Apoyo del docente durante la actividad:**
- Circulación constante.
- Mostrar errores comunes: olvido de `.fit()`, uso incorrecto de métricas.
- Proyectar ejemplos resueltos si hay dudas generales.
- Animar a comparar modelos de forma rigurosa.

---

#### **Bloque 4: Cierre y Síntesis (10 min)**

1. **Revisión colectiva (5 min):**
   - Pedir a 2–3 estudiantes que compartan resultados de comparación de modelos.
   - *"¿Cambió el R² significativamente al filtrar outliers?"*
   - *"¿Fue fácil decidir cuál modelo es mejor?"*

2. **Síntesis (3 min):**
   - Explicar que estas métricas serán esenciales en el Momento III (proyecto integrador).
   - *"Ahora saben cómo construir modelos Y evaluarlos. Esto es lo que hace un data scientist en ingeniería de software."*

3. **Transición (2 min):**
   - *"Próximas semanas: validación cruzada, visualización avanzada, y empezaremos el proyecto integrador."*

---

### **RECURSOS Y ARCHIVOS PARA SEMANA 8**

#### **Código Completo de Ejemplo (Semana 8)**

Los estudiantes pueden usar este template:

```python
# ============================================================
# SEMANA 8: MÉTRICAS DE ERROR Y CALIDAD DEL MODELO
# Curso: Herramientas Computacionales para Interpretación y Validación de Resultados
# Estudiante: [nombre]
# Fecha: [fecha]
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================
# CARGAR DATOS Y MODELO (del notebook de Semana 7)
# ============================================================

# [Aquí cargan datos y entrenan modelo como en Semana 7]

# ============================================================
# PASO 1: CALCULAR RESIDUOS Y MÉTRICAS
# ============================================================

y_predicho = modelo.predict(X)
residuos = y - y_predicho

# Métricas
mse = mean_squared_error(y, y_predicho)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_predicho)
r2 = r2_score(y, y_predicho)

print("=" * 50)
print("MÉTRICAS DEL MODELO ORIGINAL")
print("=" * 50)
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
print(f"\nDesv. estándar de Y: {y.std():.4f}")
print(f"Rango de Y: [{y.min():.2f}, {y.max():.2f}]")
print("=" * 50)

# ============================================================
# PASO 2: INTERPRETAR MÉTRICAS (en markdown)
# ============================================================

# [Escribir análisis en celda markdown]

# ============================================================
# PASO 3: ANÁLISIS DE RESIDUOS
# ============================================================

# Gráfico 1: Residuos vs. Predichos
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_predicho, residuos, alpha=0.6, color='steelblue')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos')
plt.grid(True, alpha=0.3)

# Gráfico 2: Distribución de Residuos
plt.subplot(1, 2, 2)
plt.hist(residuos, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Distribución de Residuos')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# PASO 4: MODELO ALTERNATIVO (Ejemplo: sin outliers)
# ============================================================

# Detectar y filtrar outliers (método IQR)
Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

mascara = (y >= limite_inferior) & (y <= limite_superior)
print(f"Registros totales: {len(y)}")
print(f"Registros después de filtro: {mascara.sum()}")
print(f"Outliers removidos: {(~mascara).sum()}")

# Datos sin outliers
X_filtrado = X[mascara]
y_filtrado = y[mascara]

# Entrenar modelo alternativo
modelo_alt = LinearRegression()
modelo_alt.fit(X_filtrado, y_filtrado)
y_pred_alt = modelo_alt.predict(X_filtrado)

# Métricas del modelo alternativo
mse_alt = mean_squared_error(y_filtrado, y_pred_alt)
rmse_alt = np.sqrt(mse_alt)
mae_alt = mean_absolute_error(y_filtrado, y_pred_alt)
r2_alt = r2_score(y_filtrado, y_pred_alt)

# ============================================================
# COMPARACIÓN DE MODELOS
# ============================================================

print("\n" + "=" * 50)
print("COMPARACIÓN: MODELO ORIGINAL vs. SIN OUTLIERS")
print("=" * 50)

comparacion = pd.DataFrame({
    'Métrica': ['R²', 'RMSE', 'MAE'],
    'Modelo Original': [f'{r2:.4f}', f'{rmse:.4f}', f'{mae:.4f}'],
    'Modelo Sin Outliers': [f'{r2_alt:.4f}', f'{rmse_alt:.4f}', f'{mae_alt:.4f}']
})

print(comparacion.to_string(index=False))

# Visualizar ambos modelos
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Modelo original
ax = axes[0]
ax.scatter(X, y, alpha=0.6, label='Datos', color='steelblue')
X_linea = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_linea = modelo.predict(X_linea)
ax.plot(X_linea, y_linea, 'r-', linewidth=2, label='Recta ajustada')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'Modelo Original (R²={r2:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Modelo sin outliers
ax = axes[1]
ax.scatter(X_filtrado, y_filtrado, alpha=0.6, label='Datos sin outliers', color='steelblue')
y_linea_alt = modelo_alt.predict(X_linea)
ax.plot(X_linea, y_linea_alt, 'r-', linewidth=2, label='Recta ajustada')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(f'Modelo Sin Outliers (R²={r2_alt:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# PASO 5: CONCLUSIONES (en markdown)
# ============================================================

# [Escribir reflexión y recomendaciones en celda markdown]
```

---

---

## EVIDENCIA DE APRENDIZAJE: INFORME PARCIAL DEL PROYECTO

### **Enunciado Completo para Estudiantes**

---

#### **ACTIVIDAD: Informe Parcial del Proyecto de Validación de Modelo en Ingeniería de Software**

**Semanas:** 7–8  
**Duración:** 2 semanas  
**Modalidad:** Individual o en parejas  
**Entrega:** Final de la semana 8  
**Formato:** Notebook de Google Colab (.ipynb)  

---

### **A. OBJETIVO GENERAL**

Construir un informe técnico inicial en formato de notebook ejecutable en Google Colab que demuestre la capacidad de:
1. Identificar un problema de ingeniería de software susceptible de modelado con regresión lineal.
2. Cargar, explorar y preparar datos reales o simulados.
3. Implementar un modelo de regresión lineal simple con justificación teórica.
4. Calcular e interpretar métricas de error (MSE, RMSE, MAE, R²).
5. Comunicar de forma clara y rigurosa conclusiones sobre la utilidad del modelo.

---

### **B. PRODUCTOS ESPERADOS**

El notebook debe contener las siguientes secciones:

#### **1. Portada (Celda de Texto Markdown)**

```markdown
# Informe Parcial: Modelado Predictivo en Ingeniería de Software

**Título del Proyecto:** [Título descriptivo, ej: "Predicción de Tiempo de Respuesta en Servidores Web"]

**Autor(es):** [Nombre(s) del(los) estudiante(s)]

**Código de Estudiante:** [Código(s)]

**Curso:** Herramientas Computacionales para Interpretación y Validación de Resultados

**Programa:** Ingeniería de Software

**Institución:** Universidad Cooperativa de Colombia

**Fecha:** [Fecha de entrega]

**Semestre:** [Semestre]

---

## Resumen Ejecutivo

[Párrafo de 3–4 líneas que resuma el problema, el modelo usado, resultados clave y conclusión principal. Ej: "Este informe presenta un modelo de regresión lineal que predice el tiempo de respuesta de una API REST basándose en el número de peticiones simultáneas. El modelo alcanzó un R² de 0.87, explicando el 87% de la variabilidad. Se concluye que el modelo es útil para estimaciones de capacidad pero requiere validación adicional con nuevos datos."]
```

#### **2. Definición del Problema (Celda de Texto Markdown)**

Describe en 2–3 párrafos:
- ¿Qué problema de ingeniería de software quieres resolver?
- ¿Por qué es importante?
- ¿Quién se beneficia de una predicción acertada?

**Ejemplo:**
```markdown
## 1. Definición del Problema

En una empresa que opera un servicio de microservicios en la nube, el equipo de DevOps necesita 
predecir el tiempo de respuesta de la API REST principal para dimensionar correctamente la 
infraestructura. Actualmente, los tiempos de respuesta fluctúan entre 50 ms y 300 ms según 
la carga del sistema.

**Pregunta de investigación:** ¿Es posible predecir el tiempo de respuesta promedio con base 
en el número de peticiones simultáneas?

**Importancia:** Una predicción acertada permite al equipo provisionar servidores de forma 
anticipada, evitando picos de latencia que afecten la experiencia de usuario.
```

#### **3. Dataset: Descripción y Origen (Celda de Texto Markdown)**

Describe:
- Nombre del dataset.
- Origen (¿de dónde proviene? ¿es simulado? ¿real?).
- Variables principales (nombre, tipo de dato, rango de valores).
- Número de registros.
- Posibles limitaciones o sesgos.

**Ejemplo:**
```markdown
## 2. Dataset y Descripción

**Nombre del Dataset:** `api_response_times.csv`

**Origen:** Simulación de logs de una API REST. Los datos fueron generados usando 
patrones observados en un servidor de producción real durante 30 días de operación.

**Variables:**

| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| `peticiones_simultaneas` | Numérico (int) | 5–500 | Número de peticiones HTTP simultáneas |
| `tiempo_respuesta_ms` | Numérico (float) | 45–320 | Latencia promedio en milisegundos |
| `cpu_usage` | Numérico (float) | 10–95 | Utilización de CPU en porcentaje |

**Número de registros:** 45 observaciones (una por día en 45 días de operación).

**Limitaciones:** Los datos son simulados, por lo que no capturan toda la complejidad de 
un sistema real (ej: picos de tráfico impredecibles, fallos de hardware, cambios de versión).
```

#### **4. Preparación de Datos (Código + Texto)**

**Código Python:**
```python
# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================
# PASO 1: CARGAR DATASET
# ============================================================

# Si está en Google Drive:
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/api_response_times.csv')

# Exploración inicial
print("Primeros registros:")
print(df.head(10))
print("\nForma del dataset:", df.shape)
print("\nTipos de datos:")
print(df.dtypes)
print("\nEstadísticas descriptivas:")
print(df.describe())
print("\nValores faltantes:")
print(df.isnull().sum())
```

**Texto explicativo (Markdown):**
```markdown
## 3. Preparación de Datos

### 3.1 Carga y Exploración

El dataset contiene 45 registros de observaciones diarias de una API REST. 
La exploración inicial reveló:

- No hay valores faltantes.
- Las variables están en rango esperado (sin anomalías evidentes).
- La distribución de `tiempo_respuesta_ms` tiene leve sesgo positivo (algunos picos altos).
```

#### **5. Construcción del Modelo de Regresión Lineal Simple (Código + Texto)**

**Código Python:**
```python
# ============================================================
# PASO 2: REGRESIÓN LINEAL SIMPLE
# ============================================================

# Seleccionar variable independiente (X) y dependiente (Y)
X = df[['peticiones_simultaneas']].values  # Matriz 2D
y = df['tiempo_respuesta_ms'].values       # Vector 1D

# Crear e instanciar el modelo
modelo = LinearRegression()

# Entrenar el modelo
modelo.fit(X, y)

# Extraer parámetros
intercepto = modelo.intercept_
pendiente = modelo.coef_[0]

# Hacer predicciones
y_predicho = modelo.predict(X)

# Mostrar ecuación
print("=" * 60)
print("MODELO DE REGRESIÓN LINEAL SIMPLE")
print("=" * 60)
print(f"Ecuación: Y = {intercepto:.2f} + {pendiente:.4f} * X")
print(f"\nInterpretación:")
print(f"  - Intercepto: {intercepto:.2f} ms (tiempo base sin carga)")
print(f"  - Pendiente: {pendiente:.4f} ms/petición")
print(f"    (Cada petición simultánea adicional aumenta latencia ~{pendiente:.4f} ms)")
```

**Texto explicativo (Markdown):**
```markdown
## 4. Construcción del Modelo de Regresión Lineal

### 4.1 Selección de Variables

**Variable Independiente (X):** `peticiones_simultaneas`  
**Variable Dependiente (Y):** `tiempo_respuesta_ms`

**Justificación:** En sistemas de software, el tiempo de respuesta típicamente aumenta 
con la carga del sistema. La relación es aproximadamente lineal en rangos operacionales 
normales, según la literatura de performance engineering.

### 4.2 Parámetros del Modelo

La ecuación del modelo ajustado es:

**Y = 42.15 + 0.5523 * X**

**Interpretación:**
- **Intercepto (42.15 ms):** Latencia base del sistema sin carga. Representa tiempo de 
  procesamiento mínimo.
- **Pendiente (0.5523 ms/petición):** Por cada petición simultánea adicional, la latencia 
  promedio aumenta aproximadamente 0.55 ms. Esto es razonable para un servidor moderno.
```

#### **6. Cálculo e Interpretación de Métricas de Error (Código + Texto)**

**Código Python:**
```python
# ============================================================
# PASO 3: MÉTRICAS DE ERROR
# ============================================================

# Calcular residuos
residuos = y - y_predicho

# Calcular métricas
mse = mean_squared_error(y, y_predicho)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_predicho)
r2 = r2_score(y, y_predicho)

# Mostrar resultados
print("=" * 60)
print("MÉTRICAS DE DESEMPEÑO DEL MODELO")
print("=" * 60)
print(f"MSE (Mean Squared Error):        {mse:.4f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.4f} ms")
print(f"MAE (Mean Absolute Error):       {mae:.4f} ms")
print(f"R² (Coeficiente de determinación): {r2:.4f}")
print("\n" + "=" * 60)
print("CONTEXTO ESTADÍSTICO")
print("=" * 60)
print(f"Desv. estándar de Y:  {y.std():.4f} ms")
print(f"Rango de Y:           [{y.min():.2f}, {y.max():.2f}] ms")
print(f"Media de Y:           {y.mean():.2f} ms")
```

**Texto explicativo (Markdown):**
```markdown
## 5. Métricas de Error y Desempeño

### 5.1 Resultados de Métricas

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **MSE** | 562.45 | Error cuadrado promedio |
| **RMSE** | 23.72 ms | Error promedio en unidades originales |
| **MAE** | 18.34 ms | Desviación promedio absoluta |
| **R²** | 0.823 | El modelo explica el 82.3% de la variabilidad |

### 5.2 Análisis e Interpretación

**RMSE = 23.72 ms:** En promedio, el modelo se equivoca en ±23.72 ms. Considerando que 
el rango de tiempos es ~45–320 ms (275 ms de amplitud), un error de ~24 ms representa 
aproximadamente el 8.7% del rango, lo cual es razonable.

**MAE = 18.34 ms:** Sin dar peso adicional a errores grandes, la desviación promedio es 
de 18.34 ms, ligeramente mejor que RMSE. Esto sugiere que hay algunos valores atípicos 
pero no extremos.

**R² = 0.823:** El modelo explica el 82.3% de la variación en tiempos de respuesta. Esto 
es una buena bondad de ajuste. El 17.7% restante se debe a factores no modelados (ej: 
variabilidad inherente del sistema, otras variables de rendimiento).

### 5.3 Conclusión sobre Calidad del Modelo

El modelo demuestra un **desempeño satisfactorio** para predicciones de latencia. Es 
útil para:
- Estimaciones aproximadas de capacidad.
- Identificar tendencias generales.
- Planificación de infraestructura a nivel macro.

**Limitaciones:**
- No es suficientemente preciso para SLAs estrictos (ej: garantizar < 100 ms).
- Variables omitidas (tipo de consulta, recursos disponibles) afectarían precisión.
```

#### **7. Visualizaciones (Código + Texto)**

**Código Python - Gráfico 1: Dispersión + Recta de Regresión**
```python
# ============================================================
# VISUALIZACIONES
# ============================================================

# Gráfico 1: Dispersión + Recta de Regresión
plt.figure(figsize=(10, 6))

# Datos observados
plt.scatter(X, y, alpha=0.6, s=60, label='Datos observados', color='steelblue')

# Recta de regresión
X_linea = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_linea = modelo.predict(X_linea)
plt.plot(X_linea, y_linea, 'r-', linewidth=2.5, label='Recta de regresión')

# Etiquetas y títulos
plt.xlabel('Peticiones Simultáneas', fontsize=12, fontweight='bold')
plt.ylabel('Tiempo de Respuesta (ms)', fontsize=12, fontweight='bold')
plt.title('Regresión Lineal Simple: Latencia vs. Carga', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)

# Añadir ecuación en gráfico
ecuacion = f'Y = {intercepto:.2f} + {pendiente:.4f}*X\nR² = {r2:.4f}'
plt.text(0.98, 0.05, ecuacion, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()
```

**Código Python - Gráfico 2: Análisis de Residuos**
```python
# Gráfico 2: Análisis de Residuos
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuos vs. Predichos
ax = axes[0]
ax.scatter(y_predicho, residuos, alpha=0.6, s=60, color='steelblue')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Valores Predichos (ms)', fontsize=11)
ax.set_ylabel('Residuos (ms)', fontsize=11)
ax.set_title('Residuos vs. Valores Predichos', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Distribución de Residuos
ax = axes[1]
ax.hist(residuos, bins=12, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Residuos (ms)', fontsize=11)
ax.set_ylabel('Frecuencia', fontsize=11)
ax.set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"Residuo promedio: {residuos.mean():.6f} (debe ser ~0)")
print(f"Desv. estándar de residuos: {residuos.std():.4f}")
```

**Texto explicativo (Markdown):**
```markdown
## 6. Visualizaciones e Interpretaciones

### 6.1 Gráfico 1: Ajuste del Modelo

[Se espera que el estudiante incluya el gráfico generado]

**Interpretación:** La recta de regresión captura bien la tendencia general de los datos. 
Los puntos están distribuidos alrededor de la recta, con algunos alejamientos que pueden 
deberse a variabilidad inherente del sistema.

### 6.2 Gráfico 2: Análisis de Residuos

[Se espera que el estudiante incluya ambos gráficos de residuos]

**Interpretación del gráfico de residuos vs. predichos:**
Los residuos están aproximadamente distribuidos alrededor de 0, sin patrón claro de 
sesgo. Esto sugiere que la relación lineal es apropiada. Hay 1–2 outliers moderados 
pero no extremos.

**Interpretación del histograma de residuos:**
La distribución es aproximadamente normal (campana), lo que satisface el supuesto de 
normalidad de errores. El pico está cerca de 0, indicando buen ajuste central.
```

#### **8. Conclusiones y Perspectivas Futuras (Celda de Texto Markdown)**

```markdown
## 7. Conclusiones y Recomendaciones

### 7.1 Hallazgos Principales

1. **Modelo válido:** La regresión lineal captura adecuadamente la relación entre 
   peticiones simultáneas y tiempo de respuesta (R² = 0.823).

2. **Parámetros interpretables:** La pendiente de 0.5523 ms/petición es consistente 
   con capacidad conocida del servidor.

3. **Precisión aceptable:** RMSE de 23.72 ms es lo suficientemente bajo (8.7% del rango) 
   para decisiones de planificación de capacidad.

### 7.2 Limitaciones del Modelo Actual

1. **Variables omitidas:** El modelo no incluye tipo de consulta, índices de caché, 
   o estado de la infraestructura.

2. **Extrapolación:** Predicciones fuera del rango observado (< 5 o > 500 peticiones) 
   pueden ser imprecisas.

3. **Datos simulados:** El dataset es simulado, no captura anomalías reales del sistema.

### 7.3 Próximos Pasos para Mejora

1. **Validación cruzada:** Implementar k-fold cross-validation para evaluar estabilidad 
   del modelo con datos nuevos.

2. **Variables adicionales:** Incorporar otras métricas (CPU, memoria, tipo de consulta).

3. **Modelos alternativos:** Explorar regresión polinomial o modelos no lineales.

4. **Datos reales:** Recopilar datos de producción real para validar predicciones.

### 7.4 Recomendación Final

**¿Es útil este modelo para tomar decisiones?** Sí, pero con reservas. Es apropiado 
para estimaciones gruesas de capacidad y tendencias. Para SLAs estrictos o decisiones 
críticas, se recomienda:
- Validar con datos reales de producción.
- Incluir más variables explicativas.
- Usar técnicas de validación más robustas (cross-validation).

Este modelo es un buen punto de partida que será refinado en el Momento III del curso 
con técnicas avanzadas.
```

---

### **C. CRITERIOS DE EVALUACIÓN**

Se evalúa el informe con la siguiente rúbrica, alineada con Saber–Ser–Hacer:

---

#### **RÚBRICA DE EVALUACIÓN**

**Escala:** 4 niveles (Insuficiente, Básico, Proficiente, Avanzado)

**Puntuación total:** 100 puntos

##### **DIMENSIÓN 1: SABER (Dominio Conceptual) – 30 puntos**

| Indicador | Insuficiente (0–7 pts) | Básico (8–15 pts) | Proficiente (16–23 pts) | Avanzado (24–30 pts) |
|-----------|------------------------|-------------------|------------------------|----------------------|
| **Identificación de variables** | No identifica X e Y claramente. | Identifica X e Y pero sin justificación. | Identifica correctamente X e Y con breve justificación. | Identifica X e Y con justificación sólida y contexto de ingeniería. |
| **Interpretación de métricas** | No calcula métricas o valores incorrectos. | Calcula métricas pero interpretación confusa. | Calcula correctamente e interpreta en contexto del problema. | Interpreta profundamente, compara con estadísticas de Y, discute implicaciones. |
| **Comprensión de supuestos** | No menciona supuestos de regresión. | Menciona brevemente supuestos. | Explica supuestos y revisa algunos (ej: normalidad de residuos). | Analiza supuestos en detalle y comenta implicaciones de incumplimientos. |

---

##### **DIMENSIÓN 2: HACER (Implementación Técnica) – 40 puntos**

| Indicador | Insuficiente (0–10 pts) | Básico (11–20 pts) | Proficiente (21–30 pts) | Avanzado (31–40 pts) |
|-----------|------------------------|-------------------|------------------------|----------------------|
| **Implementación del modelo** | Código no ejecuta o errores críticos. | Código ejecuta pero con problemas de lógica. | Modelo entrena correctamente, parámetros adecuados. | Modelo implementado correctamente, documentación de pasos, código limpio. |
| **Cálculo de métricas** | Métricas no calculadas o cálculos incorrectos. | Calcula métricas con errores menores. | Calcula correctamente MSE, RMSE, MAE, R². | Calcula métricas correctamente, usa librerías apropiadas, incluye verificaciones. |
| **Visualizaciones** | Sin gráficos o gráficos incorrectos. | Gráficos presentes pero incompletos (faltan etiquetas, leyendas). | Gráficos claros, etiquetados, apropiados al análisis. | Gráficos profesionales, bien etiquetados, incluyen ecuación/métricas, múltiples perspectivas. |
| **Notebook ejecutable** | Notebook con errores que impiden ejecución. | Notebook ejecutable con pequeñas correcciones. | Notebook ejecutable sin errores, celdas bien organizadas. | Notebook impecable, bien organizado con secciones claras, comentarios, sin errores. |

---

##### **DIMENSIÓN 3: SER (Comunicación, Rigor y Reflexión Crítica) – 30 puntos**

| Indicador | Insuficiente (0–7 pts) | Básico (8–15 pts) | Proficiente (16–23 pts) | Avanzado (24–30 pts) |
|-----------|------------------------|-------------------|------------------------|----------------------|
| **Claridad de redacción** | Redacción confusa, poco profesional, muchas faltas. | Redacción clara pero con algunos errores o imprecisiones. | Redacción clara, académica, sin faltas significativas. | Redacción impecable, profesional, fluida, bien estructurada. |
| **Justificación de decisiones** | No justifica selecciones (modelo, variables). | Justificación superficial. | Justifica adecuadamente por qué se eligió cada enfoque. | Justificación profunda, considera alternativas, explica trade-offs. |
| **Análisis crítico** | No analiza limitaciones o limitaciones obvias. | Menciona limitaciones superficialmente. | Identifica limitaciones claras del modelo y datos. | Análisis crítico profundo: limitaciones, sesgos, suposiciones, impacto en aplicabilidad. |
| **Conclusiones y recomendaciones** | Conclusiones débiles o ausentes. | Conclusiones presentes pero genéricas. | Conclusiones claras, contextualizadas al problema. | Conclusiones insightful, propone mejoras concretas, reflexiona sobre aplicabilidad real. |

---

#### **Desglose de Puntuación**

```
TOTAL = Saber (30) + Hacer (40) + Ser (30) = 100 puntos

Conversión a escala institucional (0–5.0 si es escala decimal):
- 90–100 pts → 5.0 (Excelente)
- 80–89 pts → 4.5 (Muy Bueno)
- 70–79 pts → 4.0 (Bueno)
- 60–69 pts → 3.5 (Satisfactorio)
- 50–59 pts → 3.0 (Aceptable)
- < 50 pts → < 3.0 (Insuficiente)
```

---

### **D. PASOS PARA ENTREGA**

1. **Crear notebook en Google Colab** con nombre: `Informe_Parcial_[Apellido_Nombre]_Semanas_7_8.ipynb`

2. **Completar todas las secciones** indicadas arriba (portada, problema, dataset, código, visualizaciones, conclusiones).

3. **Ejecutar todo el notebook** de principio a fin para verificar que no hay errores.

4. **Guardar el notebook** en Google Drive o compartir link de acceso (ver+comentar al docente).

5. **Guardar también como PDF** (Archivo → Descargar → PDF) y cargar en el aula virtual si aplica.

6. **Fecha de entrega:** Final de semana 8 (viernes de esa semana, antes de las 23:59).

---

### **E. RECOMENDACIONES PARA ESTUDIANTES**

1. **Mantener notebook limpio:** Elimina celdas de prueba, deja código organizado.

2. **Documentación:** Usa markdown para explicar cada paso. El código debe ser autosuficiente (con comentarios).

3. **Gráficos profesionales:** Incluye títulos, etiquetas en ejes, leyendas, unidades. Usa colores apropiados.

4. **Evidencia de iteración:** Si intentaste modelos alternativos, muestra. Esto demuestra aprendizaje.

5. **Dataset:** Si usas datos simulados, documenta cómo los generaste. Si es dataset público, cite fuente.

6. **Lenguaje técnico:** Usa términos de ingeniería de software cuando sea posible (latencia, throughput, defects, etc.).

---

### **F. RÚBRICA RESUMIDA PARA IMPRESIÓN / DISTRIBUCIÓN A ESTUDIANTES**

**Para que los estudiantes sepan qué se evalúa:**

```
INFORME PARCIAL: RÚBRICA SIMPLIFICADA

SABER (30 puntos)
  ☐ Identificas variables X e Y con justificación (5 pts)
  ☐ Calculas e interpretas métricas (MSE, RMSE, MAE, R²) (15 pts)
  ☐ Explicas supuestos de regresión y los verificas (10 pts)

HACER (40 puntos)
  ☐ Código implementa modelo de regresión correctamente (15 pts)
  ☐ Calculas métricas con librerías adecuadas, sin errores (10 pts)
  ☐ Generas visualizaciones claras, etiquetadas, profesionales (10 pts)
  ☐ Notebook ejecutable, bien organizado, sin errores críticos (5 pts)

SER (30 puntos)
  ☐ Redacción clara, académica, sin faltas (8 pts)
  ☐ Justificas decisiones: por qué ese modelo, esa variable (8 pts)
  ☐ Analizas críticamente limitaciones y alcances del modelo (7 pts)
  ☐ Conclusiones insightful, consideras mejoras futuras (7 pts)

TOTAL: 100 puntos
```

---

**FIN DEL DOCUMENTO DE GUION Y EVIDENCIA**

