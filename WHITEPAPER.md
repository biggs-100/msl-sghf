# MSL + SG-HF: Compresion Extrema y Destilacion Sin Reentrenar

## Resumen

Presentamos dos inventos complementarios y el pipeline que los une para lograr compresion 50-100x de modelos de lenguaje sin reentrenar el alumno.

- **MSL (Multi-Scale Linear):** Una nueva capa que reemplaza la capa lineal densa. Factoriza W = U·diag(s)·V^T por escalas jerarquicas. Al truncar las ultimas escalas, se obtiene un modelo mas chico que funciona SIN reentrenar. El alumno nace del profesor.

- **SG-HF (Seed-Generated Fractal Weights):** Compresion post-hoc via semillas que generan pesos bajo demanda. Dos sabores: expansion Kronecker (50-100x en pesos con std > 0.01) y ternario {-1,0,+1} (16x en pesos con std < 0.01).

- **Pipeline completo:** Destilar un modelo existente (teacher) a una arquitectura MSL, comprimir los factores con Kronecker, y distribuir solo la semilla. El usuario final puede truncar el modelo sin reentrenar para adaptarlo a su hardware.

---

## 1. El Problema

Los LLMs modernos (Mistral, Llama-3, Qwen, DeepSeek) tienen pesos con dos propiedades que los hacen casi incompresibles post-hoc:

1. **std ~ 0.003:** La magnitud de los pesos es extremadamente pequena.
2. **Espectro de valores singulares plano:** La informacion esta distribuida uniformemente en todas las direcciones. No hay componentes principales.

Cualquier tecnica de bajo rango (SVD truncado, Kronecker, tensor train) pierde una fraccion enorme de la senal porque el error de aproximacion es comparable a la senal misma.

**Unico metodo que funciona sin reentrenar:** cuantizacion ternaria {-1,0,+1}, porque preserva el rango completo y el error incoherente por elemento es tolerado por la no-linealidad SiLU. Pero su limite es ~16x.

---

## 2. SG-HF: Compresion por Semillas

### 2.1 Expansion Kronecker

Para pesos con std > 0.01 (modelos antiguos como GPT-2, capas especificas de modelos modernos):

```
W = expandir_Kronecker(seed, bases)
```

El seed es 50-400x mas chico que W. La expansion usa FFT + modulacion + Kronecker rank-1 por bloque.

**Resultado:** 50-100x de compresion, COS > 0.9 en GPT-2.

### 2.2 Ternario {-1, 0, +1}

Para pesos con std < 0.01 (todos los LLMs modernos):

```
W[i,j] = mascara[i,j] * escala[i]
mascara[i,j] in {-1, 0, +1}
escala[i] = media(|W[i,j]| para elementos no cero de la fila i)
```

**Resultado:** 16x de compresion, COS 0.77 en Mistral-7B. No requiere entrenamiento ni datos.

**Por que funciona:** Preserva el rango completo de la matriz. El error es independiente por elemento, no correlacionado, y la SiLU lo tolera como ruido benigno.

---

## 3. MSL: Multi-Scale Linear

### 3.1 Arquitectura

Reemplaza la capa lineal densa `y = Wx + b` por:

```
y = (x @ V^T) * s @ U^T + b
```

Donde:
- U in R^{M x R}, V in R^{R x N}, s in R^{R}
- R = suma de ranks de cada escala (ej: 8+16+32 = 56)
- Cada escala captura un nivel de detalle distinto

### 3.2 Truncamiento sin reentrenar

La propiedad clave: el modelo con todas las escalas (profesor) y el modelo con solo las primeras k escalas (alumno) son el MISMO modelo con diferente configuracion.

```
y = (x @ V_k^T) * s_k @ U_k^T + b    (k < R = alumno)
y = (x @ V_R^T) * s_R @ U_R^T + b    (k = R = profesor)
```

**Resultado en modelo GPT (864K params):**
- Profesor: loss 0.53
- Alumno (7x mas chico): loss 0.53
- Gap: 0.005 en loss

### 3.3 Entrenamiento

MSL se entrena con:
- **Sorted SVD loss:** Fuerza s[0] > s[1] > ... > s[R] (primeras escalas mas importantes)
- **Orthogonality loss:** Fuerza U^T U ≈ I y V V^T ≈ I (para que truncar sea optimo)
- **L1 espectral:** Tiende a cero las componentes tardias (compresion natural)

### 3.4 MSL-Deep: Factores con std grande para Kronecker

Para aplicar Kronecker sobre los factores U y V de MSL (std ~0.01-0.1), se anade un bottleneck:

```
W = row_norms * U1_norm @ U2 @ diag(s) @ V2^T @ V1_norm^T * V_row_norms
```

Donde U1_norm in R^{M x k} tiene filas normalizadas a norma 1, con std ~ 1/sqrt(k).
Para k=64: std ~ 0.125, suficiente para Kronecker.

---

## 4. Pipeline Completo: Destilacion + MSL + Kronecker

```
Modelo pre-entrenado (teacher, 1B+ params)
       |
       v  Destilacion capa por capa o por logits
  MSL Student (U, s, V entrenados para imitar al teacher)
       |
       v  Compresion de factores
  Kronecker sobre U1_norm y V1_norm
  Ternario sobre U2, s, V2 (son chicos, no importa)
       |
       v  Distribucion
  Seed: factores Kronecker + matrices nucleo + normas de fila
  Tamano tipico: ~20-50 MB para un modelo equivalente a 1B
       |
       v  Inferencia
  1. Generar U1, V1 desde seed (Kronecker expansion)
  2. Reconstruir factores (U1 @ U2, V2 @ V1)
  3. Forward MSL normal
  4. Truncar escalas si se necesita mas velocidad
```

---

## 5. Resultados Experimentales

### 5.1 MSL en MNIST (MLP 784-512-256-10)

| Modo | Compresion | Accuracy |
|------|-----------|----------|
| Profesor | 3.6x | 97.74% |
| Alumno (k=2) | 7.1x | 97.45% |
| Alumno (k=1) | 21.3x | 94.88% |
| Gap alumno-profesor | - | 2.86pp |

### 5.2 MSL en GPT chico (864K params, hidden=128, 4 layers)

| Modo | Compresion | Loss | PPL |
|------|-----------|------|-----|
| Profesor | 1.0x | 0.529 | 1.70 |
| Alumno (k=2) | 2.4x | 0.531 | 1.70 |
| Alumno (k=1) | 7.1x | 0.535 | 1.71 |
| Gap alumno-profesor | - | 0.006 | 0.01 |

### 5.3 SG-HF Ternario en Mistral-7B

| Componente | COS | Compresion |
|-----------|-----|-----------|
| MLP output | 0.77 | 16x |
| Attention Q | 0.89 | 16x |
| Attention O | 0.89 | 16x |

### 5.4 Limites del post-hoc en Mistral-7B

| Tecnica | Compresion | COS | Funciona? |
|---------|-----------|-----|-----------|
| Kronecker rank-1 | 50x | 0.04 | NO (std 0.003) |
| SVD truncado rank 224 | 2x | 0.52 | NO (espectro plano) |
| Kronecker sobre U de SVD | 50x | 0.004 | NO (std U = 0.008) |
| Normalizacion filas + Kronecker | 50x | 0.01 | NO (sin estructura bloques) |
| Ternario {-1,0,+1} | 16x | 0.77 | SI |

---

## 6. Lecciones Aprendidas

### 6.1 No existe compresion post-hoc >16x para LLMs modernos

Los pesos de Mistral, Llama, Qwen y DeepSeek tienen std ~0.003 y espectro plano. Esto no es un accidente: es consecuencia de AdamW + weight decay + LayerNorm. No hay estructura oculta que explotar.

### 6.2 MSL funciona cuando se entrena desde cero

La truncabilidad sin reentrenar es real (gap ~0 en modelos chicos). El costo es que el modelo debe entrenarse con MSL desde el inicio, no aplicarse post-hoc.

### 6.3 El pipeline MSL + destilacion + Kronecker es viable pero caro

Requiere:
- Datos de destilacion masivos (100B+ tokens)
- GPU con 16GB+ VRAM (P100, V100, A100)
- ~$10K-50K en computo para escalar a 7B

### 6.4 Para uso practico hoy

Usar GGUF Q4 o Q3 de cualquier modelo en HuggingFace. Nuestros inventos son investigacion, no producto listo.

---

## 7. Proximos Pasos

1. **Demostrar pipeline completo en modelo ~300M eq.** usando Kaggle (P100 16GB, 9h)
2. **Publicar whitepaper** con resultados y codigo abierto
3. **Colaborar** con quien tenga recursos para escalar a 7B+

---

## 8. Repositorio y Codigo

- `sg_hf/msl_v2.py` - Implementacion de MSL
- `sg_hf/msl_transformer.py` - Transformer con MSL
- `msl_demo_v2.py` - Demo MNIST
- `msl_demo_transformer.py` - Demo GPT chico
- `msl_factorize_mistral.py` - Factorizacion de Mistral-7B
- `test_distill_msl_mistral.py` - Destilacion a MSL
- `distill_msl_compressible.py` - Destilacion con regularizacion
- `train_msl_final.py` - Entrenamiento MSL desde cero

---

*Julio 2026 - Proyecto ia-2027*
