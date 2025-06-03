# Startups-ML 

Modelo de Machine Learning que predice el **éxito de startups** (éxito = adquirida, fracaso = cerrada) a partir de sus características iniciales y su historial de financiación. 

---
## Tabla de contenidos
1. [Visión general](#visión-general)
2. [Dataset](#dataset)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Instalación](#instalación)
5. [Uso rápido](#uso-rápido)
   1. [Preprocesamiento](#preprocesamiento)
   2. [Entrenamiento y evaluación](#entrenamiento-y-evaluación)
   3. [API FastAPI](#api-fastapi)
6. [Metodología](#metodología)
7. [Resultados](#resultados)
8. [Buenas prácticas](#buenas-prácticas)
9. [Contribución](#contribución)
10. [Licencia](#licencia)

---
## Visión general
El objetivo es determinar, con datos disponibles en los primeros meses de vida de la compañía, si una startup acabará **adquirida** o **cerrada**. Se siguen todas las fases del ciclo de vida de un proyecto de ML:

1. Ingesta y limpieza de datos.
2. EDA e ingeniería de características.
3. Entrenamiento de varios modelos (baseline, tuning, explicación con SHAP).
4. Evaluación rigurosa (train/test + CV) y diagnóstico de sesgo-varianza.
5. Exportación del modelo y despliegue vía **FastAPI**.

---
## Dataset
| Fuente | Registros | Clases | Última actualización |
|--------|-----------|--------|----------------------|
| Crunchbase (copia anonimizada) | ≈ +900 startups | `success` / `failure` | 2013-06-01 |

El archivo original vive en `data/raw/startup_data.csv`. El **dataset procesado** principal se guarda como `data/processed/startup_data_processed.csv`.

---
## Estructura del proyecto
```text
startups-ml/
│
├── data/                  # Datos crudos, intermedios y procesados
├── notebooks/             # Jupyter notebooks (EDA, modelado, explicación)
├── src/                   # Código fuente del paquete
│   ├── config.py          # Rutas y constantes globales
│   ├── data/              # Funciones de carga y limpieza
│   ├── features/          # Ingeniería de características
│   ├── models/            # Entrenamiento, evaluación, explicación
│   └── visualization/     # Funciones de plotting reutilizables
├── models/                # Modelos entrenados (.pkl)
├── reports/               # Figuras y resultados finales
├── tests/                 # Pytest unit tests
├── app.py                 # API FastAPI
├── requirements.txt       # Dependencias en producción
└── README.md              # ¡Estás aquí!
```

---
## Instalación
```bash
# Clona el repo
$ git clone https://github.com/<user>/startups-ml.git
$ cd startups-ml

# Crea y activa entorno
$ python -m venv venv
$ source venv/bin/activate  # Windows: venv\Scripts\activate

# Instala deps extra de desarrollo (opcional)
$ pip install -r requirements-dev.txt
```

---
## Uso rápido
### Preprocesamiento
```bash
python -m src.data.preprocess   # Genera data/processed/*.csv
```

### Entrenamiento y evaluación
```python
from src.models.train import train_models
from src.data.load_data import load_and_split_data

X_train, X_test, y_train, y_test = load_and_split_data()
model_dict = train_models(X_train, y_train)
```
Los resultados (métricas, figuras, modelos `.pkl`) se almacenan automáticamente en `reports/` y `models/`.

### API FastAPI
```bash
uvicorn app:app --reload
```
Accede a `http://localhost:8000/docs` para la documentación interactiva.

---
## Metodología
| Paso | Detalle |
|------|---------|
| Split | `train_test_split` estratificado 80/20 + Cross-Validation 5-fold |
| Modelos | Árbol de decisión, Random Forest, Gradient Boosting, XGBoost |
| Métricas | **Accuracy**, *precision*, *recall*, *F1* + matriz de confusión |
| Explicabilidad | SHAP values (plots guardados en `/reports/figures/shap/`) |
| Diagnóstico | Curvas de aprendizaje, validación y complejidad |

---
## Resultados
| Modelo final | Accuracy test | F1 test |
|--------------|---------------|---------|
| XGBoost | **0.97** | 0.96 |

Figuras clave se encuentran en `reports/figures/model_performance/` y `reports/figures/shap/`.

---
## Buenas prácticas
- Rutas centralizadas en `src.config` usando `pathlib`.
- Notebooks con título, descripción y ejecución limpia.
- Datos originales **inmutables** (`data/raw/`), toda transformación genera archivos nuevos.
- Serialización con `joblib` y pipelines de Scikit-learn.
- Tests (Pytest) >90 % de cobertura.
- Formateo automático `black` + `isort`.

---
## Licencia
Este proyecto está licenciado bajo los términos de la licencia **MIT**. Consulta el archivo [`LICENSE`](LICENSE) para más información.
