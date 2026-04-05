# Pipeline de predicción de rotación de empleados

**Autor:** Jeshua Romero Guadarrama

## Descripción

Proyecto de machine learning que construye un pipeline completo de clasificación para predecir si un empleado abandonará la empresa, utilizando el dataset **IBM HR Analytics Employee Attrition & Performance**.

El pipeline integra limpieza de datos, ingeniería de características, preprocesamiento diferenciado para variables numéricas y categóricas, selección de modelos mediante `GridSearchCV` con validación cruzada estratificada, y exportación del modelo final a un archivo `.pkl`.

## Dataset

| Característica | Detalle |
|----------------|---------|
| **Fuente** | [IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Repositorio oficial de IBM en GitHub) |
| **Observaciones** | 1,470 empleados |
| **Variables** | 35 (demográficas, laborales y de satisfacción) |
| **Variable objetivo** | `Attrition` (Yes/No). Desbalanceada (83.9% No, 16.1% Sí) |

## Contenido del repositorio

| Archivo | Descripción |
|---------|-------------|
| `empleados_rotacion.csv` | Conjunto de datos utilizado |
| `pipeline_prediccion_rotacion.ipynb` | Notebook completo con todo el flujo |
| `pipeline_rotacion_empleados.pkl` | Pipeline exportado (preprocesamiento + modelo) |

## Metodología

1. **Análisis exploratorio:** Distribuciones, boxplots por clase, tasas de rotación por departamento y horas extra, matriz de correlación.
2. **Limpieza:** Eliminación de columnas sin varianza (`EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`).
3. **Ingeniería de características:** 4 variables derivadas (ingreso por año de experiencia, ratio de permanencia, ratio sin promoción, satisfacción promedio).
4. **Preprocesamiento:** `ColumnTransformer` con pipelines diferenciados:
   - Numéricas: imputación por mediana + `StandardScaler`
   - Categóricas: imputación por moda + `OneHotEncoder`
5. **Selección de modelos:** `GridSearchCV` con validación cruzada estratificada (5 pliegues) sobre 4 algoritmos: Regresión Logística, Bosque Aleatorio (Random Forest), Gradient Boosting y SVM (Support Vector Machine).
6. **Evaluación:** Accuracy, Precision, Recall, F1 Score, AUC ROC, matriz de confusión y curva ROC.
7. **Exportación:** Pipeline completo a `.pkl` con verificación de integridad.

