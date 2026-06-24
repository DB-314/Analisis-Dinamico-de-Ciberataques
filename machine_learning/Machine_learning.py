# -*- coding: utf-8 -*-
"""SISTEMA DE DETECCIÓN DE CIBERATAQUES.ipynb
"""

# ==========================================
# FASE 1 — Instalación de librerías
# ==========================================

!pip install -q xgboost lightgbm imbalanced-learn seaborn
!pip install -q catboost

print("Librerías instaladas correctamente")

# ==========================================
# FASE 2 — Importación de librerías
# ==========================================

import pandas as pd
import numpy as np
import os
import time

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from sklearn.base import clone

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

print("Librerías importadas correctamente")

# ==========================================
# FASE 3 — Montar Google Drive
# ==========================================

from google.colab import drive
drive.mount('/content/drive')

print("Google Drive montado correctamente")

# ==========================================
# FASE 4 — Definir rutas de datasets
# ==========================================

base_path = "/content/drive/MyDrive/UWF-ZeekData24/"

datasets = {
    "Benign": "Benign.csv",
    "Credential_Access": "Credential_Access.csv",
    "Defense_Evasion": "Defense_Evasion.csv",
    "Exfiltration": "Exfiltration.csv",
    "Initial_Access": "Initial_Access.csv",
    "Persistence": "Persistence.csv",
    "Privilege_Escalation": "Privilege_Escalation.csv",
    "Reconnaissance": "Reconnaissance.csv"
}

print("Rutas definidas correctamente")

# ==========================================
# FASE 5 — Cargar datasets
# ==========================================

dfs = []

for label, file in datasets.items():
    path = os.path.join(base_path, file)
    df = pd.read_csv(path)
    df["attack_type"] = label
    dfs.append(df)
    print(f"  ✓ {label}: {len(df)} registros")

data = pd.concat(dfs, ignore_index=True)

print(f"\n✅ Total de registros: {len(data)}")
print(f"✅ Total de columnas: {len(data.columns)}")

# ==========================================
# FASE 6 — Mezcla aleatoria del dataset
# ==========================================

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset mezclado correctamente")
print(f"Shape: {data.shape}")
data.head()

# ==========================================
# FASE 7 — Información general del dataset
# ==========================================

print("=== INFORMACIÓN GENERAL ===")
data.info()

print("\n=== DISTRIBUCIÓN DE CLASES ===")
class_distribution = data["attack_type"].value_counts()
print(class_distribution)

# Visualización de la distribución de clases
plt.figure(figsize=(10, 6))
class_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribución de Clases Original')
plt.xlabel('Tipo de Ataque')
plt.ylabel('Número de Muestras')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ==========================================
# FASE 8 — Revisión de valores nulos
# ==========================================

null_counts = data.isnull().sum().sort_values(ascending=False)
print("=== VALORES NULOS POR COLUMNA ===")
print(null_counts[null_counts > 0])

# Porcentaje de nulos
null_percentages = (null_counts / len(data) * 100).sort_values(ascending=False)
print("\n=== PORCENTAJE DE VALORES NULOS ===")
print(null_percentages[null_percentages > 0])

# ==========================================
# FASE 9 — Eliminación de columnas innecesarias
# ==========================================

# Copia de seguridad para exportación posterior
data_original = data.copy()

columns_to_drop = [
    "label_tactic",
    "label_technique",
    "label_binary",
    "label_cve",
    "uid",
    "datetime",
    "ts"
]

data.drop(columns=columns_to_drop, inplace=True, errors="ignore")

print("Columnas eliminadas correctamente")
print(f"Columnas restantes: {data.columns.tolist()}")

# ==========================================
# FASE 10 — Identificación de columnas categóricas
# ==========================================

categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()

# Excluir la columna objetivo
if "attack_type" in categorical_cols:
    categorical_cols.remove("attack_type")

print(f"Columnas categóricas: {categorical_cols}")

# ==========================================
# FASE 11 — Codificación de variables categóricas
# ==========================================

# Guardar los encoders para posible uso futuro
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = data[col].astype(str)
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
    print(f"  ✓ {col} codificada")

print("Variables categóricas codificadas correctamente")

# ==========================================
# FASE 12 — Manejo de valores nulos
# ==========================================

# Identificar columnas numéricas con nulos
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if "attack_type" in numeric_cols:
    numeric_cols.remove("attack_type")

# Imputación con mediana (robusto para outliers)
imputer = SimpleImputer(strategy="median")

X_temp = data[numeric_cols].copy()
X_temp_imputed = pd.DataFrame(
    imputer.fit_transform(X_temp),
    columns=X_temp.columns
)

# Reemplazar las columnas imputadas
data[numeric_cols] = X_temp_imputed

print(f"Valores nulos imputados en {len(numeric_cols)} columnas")
print(f"Total de nulos después de imputación: {data.isnull().sum().sum()}")

# ==========================================
# FASE 13 — Codificación de etiquetas objetivo
# ==========================================

target_encoder = LabelEncoder()
data["target"] = target_encoder.fit_transform(data["attack_type"])

print("=== MAPEO DE CLASES ===")
for i, clase in enumerate(target_encoder.classes_):
    print(f"  {i} → {clase}")

# ==========================================
# FASE 14 — Definición de variables predictoras y objetivo
# ==========================================

# Dataset para Machine Learning (codificado)
X = data.drop(["attack_type", "target"], axis=1)
y = data["target"]

# Dataset original (sin codificar) para exportación
X_original = data_original.drop(["attack_type"], axis=1)

print(f"Variables predictoras: {X.shape[1]}")
print(f"Variables objetivo: {len(np.unique(y))} clases")

# ==========================================
# FASE 15 — División entrenamiento/prueba
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split original para exportación
X_train_original, X_test_original = train_test_split(
    X_original,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entrenamiento: {len(X_train)} muestras")
print(f"Prueba: {len(X_test)} muestras")

print("\nDistribución en entrenamiento:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nDistribución en prueba:")
print(pd.Series(y_test).value_counts().sort_index())

# ==========================================
# FASE 16 — Balanceo de clases con SMOTETomek
# ==========================================

print("=== APLICANDO SMOTETomek ===")

smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

print(f"Tamaño original del entrenamiento: {len(X_train)}")
print(f"Tamaño después del balanceo: {len(X_train_balanced)}")

print("\nDistribución después del balanceo:")
balanced_dist = pd.Series(y_train_balanced).value_counts().sort_index()
for clase, count in balanced_dist.items():
    print(f"  {target_encoder.classes_[clase]}: {count}")

# ==========================================
# FASE 17 — Definición de modelos a comparar
# ==========================================

# Diccionario de modelos con sus nombres
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss'),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=False, random_state=42)
}

print(f"Total de modelos a comparar: {len(models)}")
for name in models.keys():
    print(f"  ✓ {name}")

# ==========================================
# FASE 18 — Entrenamiento y evaluación de modelos
# ==========================================

print("\n=== ENTRENAMIENTO Y EVALUACIÓN DE MODELOS ===")
print("=" * 60)

comparison_results = []
model_predictions = {}
model_times = {}

total_models = len(models)
current = 0

for name, model in models.items():
    current += 1
    print(f"\n[{current}/{total_models}] 📊 {name}")
    print("-" * 40)

    start_time = time.time()
    model.fit(X_train_balanced, y_train_balanced)
    elapsed_time = time.time() - start_time
    model_times[name] = elapsed_time

    y_pred = model.predict(X_test)


    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()

    model_predictions[name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')

    comparison_results.append({
        "Modelo": name,
        "Accuracy": accuracy,
        "F1 Macro": f1_macro,
        "F1 Weighted": f1_weighted,
        "Precision Macro": precision_macro,
        "Recall Macro": recall_macro,
        "Tiempo (s)": elapsed_time
    })

    print(f"  ✅ F1 Macro: {f1_macro:.4f}")
    print(f"  ✅ Accuracy: {accuracy:.4f}")
    print(f"  ⏱️  Tiempo: {elapsed_time:.2f} segundos")

comparison_df = pd.DataFrame(comparison_results)

print("\n" + "=" * 60)
print("✅ Todos los modelos entrenados correctamente")

# ==========================================
# FASE 19 — Validación Cruzada (5-Fold)
# ==========================================

print("\n=== VALIDACIÓN CRUZADA (5-FOLD) ===")
print("=" * 60)

cv_results = []

for name, model in models.items():
    print(f"\n  Evaluando {name}...")


    model_clone = clone(model)

    scores = cross_val_score(
        model_clone,
        X_train_balanced,
        y_train_balanced,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )

    cv_results.append({
        "Modelo": name,
        "F1 Macro CV Promedio": scores.mean(),
        "F1 Macro CV Desv. Estándar": scores.std()
    })

    print(f"    F1 Macro: {scores.mean():.4f} ± {scores.std():.4f}")

cv_df = pd.DataFrame(cv_results)

# Unir resultados
comparison_df = comparison_df.merge(cv_df, on="Modelo")

print("\n=== RESUMEN COMPLETO DE COMPARACIÓN ===")
print(comparison_df.round(4).to_string(index=False))

# ==========================================
# FASE 20 — Visualización de la Comparación
# ==========================================

print("\n=== VISUALIZANDO RESULTADOS ===")

comparison_df_sorted = comparison_df.sort_values("F1 Macro", ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico 1: F1 Macro
ax1 = axes[0, 0]
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(comparison_df_sorted))]
bars = ax1.barh(comparison_df_sorted["Modelo"], comparison_df_sorted["F1 Macro"], color=colors)
ax1.set_xlabel("F1 Macro")
ax1.set_title("Comparación de Modelos - F1 Macro")
ax1.set_xlim(0, 1)
for bar, value in zip(bars, comparison_df_sorted["F1 Macro"]):
    ax1.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.4f}',
             va='center', ha='left', fontsize=10)

# Gráfico 2: Tiempo
ax2 = axes[0, 1]
comparison_df_sorted_time = comparison_df.sort_values("Tiempo (s)", ascending=True)
bars = ax2.barh(comparison_df_sorted_time["Modelo"], comparison_df_sorted_time["Tiempo (s)"], color='coral')
ax2.set_xlabel("Tiempo de Entrenamiento (segundos)")
ax2.set_title("Tiempo de Entrenamiento por Modelo")
for bar, value in zip(bars, comparison_df_sorted_time["Tiempo (s)"]):
    ax2.text(value + 0.5, bar.get_y() + bar.get_height()/2, f'{value:.1f}s',
             va='center', ha='left', fontsize=10)

# Gráfico 3: Métricas
ax3 = axes[1, 0]
metrics_to_plot = ["Accuracy", "F1 Macro", "F1 Weighted"]
x = np.arange(len(comparison_df_sorted))
width = 0.25
for i, metric in enumerate(metrics_to_plot):
    offset = (i - 1) * width
    ax3.bar(x + offset, comparison_df_sorted[metric], width, label=metric)
ax3.set_xticks(x)
ax3.set_xticklabels(comparison_df_sorted["Modelo"], rotation=45, ha='right')
ax3.set_ylabel("Puntuación")
ax3.set_title("Comparación de Métricas por Modelo")
ax3.legend()
ax3.set_ylim(0, 1)

# Gráfico 4: CV vs Directo
ax4 = axes[1, 1]
ax4.scatter(comparison_df["F1 Macro"], comparison_df["F1 Macro CV Promedio"], s=100)
ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5)
for idx, row in comparison_df.iterrows():
    ax4.annotate(row["Modelo"], (row["F1 Macro"], row["F1 Macro CV Promedio"]),
                fontsize=9, ha='center', va='bottom')
ax4.set_xlabel("F1 Macro (Entrenamiento Directo)")
ax4.set_ylabel("F1 Macro (CV Promedio)")
ax4.set_title("Correlación: Entrenamiento Directo vs CV")
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

plt.tight_layout()
plt.show()

# ==========================================
# FASE 21 — Identificación del Mejor Modelo
# ==========================================

print("\n=== IDENTIFICANDO EL MEJOR MODELO ===")
print("=" * 60)

# Verificar que las etiquetas sean consistentes
print("\n🔍 Verificando consistencia de etiquetas...")
print(f"Clases en target_encoder: {list(target_encoder.classes_)}")
print(f"Valores únicos en y_test: {np.unique(y_test)}")

# Identificar el mejor modelo por F1 Macro
best_model_row = comparison_df.loc[comparison_df["F1 Macro"].idxmax()]
best_model_name = best_model_row["Modelo"]
best_model = models[best_model_name]
best_pred = model_predictions[best_model_name]

# Asegurar que best_pred sea 1-dimensional
if len(best_pred.shape) > 1:
    best_pred = best_pred.flatten()

print(f"✅ best_pred shape después de corrección: {best_pred.shape}")
print(f"Valores únicos en best_pred: {np.unique(best_pred)}")

print(f"\n🏆 MEJOR MODELO: {best_model_name}")
print(f"   F1 Macro: {best_model_row['F1 Macro']:.4f}")
print(f"   F1 Weighted: {best_model_row['F1 Weighted']:.4f}")
print(f"   Accuracy: {best_model_row['Accuracy']:.4f}")
print(f"   Tiempo: {best_model_row['Tiempo (s)']:.2f}s")

print("\n=== RANKING DE MODELOS ===")
ranking = comparison_df[["Modelo", "F1 Macro", "F1 Weighted", "Accuracy", "Tiempo (s)"]].sort_values("F1 Macro", ascending=False)
for i, (idx, row) in enumerate(ranking.iterrows(), 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
    print(f"  {medal} {row['Modelo']}: F1 Macro = {row['F1 Macro']:.4f} ({row['Tiempo (s)']:.1f}s)")

# ==========================================
# FASE 22 — Evaluación Detallada del Mejor Modelo
# ==========================================

print(f"\n=== EVALUACIÓN DETALLADA: {best_model_name} ===")
print("=" * 60)

#  Asegurar que best_pred sea 1-dimensional
if len(best_pred.shape) > 1:
    best_pred = best_pred.flatten()

print(f"y_test shape: {y_test.shape}")
print(f"best_pred shape: {best_pred.shape}")

# Métricas
accuracy = accuracy_score(y_test, best_pred)
precision_macro = precision_score(y_test, best_pred, average="macro")
recall_macro = recall_score(y_test, best_pred, average="macro")
f1_macro = f1_score(y_test, best_pred, average="macro")
f1_weighted = f1_score(y_test, best_pred, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision Macro: {precision_macro:.4f}")
print(f"Recall Macro: {recall_macro:.4f}")
print(f"F1 Macro: {f1_macro:.4f}")
print(f"F1 Weighted: {f1_weighted:.4f}")

# Verificar coincidencias
matches = (y_test == best_pred)
print(f"\nCoincidencias: {matches.sum()} de {len(matches)}")
print(f"Tasa de acierto: {matches.mean():.4f}")

print("\n=== CLASSIFICATION REPORT ===")
report = classification_report(
    y_test,
    best_pred,
    target_names=target_encoder.classes_,
    output_dict=True
)
report_df = pd.DataFrame(report).transpose()
print(report_df.round(4))

print("\n=== CLASES PROBLEMÁTICAS (F1 < 0.7) ===")
problematic_classes = []
for clase in target_encoder.classes_:
    idx = target_encoder.transform([clase])[0]
    if str(idx) in report:
        f1 = report[str(idx)]['f1-score']
        if f1 < 0.7:
            problematic_classes.append((clase, f1))
            print(f"  ⚠️ {clase}: F1 = {f1:.4f}")

if not problematic_classes:
    print("  ✅ Todas las clases tienen F1 >= 0.7")

# ==========================================
# FASE 23 — Matriz de Confusión del Mejor Modelo
# ==========================================

print(f"\n=== MATRIZ DE CONFUSIÓN: {best_model_name} ===")

cm = confusion_matrix(y_test, best_pred)

# Visualización
plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_encoder.classes_,
    yticklabels=target_encoder.classes_,
    annot_kws={'size': 10}
)
plt.xlabel("Predicción", fontsize=12)
plt.ylabel("Valor Real", fontsize=12)
plt.title(f"Matriz de Confusión - {best_model_name}", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#  ANÁLISIS DE ERRORES
print("\n=== ANÁLISIS DE ERRORES POR CLASE ===")
error_analysis = []

for i, clase in enumerate(target_encoder.classes_):
    total = cm[i, :].sum()
    aciertos = cm[i, i]
    errores = total - aciertos
    error_rate = errores / total if total > 0 else 1

    error_analysis.append({
        "Clase": clase,
        "Total": total,
        "Aciertos": aciertos,
        "Errores": errores,  
        "Error Rate": error_rate,
        "Principales Confusiones": []
    })

    if aciertos < total:
        confusions = [(j, cm[i, j]) for j in range(len(target_encoder.classes_))
                     if j != i and cm[i, j] > 0]
        confusions.sort(key=lambda x: x[1], reverse=True)
        error_analysis[-1]["Principales Confusiones"] = [
            (target_encoder.classes_[j], count) for j, count in confusions[:3]
        ]

# Mostrar análisis (solo clases con error > 10%)
print("\n📊 Clases con Error Rate > 10%:")
has_problems = False
for analysis in error_analysis:
    if analysis["Error Rate"] > 0.1:
        has_problems = True
        print(f"\n📌 {analysis['Clase']}:")
        print(f"   Error Rate: {analysis['Error Rate']*100:.1f}% ({analysis['Errores']}/{analysis['Total']})")
        if analysis["Principales Confusiones"]:
            print("   Confundido con:")
            for clase_conf, count in analysis["Principales Confusiones"]:
                print(f"     → {clase_conf}: {count}")

if not has_problems:
    print("  ✅ Todas las clases tienen Error Rate <= 10%")

# ==========================================
# FASE 24 — Exportación de Resultados para Neo4j
# ==========================================

print("\n=== EXPORTANDO RESULTADOS ===")

# Asegurar que best_pred sea 1-dimensional
if len(best_pred.shape) > 1:
    best_pred = best_pred.flatten()

# Usar X_test_original (datos sin codificar)
results_df = X_test_original.copy()

# Agregar etiquetas (decodificadas)
results_df["real_label"] = target_encoder.inverse_transform(y_test)
results_df["predicted_label"] = target_encoder.inverse_transform(best_pred)
results_df["correct"] = (results_df["real_label"] == results_df["predicted_label"])

# Verificar formato
print("\n📋 Columnas en el archivo de exportación:")
print(results_df.columns.tolist())

print("\n🔍 Muestra de datos (formato original):")
print(results_df.head())

# Exportar
output_path = "/content/drive/MyDrive/UWF-ZeekData24/predictions_for_neo4j.csv"
results_df.to_csv(output_path, index=False)

print(f"\n✅ Archivo exportado en FORMATO ORIGINAL")
print(f"📍 Ruta: {output_path}")
print(f"📊 Total: {len(results_df)} registros")
print(f"✅ Aciertos: {results_df['correct'].sum()}")
print(f"❌ Errores: {(~results_df['correct']).sum()}")

# ==========================================
# FASE 25 — Resumen Final
# ==========================================

print("\n" + "=" * 70)
print("                    RESUMEN FINAL")
print("=" * 70)

print(f"\n📊 DATASET:")
print(f"  • Total: {len(data)} registros")
print(f"  • Características: {X.shape[1]}")
print(f"  • Clases: {len(target_encoder.classes_)}")

print(f"\n🏆 MEJOR MODELO: {best_model_name}")
print(f"  • F1 Macro: {best_model_row['F1 Macro']:.4f}")
print(f"  • Accuracy: {best_model_row['Accuracy']:.4f}")
print(f"  • Tiempo: {best_model_row['Tiempo (s)']:.2f}s")
print(f"  • F1 Macro CV: {best_model_row['F1 Macro CV Promedio']:.4f} ± {best_model_row['F1 Macro CV Desv. Estándar']:.4f}")

print(f"\n📈 RANKING:")
print(ranking.round(4).to_string(index=False))

print(f"\n⚠️ CLASES PROBLEMÁTICAS:")
if problematic_classes:
    for clase, f1 in problematic_classes:
        print(f"  • {clase}: F1 = {f1:.4f}")
else:
    print("  ✅ Todas las clases tienen F1 >= 0.7")

print(f"\n📂 ARCHIVO GENERADO:")
print(f"  • predictions_for_neo4j.csv")
print(f"  • {len(results_df)} registros en FORMATO ORIGINAL")

print("\n" + "=" * 70)
print("              ✅ PROYECTO COMPLETADO")
print("=" * 70)
