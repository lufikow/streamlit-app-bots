import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "bots_vs_users.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model_weights.mw")

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Unknown → NaN, затем числовая конвертация
    df = df.replace("Unknown", np.nan)

    # city: бинарный признак «город указан»
    df["city"] = df["city"].apply(lambda x: 1 if pd.notna(x) else 0)

    # Все остальные колонки — в float, NaN → 0
    for c in df.columns:
        if c != "target":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    # is_confirmed — однозначный признак (все = 1), удаляем
    if "is_confirmed" in df.columns:
        df.drop("is_confirmed", axis=1, inplace=True)

    return df


# Загрузка и обработка
print("Загрузка данных...")
raw = pd.read_csv(DATA_PATH)
df  = preprocess(raw)

X = df.drop("target", axis=1)
y = df["target"]

# Train / Test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Обучение 
print("Обучение RandomForestClassifier (n=100)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_sc, y_train)

# Метрики
pred = model.predict(X_test_sc)
acc = accuracy_score(y_test, pred)
print(f"\nТочность на тестовой выборке : {acc:.4f}")
print("\nОтчёт по классификации:")
print(classification_report(y_test, pred, target_names=["Человек", "Бот"]))

# Feature importances
importances = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)
print("Топ-10 признаков по важности:")
print(importances.head(10))

# Сохранение
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump({
    "model": model,
    "scaler": scaler,
    "feature_names": list(X.columns),
    "importances": importances,
}, MODEL_PATH)
print(f"\nВеса сохранены → {MODEL_PATH}")
