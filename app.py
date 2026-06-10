import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from sklearn.metrics import confusion_matrix

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "bots_vs_users.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model_weights.mw")

def preprocess(df):
    df = df.copy()
    df = df.replace("Unknown", np.nan)
    df["city"] = df["city"].apply(lambda x: 1 if pd.notna(x) else 0)
    for c in df.columns:
        if c != "target":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    if "is_confirmed" in df.columns:
        df.drop("is_confirmed", axis=1, inplace=True)
    return df

@st.cache_data
def load_data():
    return preprocess(pd.read_csv(DATA_PATH))

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

# Заголовок 
st.title("Боты vs Люди — анализ аккаунтов ВКонтакте")
st.write("Target: 1 — бот, 0 — человек.")

# фильтр по классу 
show = st.selectbox("Показывать аккаунты", ["Все", "Только людей", "Только ботов"])
if show == "Только людей":
    df_view = df[df["target"] == 0]
elif show == "Только ботов":
    df_view = df[df["target"] == 1]
else:
    df_view = df

st.markdown("---")

# Первичный анализ
st.subheader("Первичный анализ")
st.write(df_view.describe())

# # Распределение классов
# st.subheader("Распределение классов")
# fig, ax = plt.subplots()
# df_view["target"].value_counts().sort_index().plot(kind="bar", ax=ax,
#     color=["steelblue", "tomato"], edgecolor="white")
# ax.set_xticklabels(["Человек (0)", "Бот (1)"], rotation=0)
# ax.set_ylabel("Количество")
# st.pyplot(fig)

st.markdown("---")

# выбор признака для анализа
st.subheader("Графическое представление признака")
feat = st.selectbox("Выберите признак", [c for c in df_view.columns if c != "target"])

fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, cls in enumerate([0, 1]):
    vals = df_view[df_view["target"] == cls][feat]
    axes[i].hist(vals.clip(upper=vals.quantile(0.99)), bins=20,
                 color=["steelblue", "tomato"][i], edgecolor="white")
    axes[i].set_title(f"{'Люди' if cls == 0 else 'Боты'} — {feat}")
    axes[i].axvline(vals.mean(), color="black", linestyle="--",
                    label=f"среднее: {vals.mean():.2f}")
    axes[i].legend()
st.pyplot(fig2)

st.markdown("---")

# Корреляции 
st.subheader("Корреляция признаков с таргетом")
corr = df_view.corr()["target"].drop("target").abs().sort_values(ascending=False).head(15)

fig3, ax3 = plt.subplots(figsize=(8, 5))
corr[::-1].plot(kind="barh", ax=ax3, color="steelblue")
ax3.set_xlabel("Корреляция с target (по модулю)")
ax3.set_title("Топ-15 признаков")
st.pyplot(fig3)

st.markdown("---")

# Прогноз
st.subheader("Прогноз: бот или человек?")
st.write("Заполните несколько характеристик аккаунта и нажмите кнопку.")

col1, col2 = st.columns(2)
with col1:
    has_photo = st.selectbox("Есть фото?",    [("Да", 1), ("Нет", 0)], format_func=lambda x: x[0])[1]
    has_mobile = st.selectbox("Есть телефон?", [("Да", 1), ("Нет", 0)], format_func=lambda x: x[0])[1]
    city = st.selectbox("Город указан?", [("Да", 1), ("Нет", 0)], format_func=lambda x: x[0])[1]
    has_status = st.selectbox("Есть статус?",  [("Да", 1), ("Нет", 0)], format_func=lambda x: x[0])[1]
with col2:
    # Контрол 3: числовые признаки
    subscribers = st.number_input("Количество подписчиков", 0, 200000, 0, step=10)
    avg_likes = st.slider("Среднее лайков на пост", 0.0, 200.0, 0.0)
    avg_comments = st.slider("Среднее комментариев", 0.0, 20.0, 0.0, 0.1)

if st.button("Определить"):
    inp = {f: 0.0 for f in feature_names}
    inp["has_photo"] = float(has_photo)
    inp["has_mobile"] = float(has_mobile)
    inp["city"] = float(city)
    inp["has_status"] = float(has_status)
    inp["subscribers_count"] = float(subscribers)
    inp["avg_likes"] = float(avg_likes)
    inp["avg_comments"] = float(avg_comments)

    X_inp = np.array([[inp[f] for f in feature_names]])
    X_inp_sc = scaler.transform(X_inp)
    pred = model.predict(X_inp_sc)[0]
    prob = model.predict_proba(X_inp_sc)[0]

    if pred == 1:
        st.error(f"Это БОТ — вероятность {prob[1]:.1%}")
    else:
        st.success(f"Это ЧЕЛОВЕК — вероятность {prob[0]:.1%}")
