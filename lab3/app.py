import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Функція для очищення тексту від HTML-тегів
def remove_html(text):
    return re.sub(r'<.*?>', '', text)

# Функція для зчитування та обробки CSV-файлів
def read_clean_csv(filepath):
    headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI', 'empty']
    try:
        df = pd.read_csv(
            filepath, header=1, names=headers,
            converters={'Year': remove_html}, skipinitialspace=True
        )
        df = df.drop(columns=['empty'], errors='ignore')
        df = df.dropna()
        df = df[df['VHI'] != -1]

        # Визначаємо область за назвою файлу
        match = re.search(r'NOAA_ID(\d+)_', filepath)
        region_id = int(match.group(1)) if match else None
        df['region_id'] = region_id

        # Перевести стовпець Year в числовий тип
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

        return df
    except pd.errors.ParserError as e:
        print(f"Помилка при зчитуванні {filepath}: {e}")
        return None

# Функція для зчитування всіх файлів у директорії
def load_data_from_directory(directory):
    data_frames = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = read_clean_csv(file_path)
            if df is not None:
                data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

# Завантаження даних
DATA_DIR = "data"
df = load_data_from_directory(DATA_DIR)

if df.empty:
    st.error("Не вдалося зчитати жоден файл. Перевірте формат CSV-файлів!")
    st.stop()

# Ініціалізація session_state з перевіркою на один рік
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

if "selected_param" not in st.session_state:
    st.session_state.selected_param = "VCI"
if "region_ids" not in st.session_state:
    st.session_state.region_ids = df['region_id'].unique()[0]
if "week_range" not in st.session_state:
    st.session_state.week_range = (1, 52)
if "year_range" not in st.session_state:
    st.session_state.year_range = (min_year, max_year)

# Створення макету
col1, col2 = st.columns([1, 2])

with col1:
    selected_param = st.selectbox(
        "Оберіть параметр",
        ["VCI", "TCI", "VHI"],
        key="selected_param"
    )
    
    region_ids = df['region_id'].unique()
    selected_region = st.selectbox(
        "Оберіть область (region_id)",
        options=region_ids,
        key="region_ids"
    )
    
    week_range = st.slider(
        "Виберіть інтервал тижнів",
        1, 52,
        (1, 52),
        key="week_range"
    )

    # Перевірка для слайдера років
    if min_year == max_year:
        st.write(f"Дані доступні лише за {min_year} рік")
        year_range = (min_year, max_year)
        st.session_state.year_range = year_range  # оновити session_state вручну
    else:
        year_range = st.slider(
            "Виберіть інтервал років",
            min_year,
            max_year,
            (min_year, max_year),
            key="year_range"
        )
    
    def clear():
        st.session_state.update({
            "selected_param": "VCI",
            "region_ids": df['region_id'].unique()[0],
            "week_range": (1, 52),
            "year_range": (min_year, max_year),
            "sort_asc": False,
            "sort_desc": False
        })

    st.button("Скинути фільтри", on_click=clear)

    sort_ascending = st.checkbox("Сортувати за зростанням", key="sort_asc")
    sort_descending = st.checkbox("Сортувати за спаданням", key="sort_desc")

# Фільтрація даних
filtered_df = df[
    (df["region_id"] == selected_region) &
    (df["Year"].between(*year_range)) &
    (df["Week"].between(*week_range))
]

# Сортування
if sort_ascending and sort_descending:
    st.warning("Неможливо одночасно сортувати за зростанням і спаданням.")
elif sort_ascending:
    filtered_df = filtered_df.sort_values(by=selected_param, ascending=True)
elif sort_descending:
    filtered_df = filtered_df.sort_values(by=selected_param, ascending=False)

with col2:
    tab1, tab2, tab3 = st.tabs(["Таблиця", "Графік", "Порівняння по областях"])

    with tab1:
        st.dataframe(filtered_df)

    with tab2:
        fig, ax = plt.subplots()
        sns.lineplot(data=filtered_df, x="Week", y=selected_param, ax=ax)
        ax.set_title(f"{selected_param} для region_id {selected_region}")
        st.pyplot(fig)

    with tab3:
        comparison_df = df[
            (df["Year"].between(*year_range)) &
            (df["Week"].between(*week_range))
        ].copy()
        comparison_df["highlight"] = comparison_df["region_id"].apply(
            lambda x: "Обрана область" if x == selected_region else "Інші області"
        )
        fig, ax = plt.subplots()
        sns.boxplot(data=comparison_df, x="region_id", y=selected_param, hue="highlight", ax=ax)
        ax.set_title(f"Порівняння {selected_param} для region_id {selected_region} з іншими")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=2)
        st.pyplot(fig)
