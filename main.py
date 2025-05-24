# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pickle
import warnings

# Игнорируем предупреждения
warnings.filterwarnings('ignore')



def haversine(lon1, lat1, lon2, lat2):
    """Рассчитывает расстояние между двумя точками на Земле (в км)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))


def preprocess_data(df):
    """Предобработка и очистка данных"""
    # Создаем копию DataFrame
    df_clean = df.copy()

    # Преобразуем дату/время
    df_clean['pickup_datetime'] = pd.to_datetime(
        df_clean['pickup_datetime'],
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )

    # Удаляем строки с некорректными датами
    df_clean = df_clean.dropna(subset=['pickup_datetime'])

    # Фильтрация аномальных значений
    df_clean = df_clean[
        (df_clean['trip_duration'] > 60) &
        (df_clean['trip_duration'] < 24 * 3600)
        ].copy()

    # Добавляем временные признаки
    df_clean['hour'] = df_clean['pickup_datetime'].dt.hour
    df_clean['weekday'] = df_clean['pickup_datetime'].dt.weekday
    df_clean['month'] = df_clean['pickup_datetime'].dt.month
    df_clean['is_weekend'] = df_clean['weekday'].isin([5, 6]).astype(int)

    # Рассчитываем расстояние
    df_clean['distance_km'] = df_clean.apply(
        lambda row: haversine(
            row['pickup_longitude'], row['pickup_latitude'],
            row['dropoff_longitude'], row['dropoff_latitude']
        ), axis=1
    )

    # Удаляем аномалии расстояния
    df_clean = df_clean[
        (df_clean['distance_km'] > 0.1) &
        (df_clean['distance_km'] < 100)
        ]

    return df_clean



# Основной код

if __name__ == "__main__":
    print("=== NYC Taxi Trip Duration Prediction ===")

    # Загрузка данных
    print("\n[1/5] Загрузка данных...")
    try:
        df = pd.read_csv('train.csv')
        print(f"Загружено {len(df)} записей")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        exit()

    # Предобработка
    print("\n[2/5] Предобработка данных...")
    try:
        df = preprocess_data(df)
        print(f"После очистки осталось {len(df)} записей")
        print("\nПример данных:")
        print(df[['pickup_datetime', 'hour', 'weekday', 'distance_km', 'trip_duration']].head())
    except Exception as e:
        print(f"Ошибка предобработки: {e}")
        exit()

    # Визуализация
    print("\n[3/5] Создание визуализаций...")
    try:
        # Распределение поездок по часам
        plt.figure(figsize=(12, 6))
        df['hour'].value_counts().sort_index().plot(kind='bar')
        plt.title('Распределение поездок по часам дня')
        plt.xlabel('Час дня')
        plt.ylabel('Количество поездок')
        plt.grid()
        plt.savefig('trips_by_hour.png')
        plt.close()

        # Распределение длительности поездок
        plt.figure(figsize=(12, 6))
        plt.hist(df['trip_duration'] / 60, bins=50)
        plt.title('Распределение длительности поездок')
        plt.xlabel('Длительность (минуты)')
        plt.ylabel('Количество поездок')
        plt.grid()
        plt.savefig('trip_duration_dist.png')
        plt.close()

        print("Графики сохранены в файлы: trips_by_hour.png, trip_duration_dist.png")
    except Exception as e:
        print(f"Ошибка визуализации: {e}")

    # Подготовка данных для модели
    print("\n[4/5] Подготовка данных для модели...")
    try:
        X = df[['distance_km', 'hour', 'weekday', 'month', 'is_weekend']]
        y = df['trip_duration'] / 60  # Переводим секунды в минуты

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Размеры выборок: Train - {X_train.shape}, Test - {X_test.shape}")
    except Exception as e:
        print(f"Ошибка подготовки данных: {e}")
        exit()

    # Обучение модели
    print("\n[5/5] Обучение модели CatBoost...")
    try:
        model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=10,
            l2_leaf_reg=5,
            random_seed=42,
            verbose=100
        )
        model.fit(X_train, y_train)

        # Оценка модели
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"\nРезультаты:")
        print(f"- Средняя абсолютная ошибка (MAE): {mae:.2f} минут")
        print(f"- Пример предсказания: {y_pred[0]:.1f} мин (реальное: {y_test.iloc[0]:.1f} мин)")

        # Важность признаков
        feature_importance = model.get_feature_importance()
        plt.figure(figsize=(10, 5))
        plt.bar(X.columns, feature_importance)
        plt.title('Важность признаков')
        plt.ylabel('Важность')
        plt.savefig('feature_importance.png')
        plt.close()
        print("График важности признаков сохранен в feature_importance.png")

        # Сохранение модели
        model.save_model('taxi_trip_model.cbm')
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("Модель сохранена в файлы: taxi_trip_model.cbm, model.pkl")

    except Exception as e:
        print(f"Ошибка обучения модели: {e}")
        exit()

    print("\n=== Обработка завершена успешно ===")