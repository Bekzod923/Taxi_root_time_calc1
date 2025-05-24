from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import logging
from werkzeug.exceptions import HTTPException


app = Flask(__name__)

# 2. Настройка логгирования (все ошибки сохраняются в app.log)
logging.basicConfig(
    filename='app.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 3. Загрузка модели
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    logging.info("Модель успешно загружена")
except Exception as e:
    logging.critical(f"Ошибка загрузки модели: {e}")
    raise


# 4. Функция для расчёта расстояния
def haversine(lon1, lat1, lon2, lat2):
    """Рассчитывает расстояние между двумя точками на Земле (в км)."""
    try:
        # Преобразование градусов в радианы
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # Разница координат
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        # Формула гаверсинуса
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        distance = 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))

        return distance
    except Exception as e:
        logging.error(f"Ошибка в haversine: {e}")
        raise


# 5. Маршрут для главной страницы
@app.route('/', methods=['GET'])
def home():
    """Отображает HTML-форму для ввода данных."""
    return render_template('index.html')


# 6. Маршрут для обработки предсказаний
@app.route('/predict', methods=['POST'])
def predict():
    """Обрабатывает AJAX-запросы с фронтенда."""
    try:
        # Проверяем, что данные пришли в JSON-формате
        if not request.is_json:
            return jsonify({"error": "Требуется JSON-данные"}), 400

        data = request.get_json()

        # Валидация полей
        required_fields = ['start_lon', 'start_lat', 'end_lon', 'end_lat', 'hour']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Не хватает обязательных полей"}), 400

        # Расчёт расстояния
        distance = haversine(
            float(data['start_lon']),
            float(data['start_lat']),
            float(data['end_lon']),
            float(data['end_lat'])
        )

        # Создаём DataFrame для модели
        features = pd.DataFrame([[
            distance,
            int(data['hour']),  # час
            1,  # день недели (1 = понедельник)
            6,  # месяц (июнь)
            0  # не выходной
        ]], columns=['distance_km', 'hour', 'weekday', 'month', 'is_weekend'])

        # Предсказание
        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": round(prediction, 1),
            "distance_km": round(distance, 2)
        })

    except ValueError as e:
        logging.error(f"Ошибка ввода: {e}")
        return jsonify({"error": "Некорректные числовые значения"}), 400
    except Exception as e:
        logging.error(f"Ошибка предсказания: {e}")
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


# 7. Обработчик ошибок
@app.errorhandler(HTTPException)
def handle_exception(e):
    logging.error(f"HTTP ошибка {e.code}: {e.description}")
    return jsonify({"error": str(e.description)}), e.code


# 8. Запуск сервера
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)