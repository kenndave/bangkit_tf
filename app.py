from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.background import BackgroundScheduler
import tensorflow as tf
import numpy as np
import datetime
from pydantic_settings import BaseSettings
from google.cloud import firestore

app = FastAPI()

scheduler = BackgroundScheduler()

model = None


class Settings(BaseSettings):
    project_id: str = "capstone-project-442502"
    database: str = "bangkit-db"

    class Config:
        env_file = ".env"


settings = Settings()

db = firestore.Client(database=settings.database, project=settings.project_id)


def get_user_transactions(user_id: str):
    transactions_ref = (
        db.collection("users").document(user_id).collection("transactions")
    )
    transactions = transactions_ref.stream()

    transaction_list = []
    for transaction in transactions:
        data = transaction.to_dict()
        timestamp = data.get("timestamp")
        total_price = data.get("total_price")
        transaction_list.append({"timestamp": timestamp, "total_price": total_price})

    transaction_list.sort(key=lambda x: x["timestamp"])

    timestamps = [transaction["timestamp"] for transaction in transaction_list]
    total_prices = [transaction["total_price"] for transaction in transaction_list]

    return timestamps, total_prices


def train_model(user_id: str, data, time_unit):
    timestamps, total_prices = data

    if len(total_prices) < 12:
        print(f"Not enough data to train the model for user: {user_id}. Skipping...")
        return

    aggregated_data = {}
    for i, timestamp in enumerate(timestamps):
        current_time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")

        if time_unit == "months":
            key = current_time.strftime("%Y-%m")
        elif time_unit == "weeks":
            start_of_week = current_time - datetime.timedelta(
                days=current_time.weekday()
            )
            key = start_of_week.strftime("%Y-%m-%d")
        else:
            key = current_time.strftime("%Y-%m-%d")

        if key not in aggregated_data:
            aggregated_data[key] = 0
        aggregated_data[key] += total_prices[i]

    sorted_aggregated_data = sorted(aggregated_data.items())
    _, aggregated_prices = zip(*sorted_aggregated_data)

    aggregated_prices = np.array(aggregated_prices, dtype=np.float32)
    if len(aggregated_prices) < 12:
        print(
            f"Not enough aggregated data to train the model for user: {user_id}. Skipping..."
        )
        return

    X_train = []
    y_train = []

    for i in range(len(aggregated_prices) - 12):
        X_train.append(aggregated_prices[i : i + 12])
        y_train.append(aggregated_prices[i + 12])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                256,
                activation="relu",
                return_sequences=True,
                input_shape=(X_train.shape[1], 1),
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(128, activation="relu", return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_absolute_percentage_error")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=False)

    model_path = f"models/{user_id}_{time_unit}_transaction_model.h5"
    model.save(model_path)

    print(f"Model trained and saved for user: {user_id} at {datetime.datetime.now()}")


def retrain_models_daily():
    users_ref = db.collection("users")
    users = users_ref.stream()

    for user in users:
        user_id = user.id
        print(f"Retraining model for user: {user_id}")

        data = get_user_transactions(user_id)
        if len(data[1]) < 2:
            print(f"Skipping user {user_id}: Not enough data for training.")
            continue

        timestamps, _ = data
        first_timestamp = datetime.datetime.strptime(
            timestamps[0], "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        last_timestamp = datetime.datetime.strptime(
            timestamps[-1], "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        data_range = (last_timestamp - first_timestamp).days

        if data_range > 730:
            time_unit = "months"
        elif data_range > 365:
            time_unit = "weeks"
        else:
            time_unit = "days"

        train_model(user_id, data, time_unit)

    print(f"Models retrained at {datetime.datetime.now()}")


scheduler.add_job(
    retrain_models_daily, "interval", days=1, start_date=datetime.datetime.now()
)


@app.on_event("startup")
async def startup_event():
    print("Starting initial training...")
    retrain_models_daily()
    scheduler.start()


@app.get("/")
async def root():
    return {"message": "Forecasting is ready!"}


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "OK", "message": "Service is running"}


@app.get("/predict/{user_id}")
async def predict_future_steps(user_id: str):
    timestamps, total_prices = get_user_transactions(user_id)
    if len(total_prices) == 0:
        return {"user_id": user_id, "current_data": [], "predictions": []}

    sorted_data = sorted(zip(timestamps, total_prices), key=lambda x: x[0])
    timestamps, total_prices = zip(*sorted_data)

    first_timestamp = datetime.datetime.strptime(timestamps[0], "%Y-%m-%dT%H:%M:%S.%fZ")
    last_timestamp = datetime.datetime.strptime(timestamps[-1], "%Y-%m-%dT%H:%M:%S.%fZ")
    data_range = (last_timestamp - first_timestamp).days

    if data_range > 730:
        time_unit = "months"
    elif data_range > 365:
        time_unit = "weeks"
    else:
        time_unit = "days"

    aligned_data = []
    if time_unit == "months":
        grouped_data = {}
        for i, timestamp in enumerate(timestamps):
            current_time = datetime.datetime.strptime(
                timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            month_year = current_time.strftime("%Y-%m")
            grouped_data[month_year] = grouped_data.get(month_year, 0) + total_prices[i]

        for month_year, price_sum in grouped_data.items():
            aligned_data.append(
                {"timestamp": f"{month_year}-01T00:00:00Z", "price": price_sum}
            )

    elif time_unit == "weeks":
        grouped_data = {}
        for i, timestamp in enumerate(timestamps):
            current_time = datetime.datetime.strptime(
                timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            start_of_week = current_time - datetime.timedelta(
                days=current_time.weekday()
            )
            week_str = start_of_week.strftime("%Y-%m-%d")
            grouped_data[week_str] = grouped_data.get(week_str, 0) + total_prices[i]

        for week_str, price_sum in grouped_data.items():
            aligned_data.append(
                {"timestamp": f"{week_str}T00:00:00Z", "price": price_sum}
            )

    else:
        grouped_data = {}
        for i, timestamp in enumerate(timestamps):
            current_time = datetime.datetime.strptime(
                timestamp, "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            day_str = current_time.strftime("%Y-%m-%d")
            grouped_data[day_str] = grouped_data.get(day_str, 0) + total_prices[i]

        for day_str, price_sum in grouped_data.items():
            aligned_data.append(
                {"timestamp": f"{day_str}T00:00:00Z", "price": price_sum}
            )

    current_data = aligned_data

    try:
        model = tf.keras.models.load_model(
            f"models/{user_id}_{time_unit}_transaction_model.h5"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"user_id": user_id, "current_data": current_data, "predictions": []}

    if len(aligned_data) < 12:
        print("Not enough data to make predictions.")
        return {"user_id": user_id, "current_data": current_data, "predictions": []}

    last_prices = np.array(
        [entry["price"] for entry in aligned_data[-12:]], dtype=np.float32
    ).reshape(1, 12, 1)

    predictions = []
    current_time = datetime.datetime.strptime(
        aligned_data[-1]["timestamp"], "%Y-%m-%dT%H:%M:%SZ"
    )
    forecast_steps = 12

    for _ in range(1, forecast_steps + 1):
        next_price = model.predict(last_prices, verbose=0)
        next_price_value = float(next_price[0][0])

        if time_unit == "months":
            current_time += datetime.timedelta(days=30)
        elif time_unit == "weeks":
            current_time += datetime.timedelta(weeks=1)
        else:
            current_time += datetime.timedelta(days=1)

        predictions.append(
            {"timestamp": current_time.isoformat() + "Z", "price": next_price_value}
        )

        last_prices = np.append(last_prices[:, 1:, :], [[[next_price_value]]], axis=1)

    return {
        "user_id": user_id,
        "current_data": current_data,
        "predictions": predictions,
    }
