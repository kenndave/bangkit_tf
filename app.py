from fastapi import FastAPI, HTTPException
from apscheduler.schedulers.background import BackgroundScheduler
import tensorflow as tf
import numpy as np
import datetime
from pydantic_settings import BaseSettings
from google.cloud import firestore
from google.oauth2 import service_account

app = FastAPI()

scheduler = BackgroundScheduler()

model = None

class Settings(BaseSettings):
    firebase_credentials: str
    project_id: str
    database: str

    class Config:
        env_file = ".env"

settings = Settings()

credentials = service_account.Credentials.from_service_account_file(settings.firebase_credentials)

db = firestore.Client(
    database=settings.database,
    project=settings.project_id,
    credentials=credentials
)

def get_user_transactions(user_id: str):
    transactions_ref = db.collection('users').document(user_id).collection('transactions')
    transactions = transactions_ref.stream()

    transaction_list = []
    for transaction in transactions:
        data = transaction.to_dict()
        timestamp = data.get('timestamp')
        total_price = data.get('total_price')
        transaction_list.append({'timestamp': timestamp, 'total_price': total_price})

    transaction_list.sort(key=lambda x: x['timestamp'])

    timestamps = [transaction['timestamp'] for transaction in transaction_list]
    total_prices = [transaction['total_price'] for transaction in transaction_list]
    
    return timestamps, total_prices

def train_model(user_id: str, data, time_unit):
    _, total_prices = data

    if len(total_prices) < 2:
        print(f"Not enough data to train the model for user: {user_id}. Skipping...")
        return

    total_prices = np.array(total_prices, dtype=np.float32)
    X_train = total_prices[:-1].reshape(-1, 1, 1)
    y_train = total_prices[1:]

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(256, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.LSTM(128, activation='relu', return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    model_path = f'models/{user_id}_{time_unit}_transaction_model.h5'
    model.save(model_path)

    print(f"Model trained and saved for user: {user_id} at {datetime.datetime.now()}")

def retrain_models_daily():
    users_ref = db.collection('users')
    users = users_ref.stream()

    for user in users:
        user_id = user.id
        print(f"Retraining model for user: {user_id}")

        data = get_user_transactions(user_id)
        if len(data[1]) < 2:
            print(f"Skipping user {user_id}: Not enough data for training.")
            continue
        
        timestamps, total_prices = data
        first_timestamp = datetime.datetime.strptime(timestamps[0], "%Y-%m-%dT%H:%M:%S.%fZ")
        last_timestamp = datetime.datetime.strptime(timestamps[-1], "%Y-%m-%dT%H:%M:%S.%fZ")
        data_range = (last_timestamp - first_timestamp).days
        
        if data_range > 730:
            time_unit = "months"
        elif data_range > 365:
            time_unit = "weeks"
        else:
            time_unit = "days"
        
        train_model(user_id, data, time_unit)

    print(f"Models retrained at {datetime.datetime.now()}")

scheduler.add_job(retrain_models_daily, 'interval', days=1, start_date=datetime.datetime.now())

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
        return {
            "user_id": user_id,
            "current_data": [],
            "predictions": []
        }

    sorted_data = sorted(zip(timestamps, total_prices), key=lambda x: x[0])
    timestamps, total_prices = zip(*sorted_data)

    current_data = []

    first_timestamp = datetime.datetime.strptime(timestamps[0], "%Y-%m-%dT%H:%M:%S.%fZ")
    last_timestamp = datetime.datetime.strptime(timestamps[-1], "%Y-%m-%dT%H:%M:%S.%fZ")
    data_range = (last_timestamp - first_timestamp).days
    
    if data_range > 730:  # More than 2 years
        time_unit = "months"
    elif data_range > 365:  # More than 1 year but less than 2 years
        time_unit = "weeks"
    else:  # Less than or equal to 1 year
        time_unit = "days"
    
    aligned_data = []
    current_time = datetime.datetime.strptime(timestamps[0], "%Y-%m-%dT%H:%M:%S.%fZ")

    if time_unit == "months":
        grouped_data = {}
        for i, timestamp in enumerate(timestamps):
            current_time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            month_year = current_time.strftime("%Y-%m")
            if month_year not in grouped_data:
                grouped_data[month_year] = 0
            grouped_data[month_year] += total_prices[i]

        for month_year, price_sum in grouped_data.items():
            aligned_data.append({
                "timestamp": f"{month_year}-01T00:00:00Z",
                "price": price_sum
            })

    elif time_unit == "weeks":
        grouped_data = {}
        for i, timestamp in enumerate(timestamps):
            current_time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            start_of_week = current_time - datetime.timedelta(days=current_time.weekday())
            week_str = start_of_week.strftime("%Y-%m-%d")
            if week_str not in grouped_data:
                grouped_data[week_str] = 0
            grouped_data[week_str] += total_prices[i]

        for week_str, price_sum in grouped_data.items():
            aligned_data.append({
                "timestamp": f"{week_str}T00:00:00Z",
                "price": price_sum
            })

    else:
        grouped_data = {}
        for i, timestamp in enumerate(timestamps):
            current_time = datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
            day_str = current_time.strftime("%Y-%m-%d")
            if day_str not in grouped_data:
                grouped_data[day_str] = 0
            grouped_data[day_str] += total_prices[i]

        for day_str, price_sum in grouped_data.items():
            aligned_data.append({
                "timestamp": f"{day_str}T00:00:00Z",
                "price": price_sum
            })

    current_data = aligned_data

    try:
        model = tf.keras.models.load_model(f'models/{user_id}_{time_unit}_transaction_model.h5')
    except Exception as e:
        print(e)
        return {
            "user_id": user_id,
            "current_data": current_data,
            "predictions": []
        }

    last_price = np.array(total_prices[-1], dtype=np.float32).reshape(1, 1, 1)
    predictions = []
    forecast_steps = 12

    for step in range(1, forecast_steps + 1):
        if time_unit == "months":
            next_timestamp = current_time + datetime.timedelta(weeks=4 * step)
        elif time_unit == "weeks":
            next_timestamp = current_time + datetime.timedelta(weeks=step)
        else:
            next_timestamp = current_time + datetime.timedelta(days=step)

        next_price = model.predict(last_price, verbose=0)
        predictions.append({
            "timestamp": next_timestamp.isoformat(),
            "price": float(next_price[0][0])
        })
        last_price = next_price.reshape(1, 1, 1)

    return {
        "user_id": user_id,
        "current_data": current_data,
        "predictions": predictions
    }
