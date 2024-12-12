from fastapi import FastAPI
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
    
    timestamps = []
    total_prices = []
    
    for transaction in transactions:
        data = transaction.to_dict()
        timestamps.append(data.get('timestamp'))
        total_prices.append(data.get('total_price'))
    
    return timestamps, total_prices

def train_model(user_id: str, data):
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

    model_path = f'models/{user_id}_transaction_model.h5'
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
        
        train_model(user_id, data)
    
    print(f"Models retrained at {datetime.datetime.now()}")

scheduler.add_job(retrain_models_daily, 'interval', days=1, start_date=datetime.datetime.now())

@app.on_event("startup")
async def startup_event():
    print("Starting initial training...")
    retrain_models_daily() 
    scheduler.start()

@app.get("/")
async def root():
    return {"message": "Model training scheduler is running!"}

@app.get("/predict/{user_id}")
async def predict_future_steps(user_id: str):
    model_path = f'models/{user_id}_transaction_model.h5'

    timestamps, total_prices = get_user_transactions(user_id)
    if len(total_prices) == 0:
        return {
            "user_id": user_id,
            "current_data": [],
            "predictions": []
        }

    current_data = [{"timestamp": timestamp, "price": float(price)} for timestamp, price in zip(timestamps, total_prices)]

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception:
        return {
            "user_id": user_id,
            "current_data": current_data,
            "predictions": []
        }

    last_price = np.array(total_prices[-1], dtype=np.float32).reshape(1, 1, 1)
    predictions = []

    current_time = datetime.datetime.utcnow()
    for step in range(12):
        next_price = model.predict(last_price, verbose=0)
        predictions.append({
            "timestamp": (current_time + datetime.timedelta(hours=step)).isoformat(),
            "price": float(next_price[0][0])
        })
        last_price = next_price.reshape(1, 1, 1)

    return {
        "user_id": user_id,
        "current_data": current_data,
        "predictions": predictions
    }
