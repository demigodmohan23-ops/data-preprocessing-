"""
Kafka consumer that demonstrates rolling average and incremental training
"""
import json
from collections import deque, defaultdict
from kafka import KafkaConsumer
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

BROKER = "localhost:9092"
TOPIC = "sensor_data"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=[BROKER],
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

WINDOW_SIZE = 5
buffers = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
rolling_avgs = {}

model = SGDRegressor(max_iter=1, tol=None, learning_rate='invscaling')
scaler = StandardScaler()
model_initialized = False

try:
    for msg in consumer:
        data = msg.value
        device = data.get("id")
        temp = float(data.get("temperature"))

        b = buffers[device]
        b.append(temp)
        if len(b) > 0:
            rolling_avgs[device] = sum(b) / len(b)
        else:
            rolling_avgs[device] = None

        if len(b) == WINDOW_SIZE:
            X = np.array(b).reshape(1, -1)
            y = np.array([rolling_avgs[device]])

            if not model_initialized:
                scaler.partial_fit(X)
                Xs = scaler.transform(X)
                model.partial_fit(Xs, y)
                model_initialized = True
            else:
                Xs = scaler.transform(X)
                model.partial_fit(Xs, y)

        print(f"Device: {device} | Temp: {temp:.3f} | RollingAvg: {rolling_avgs[device]:.3f}")

except KeyboardInterrupt:
    print("Shutting down consumer")
    consumer.close()
