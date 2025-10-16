"""
CDC processor: consumes CDC events from a Kafka topic and applies incremental updates
"""
import json
from kafka import KafkaConsumer
import sqlite3
from sklearn.linear_model import SGDRegressor
import numpy as np

BROKER = "localhost:9092"
CDC_TOPIC = "dbserver1.inventory.products"  # example topic from Debezium

consumer = KafkaConsumer(
    CDC_TOPIC,
    bootstrap_servers=[BROKER],
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

SQLITE_DB = "cdc_local.db"
conn = sqlite3.connect(SQLITE_DB)
cur = conn.cursor()

cur.execute('''CREATE TABLE IF NOT EXISTS products (id TEXT PRIMARY KEY, name TEXT, quantity INTEGER, price REAL)''')
conn.commit()

model = SGDRegressor(max_iter=1, tol=None)
model_initialized = False

try:
    for msg in consumer:
        envelope = msg.value
        op = envelope.get('op') if isinstance(envelope, dict) else None
        payload = envelope.get('after') if isinstance(envelope, dict) else envelope

        if op in ('c', 'r', 'u') and payload:
            rid = payload.get('id')
            name = payload.get('name')
            quantity = payload.get('quantity') or 0
            price = payload.get('price') or 0.0

            cur.execute('REPLACE INTO products (id, name, quantity, price) VALUES (?, ?, ?, ?)',
                        (rid, name, quantity, price))
            conn.commit()

            X = np.array([[quantity]])
            y = np.array([price])
            if not model_initialized:
                model.partial_fit(X, y)
                model_initialized = True
            else:
                model.partial_fit(X, y)

        elif op == 'd' and envelope.get('before'):
            rid = envelope['before'].get('id')
            cur.execute('DELETE FROM products WHERE id = ?', (rid,))
            conn.commit()

except KeyboardInterrupt:
    cur.close()
    conn.close()
    consumer.close()
