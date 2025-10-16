"""
Kafka producer example (simulates sensor data)
"""
import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer

BROKER = "localhost:9092"  # replace with your broker
TOPIC = "sensor_data"

producer = KafkaProducer(
    bootstrap_servers=[BROKER],
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def generate_event(device_id: int):
    event = {
        "id": f"device_{device_id}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "temperature": round(20 + random.random() * 10 + (device_id % 5), 3),
        "humidity": round(30 + random.random() * 20, 3)
    }
    return event

if __name__ == "__main__":
    device_count = 5
    try:
        while True:
            for d in range(device_count):
                ev = generate_event(d)
                producer.send(TOPIC, ev)
            producer.flush()
            time.sleep(1)
    except KeyboardInterrupt:
        producer.close()
