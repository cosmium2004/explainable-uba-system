import csv
import random
from datetime import datetime, timedelta

# Sample data for generation
user_ids = [f"user{str(i).zfill(3)}" for i in range(1, 21)]
action_types = ["login", "logout", "download", "upload", "delete", "modify", "view"]
resource_ids = ["fileA.txt", "fileB.csv", "report1.pdf", "image.png", "project.docx", "data.json", ""]
ip_addresses = [f"192.168.1.{i}" for i in range(1, 21)]
device_types = ["Windows", "Mac", "Linux", "Android", "iOS", ""]

def random_timestamp(start, end):
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return (start + timedelta(seconds=random_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")

def generate_rows(num_rows=50):
    now = datetime.utcnow()
    start_time = now - timedelta(days=7)
    rows = []
    for _ in range(num_rows):
        row = {
            "user_id": random.choice(user_ids),
            "timestamp": random_timestamp(start_time, now),
            "action_type": random.choice(action_types),
            "resource_id": random.choice(resource_ids),
            "ip_address": random.choice(ip_addresses),
            "device_type": random.choice(device_types)
        }
        rows.append(row)
    return rows

def write_csv(filename, rows):
    fieldnames = ["user_id", "timestamp", "action_type", "resource_id", "ip_address", "device_type"]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == "__main__":
    rows = generate_rows(50)
    write_csv("test_batch.csv", rows)
    print("Test batch dataset 'test_batch.csv' generated successfully.")