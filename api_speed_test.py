import time
import requests
import statistics

# Number of distinct queries to send
n_requests = 25
url = "http://localhost:8000/recommend"
top_k = 5

# 25 distinct Royal Road novel types/genres to test
queries = [
    "fantasy",
    "litRPG",
    "science fiction",
    "action adventure",
    "romance",
    "isekai",
    "cultivation",
    "supernatural",
    "horror",
    "historical fiction",
    "mystery",
    "slice of life",
    "comedy",
    "drama",
    "post-apocalyptic",
    "steampunk",
    "cyberpunk",
    "superhero",
    "western",
    "sports",
    "military",
    "survival",
    "travel",
    "political intrigue",
    "revenge",
]

latencies = []

print(f"Sending {n_requests} distinct requests to {url} (top_k={top_k})\n")
for i, query in enumerate(queries, start=1):
    payload = {"query": query, "top_k": top_k}
    start_time = time.time()
    try:
        resp = requests.post(url, json=payload)
        elapsed = time.time() - start_time
        if resp.status_code != 200:
            print(f"Request {i:>2} (query='{query}') failed with status {resp.status_code}")
        else:
            latencies.append(elapsed)
            print(f"{i:>2}: query='{query}' -> {elapsed*1000:>8.2f} ms")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Request {i:>2} (query='{query}') exception: {e} (after {elapsed*1000:.2f} ms)")

# Summary statistics
if latencies:
    total_time = sum(latencies)
    avg_latency = statistics.mean(latencies)
    med_latency = statistics.median(latencies)
    p90_latency = statistics.quantiles(latencies, n=100)[89]
    p95_latency = statistics.quantiles(latencies, n=100)[94]

    print("\nSummary:")
    print(f"Total requests sent: {n_requests}")
    print(f"Successful responses: {len(latencies)}")
    print(f"Total time: {total_time:.4f} s")
    print(f"Average latency: {avg_latency*1000:.2f} ms")
    print(f"Median latency: {med_latency*1000:.2f} ms")
    print(f"90th percentile latency: {p90_latency*1000:.2f} ms")
    print(f"95th percentile latency: {p95_latency*1000:.2f} ms")
else:
    print("No successful requests to summarize.")
