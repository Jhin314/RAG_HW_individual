import requests

url = "http://127.0.0.1:8000/api/prompt"
payload = {
    "question": "I need actionable tips on how to prepare for a system design interview. Which articles would you recommend, and why? Suggest three articles."
}

print("Sending request to local RAG server...")
response = requests.post(url, json=payload)

if response.status_code == 200:
    print("\n--- Success! Response received ---")
    import json
    print(json.dumps(response.json(), indent=2))
else:
    print(f"\nFailed with status code: {response.status_code}")
    print(response.text)
