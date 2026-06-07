import requests

#Developer's explanation:
#Due to the lack of any instructions on how we should interact with prompts beyond exposing the HTTP path
#I decided to add this simple prompt test which sends a question as payload and prints the response

#Developer's note: Make sure to do Vercel authentication if needed.

url = "https://rag-hw-individual.vercel.app/api/prompt"
payload = {
    "question": "I need actionable tips on how to prepare for a system design interview. Which articles would you recommend, and why? Suggest three articles."
}

print("Sending request to local RAG server...")
response = requests.post(url, json=payload)

if response.status_code == 200:
#    print("\n--- Success! Response received ---")
    import json
    print(json.dumps(response.json(), indent=2))
else:
#    print(f"\nFailed with status code: {response.status_code}")
    print(response.text)
