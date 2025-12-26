import requests
import json


# Define the local LLM endpoint (Ollama)
ollama_url = "http://localhost:11434/api/generate"
 
# Request headers
headers = {
    "Content-Type": "application/json"
}


# Define the input for the LLM
data = {
    "model": "llama3.2",  # The local model you pulled
    "prompt": f"PROMPT GOES HERE:\n\n",
    "stream": False
}
 
# Send the POST request
response = requests.post(ollama_url, headers=headers, data=json.dumps(data))

# Check if the request was successful
if response.status_code == 200:
    # Process the response
    if data["stream"]:
        # Handle streaming response
        for line in response.iter_lines():
            if line:
                body = json.loads(line)
                print(body.get("response", ""))
    else:
        # Handle non-streaming response
        response_data = response.json()
        print(response_data["response"])
else:
    # Print error message
    print("Error:", response.status_code, response.text)
