import json
import requests
import dotenv
import os

dotenv.load_dotenv()

class OllamaClient:

    def __init__(self, model=os.getenv("LLM_MODEL"), base_url=os.getenv("LLM_BASE_URL")):
        self.base_url = base_url
        self.model = model

    def prompt(self, prompt, role="user"):
        """Sends a prompt to the Ollama model and returns the response."""
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "why is the sky blue?"}],
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            # Return the raw response text for further processing
            return response.text
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    client = OllamaClient()
    try:
        response = client.prompt("What is the capital of France?")
        # Ollama may return multiple JSON objects separated by newlines
        for line in response.strip().split("\n"):
            if line:
                data = json.loads(line)
                print(data["message"]["content"], end="")
        print("Response:", response.json())
    except Exception as e:
        print("An error occurred:", e)
