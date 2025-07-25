from openai import OpenAI

class OpenAIClient:
    """A simple client to interact with OpenAI's API using the OpenAI Python library."""

    def __init__(self, base_url="http://localhost:11434/v1", api_key="dummy_key"):
        """Initializes the OpenAI client with the base URL and API key."""
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, model="llama3.2", prompt="Hi there!"):
        """Send a chat message to the OpenAI model and return the response."""
        response = self.client.chat.completions.create(
            model=model, messages=[{'role': "user", "content": prompt}]
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    client = OpenAIClient()
    try:
        response = client.chat(prompt="tell me another interesting fact about elephants")
    except Exception as e:
        print("An error occurred:", e)
    else:
        print("Response from OpenAI client:", response)
