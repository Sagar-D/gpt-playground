from ollama import chat
from ollama import Client
import dotenv
import os

dotenv.load_dotenv()

def directChat(prompt):
    """Directly use the chat function from the ollama module."""
    response = chat(model=os.getenv("LLM_MODEL"), messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    return response.message.content

def chatByCreatingClient(host=os.getenv("LLM_BASE_URL"), prompt="Why is the sky blue?"):
    """Use the OllamaClient to send a chat message."""
    client = Client(
      host=host,
      headers={'x-some-header': 'some-value'}
    )
    response = client.chat(model='llama3.2', messages=[
      {
        'role': 'user',
        'content': prompt,
      },
    ])
    return response.message.content

if __name__ == "__main__":
    prompt = "Can I use OpenAI API python library to interact with Ollama?"
    try:
        response = directChat(prompt)
    except Exception as e:
        print("An error occurred in directChat:", e)
    else :
        print("Response from directChat:", response)
    
    # host = "http://localhost:11434"
    # prompt = "What is the capital of India?"
    # try:
    #     response = chatByCreatingClient(host, prompt)
    # except Exception as e:
    #     print("An error occurred in chatByCreatingClient:", e)
    # else:
    #     print("Response from chatByCreatingClient:", response)

