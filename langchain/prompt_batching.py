from langchain_ollama import ChatOllama


class LangClient:
    """Class to create ChatOllama client for specific host and model"""

    def __init__(
        self, base_url="http://localhost:11434", model="llama3.2", temperature=0
    ):
        self.base_url = base_url
        self.model = model
        self.client = ChatOllama(
            base_url=base_url, model=model, temperature=temperature
        )

    def batch_prompts(self, prompt_list):
        """Method to invoke a batch of prompts and return the batch response"""
        return self.client.batch(prompt_list)


if __name__ == "__main__":

    prompt_list = [
        "What is the capital of India?",
        "What is the national sport of India?",
        "What is the national animal of India?",
        "What are the major achievements of ISRO?",
        "Who is the 1st President of India?",
        "Which are the major cities of India?",
        "Who are the most famous cricketers from India?",
    ]

    client = LangClient()
    response_list = client.batch_prompts(prompt_list)

    for index, prompt, response in zip(
        range(1, len(prompt_list) + 1), prompt_list, response_list
    ):
        print(f"Q{index}. {prompt.capitalize()}")
        print(response.content)
        print("\n", "--" * 50, "\n\n", sep="")
