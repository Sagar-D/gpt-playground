from langchain_ollama import ChatOllama


class LangClient:
    """Class to create a ChatOllama client object"""

    def __init__(
        self, base_url="http://localhost:11434", model="llama3.2", temperature=0
    ):
        """Initiate the ChatOllama client with host, model and temparature and create a client"""
        self.base_url = base_url
        self.model = model
        self.llm = ChatOllama(
            base_url=base_url, model=model, temperature=temperature
        )

    def invoke_prompt(self, prompt):
        """Method to invoke a single prompt and get back complete response"""
        return self.llm.invoke(prompt)


if __name__ == "__main__":

    prompt = "How many breeds of dogs are found across the world?"

    client = LangClient()
    response = client.invoke_prompt(prompt)

    print(response.content)
