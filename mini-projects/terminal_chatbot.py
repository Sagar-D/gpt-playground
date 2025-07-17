from langchain_ollama import ChatOllama


class LLMClient:

    def __init__(
        self, base_url="http://localhost:11434", model="llama3.2", temperature=0.8
    ):
        self.model = model
        self.base_url = base_url
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
        self.history = []

    def chat(self, prompt_text):

        self.history.append({"role": "user", "content": prompt_text})

        response = ""
        for chunk in self.llm.stream(self.history):
            response += chunk.content
            print(chunk.content, end="")

        self.history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    client = LLMClient()

    print("\n\nWelcome to your terminal chatbot!!")
    print("Note : To exit the chat, pass the prompt '/bye'")
    print("--" * 50)

    while True:
        print("\n\n>>> ", end="")
        prompt_text = input()

        if prompt_text.lower() == "/bye":
            break

        client.chat(prompt_text)
