from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import random
import dotenv
import os

dotenv.load_dotenv()


class LangClient:
    """Calss to create Langchain ChatOllama client"""

    def __init__(
        self,
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL"),
        temperature="0.2",
    ):
        self.model = model
        self.base_url = base_url
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)


if __name__ == "__main__":

    client = LangClient()

    prompt_template_test_case_design = ChatPromptTemplate.from_template(
        """Design {count} most important {test_type} test cases for application - {application}.
Include description, steps and expected results.
Provide only test cases, don't add any other comments or notes in the response"""
    )

    application_list = ["youtube", "gmail", "linkedin", "cred"]
    test_types = [
        "functional",
        "UI",
        "positive",
        "negative",
        "edge-case",
        "smoke",
        "sanity",
        "end-to-end",
    ]

    for application in application_list:
        count = random.randint(5, 10)
        test_type = random.choice(test_types)
        prompt = prompt_template_test_case_design.invoke(
            {"count": count, "test_type": test_type, "application": application}
        )

        print("\n\n")
        print(
            f"Below are the Top {count} {test_type} test cases for application - {application} : "
        )
        print()
        for chunk in client.llm.stream(prompt):
            print(chunk.content, end="")
        print(f"\n\n{'--'*50}")
