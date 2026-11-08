"""Exercise 1: basic ChatGroq call and response anatomy.

Findings:
- invoke() returns AIMessage, not str; text is in .content (.text always gives str)
- input_tokens grow with prompt length; output_tokens depend on generation
- gpt-oss-120b is a reasoning model: output_tokens include hidden reasoning
- the model has no internet access and hallucinates current facts -> motivation for RAG
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq


def main() -> None:
    load_dotenv()

    llm = ChatGroq(model="openai/gpt-oss-120b")
    response = llm.invoke("Який рахунок шахтар динамо сьогодні?")

    print(response.content)
    print("-----------\n")
    print(response.usage_metadata)
    print("-----------\n")
    print(response.response_metadata)


if __name__ == "__main__":
    main()
