from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

CODE_SNIPPED = """\
    def add_item(item, items=[]):
        try:
            items.append(item)
        except:
            pass
        return items
    """


def review(llm: ChatGroq, message: list[BaseMessage]) -> str:
    return llm.invoke(message).text


def main() -> None:
    load_dotenv()
    llm = ChatGroq(model="openai/gpt-oss-120b")
    messages_with_system = [
        SystemMessage(
            "You are a strict Python code reviewer. Answer in Ukrainian, max 3 sentences."
        ),
        HumanMessage(CODE_SNIPPED),
    ]

    messages_without_system = [HumanMessage(CODE_SNIPPED)]

    print("=== with system ===")
    print(review(llm, messages_with_system))
    print("=== without system ===")
    print(review(llm, messages_without_system))


if __name__ == "__main__":
    main()
