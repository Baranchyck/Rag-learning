from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def main() -> None:
    load_dotenv()
    llm = ChatGroq(model="openai/gpt-oss-120b")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are cinema critic"
                "Answer only from the context."
                "If the answer is not in the context, say: 'Не знаю з наданого контексту.'",
            ),
            ("human", "Context:\n{context} Question\n{question}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    context = """Золота пальмова гілка«Криміна́льне чти́во» (англ. Pulp Fiction) — культовий художній фільм 1994 року виробництва США, одна з найвідоміших робіт режисера Квентіна Тарантіно. Картина вважається важливою віхою в історії кінематографу, що дала відчутний імпульс розвитку незалежного американського кіно.

                Фільм описує два дні з життя низки американських злочинців. Найнесподіванішим чином їхні долі перетинаються, приводячи одних до загибелі, інших до втечі, а декого до переосмислення свого життя та повернення до чесної роботи. Оригінальна назва фільму відсилає до дешевих палп-журналів, які часто публікували низькопробні детективи.

                У головних ролях: Джон Траволта, Семюел Лірой Джексон, Ума Турман, Брюс Вілліс, Гарві Кейтель, Тім Рот, Аманда Пламмер та Вінг Реймс.

                Станом на 1 березня 2024 року фільм займає 8-у позицію у списку 250 найкращих фільмів за версією IMDb (-/10)."""

    questions = ["Imdb rating?", "Director?", "Genre?"]

    inputs = [{"context": context, "question": q} for q in questions]
    answers = chain.batch(inputs, config={"max_concurrency": 2})

    for i, (question, answer) in enumerate(zip(questions, answers), start=1):
        print(f"=== {i}. {question} ===")
        print(answer)
        print()


if __name__ == "__main__":
    main()
