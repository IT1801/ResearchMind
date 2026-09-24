import os

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    llm = ChatGoogleGenerativeAI(
        model = os.environ["GEMINI_MODEL"],
        temperature = 0
    )
    response = llm.invoke("Say Setup is complete")
    print(response.content)


if __name__ == "__main__":
    main()
