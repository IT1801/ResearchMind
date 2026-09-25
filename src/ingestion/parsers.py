from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader
)
from dotenv import load_dotenv

load_dotenv()


def text_loader(file_path: str, encoding: str = "utf-8"):
    loader = TextLoader(file_path, encoding=encoding)
    return loader.load()


def web_loader(url: str):
    loader = WebBaseLoader(url)
    return loader.load()


def directory_loader(
    directory_path: str,
    glob: str = "**/*",
    loader_cls=TextLoader,
):
    loader = DirectoryLoader(
        directory_path,
        glob=glob,
        loader_cls=loader_cls,
    )
    return loader.load()


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

    