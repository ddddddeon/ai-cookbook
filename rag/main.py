import click
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
import os
import random
import re
import requests
from bs4 import BeautifulSoup
from openai import OpenAI


class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def __call__(self, texts):
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [e.embedding for e in response.data]


def save_embeddings(collection, text):
    collection.add(ids=[str(i) for i in range(len(text))], documents=text)


def get_context(collection, query):
    results = collection.query(
        query_texts=[query],
        n_results=3,
    )

    return "\n".join([doc for document in results.get("documents") for doc in document])


def prompt(query, context):
    deepseek = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
    )
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Please use the provided context to answer questions.",
        },
        {"role": "user", "content": f"Context:\n{context}\nQuestion: {query}"},
    ]

    response = deepseek.chat.completions.create(
        model="deepseek-chat", messages=messages, temperature=0.4, stream=True
    )

    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = BeautifulSoup(response.content, "html.parser").get_text()
        text = re.sub(r"\s+", " ", text).strip()

        chunks = []
        chunk_size = 1000
        overlap = 100
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap

        return chunks
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None


@click.command()
@click.argument("url")
def main(url):
    collection = chromadb.Client().get_or_create_collection(
        name="documents", embedding_function=OpenAIEmbeddingFunction()
    )
    text = fetch_text_from_url(url)
    save_embeddings(collection, text)

    if text:
        query = input("> ")
        context = get_context(collection, query) or ""
        prompt(query, context)


if __name__ == "__main__":
    main()
