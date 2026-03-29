#------------------------------------------#
# This module defines the vector db and the retriever function
# It runs the semantic search between user query and vector db and returns k most relevant documents
#-----------------------------------------#

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# define vector db path 
CHROMA_PATH = "chroma_db"

def get_vectorstore():
    """
    define a the vector database: embedding model, path, and collection
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="author_council",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH
    )

def retrieve(query: str, domain: str = None, author: str = None, k: int = 5) -> list:  
    """
    Takes the user query, domain name & author name (if provided),
    runs a semantic search in the vector database and returns the most relevant k documents
    """

    # create a vector db variable
    vectorstore = get_vectorstore()

    filter_dict = {}
    # if the domain and author are provided, store them in a dictionary
    if domain and author:
        filter_dict = {
            "$and": [
                {"domain": {"$eq": domain}},
                {"author": {"$eq": author}}
            ]
        }
    elif domain:
        filter_dict = {"domain": {"$eq": domain}}
    elif author:
        filter_dict = {"author": {"$eq": author}}

    # run sematic search and store the most relevant k results
    results = vectorstore.similarity_search(
        query=query,                        # user query in text
        k=k,                                # how many documents to retrieve
        filter=filter_dict if filter_dict else None     # if filter found, apply it, else do it on all the chunks
    )

    return results

def pretty_print(results: list):
    for i, doc in enumerate(results, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Author : {doc.metadata.get('author')}")
        print(f"Book   : {doc.metadata.get('book')}")
        print(f"Domain : {doc.metadata.get('domain')}")
        print(f"Text   : {doc.page_content[:450]}...")

if __name__ == "__main__":
    query = "how does suffering give life meaning?"

    print(f"Query: {query}\n")
    print("=== Filtered to Carl Jung only ===")
    results = retrieve(query, author="jung", k=4)
    pretty_print(results)

    print("\n\n=== No filter — full corpus ===")
    results = retrieve(query, k=5)
    pretty_print(results)