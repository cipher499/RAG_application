#------------------------------------------#
# Define paths: documents & vector database, define metadata of the documents
# Load and chunk the documents, generate embeddings, store them into a vector db
#-----------------------------------------#

import os
import re
import pypandoc
pypandoc.download_pandoc()
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredEPubLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# to access the api keys
load_dotenv()

# paths: vector database and documents
CHROMA_PATH = "chroma_db"
BOOKS_PATH = "data/books"
BOOKS_METADATA = {}

# create a dictionary to log metadata about a document
for file in os.listdir(BOOKS_PATH):
    if not os.path.isfile(os.path.join(BOOKS_PATH , file)):
        continue
    
    name, ext = os.path.splitext(file)  # ext -> extension

    # split on "-" and fetch author and book name
    if " - " in name:
        author, title = name.split(" - ", 1)
    else:
        author, title = "unknown", name

    # parse domain from parentheses e.g. "Man and His Symbols(psychology)"
    domain_match = re.search(r'\((.+?)\)', title)
    if domain_match:
        domain = domain_match.group(1).strip().lower()
        title = re.sub(r'\(.+?\)', '', title).strip()
    else:
        domain = "unknown"

    # add the book's metadata as values
    BOOKS_METADATA[file] = {
        "author": author.strip().lower().replace(" ", "_"),
        "domain": domain,
        "book": title.strip().lower().replace(" ", "_")
    }

def load_and_chunk(file_path: str, metadata: dict) -> list:
    """
    Takes the document and its metadata defined abocve as inputs
    Loads the document, splits it into chunks using a specified chunking method, 
    returns the chunks alongwith their metdata
    """
    print(f"Loading: {file_path}")

    # load the document
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".epub":
        loader = UnstructuredEPubLoader(file_path)
    else: 
        print(f"Unsupported file type: {ext}, skipping")
        return []

    documents = loader.load()

    # create chunks by splitting the document
        # Why RCTS? how else can it be done?
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)

    # update metadata for each chunk
    for chunk in chunks:
        chunk.metadata.update(metadata)

    print(f"  Created {len(chunks)} chunks")

    # return chunks with their metadata
    return chunks

def ingest_books():
    """
    Loops through all the documents in the path,
    loads, and chunks them into the vector DB
    """
    # create a vectorstore: specify embedding model, db path, and collection name
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
            collection_name="author_council",
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH
        )

    for filename, metadata in BOOKS_METADATA.items():
        print(f"\n Ingesting: {filename}")

        # create the path to access the document
        file_path = os.path.join(BOOKS_PATH, filename)

        # skip the document if it is not in the path
        if not os.path.exists(file_path):
            print(f" File not found, skipping {file_path}")
            continue

        # call the chunking function and store the chunks 
        chunks = load_and_chunk(file_path, metadata)

        # create embeddings of the chunks and store them in vector db
        # ChromaDB has a hard batch limit of 5461 — split large books into batches
        BATCH_SIZE = 500
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            vectorstore.add_documents(batch)
            print(f"  Stored batch {i // BATCH_SIZE + 1}/{-(-len(chunks) // BATCH_SIZE)} ({len(batch)} chunks)")

        print(f"  Stored {len(chunks)} chunks total")
        print(f"  Metadata: {metadata}")

# run the script
if __name__ == "__main__":
    ingest_books()

    print("\nIngestion complete.")