import argparse
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import uuid

from langchain_community.embeddings import OllamaEmbeddings

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy


def load_and_split_text(file_path):
    """
    load the given text file and split it
    """
    # load the text file
    summary_data = TextLoader(file_path=Path(file_path).resolve())
    summary = summary_data.load()

    # split the text file
    char_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=400)
    docs = char_splitter.split_documents(summary)

    return docs


def configure_vector_store(args):
    """
    configure vector store based on the args
    """

    # embedding function
    ollama_embed = OllamaEmbeddings(model=args.embedding_model, show_progress=True)

    # configure vector store
    faiss_index = faiss.IndexFlat(len(ollama_embed.embed_query("Hello World!")))
    vector_store = FAISS(
        embedding_function=ollama_embed,
        index=faiss_index,
        index_to_docstore_id={},
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        docstore=InMemoryDocstore()
    )

    return vector_store


def create_vector_store(args, docs):
    """
    create a vector store using the given docs and embedding model
    """
    
    # configure the vector store
    vector_store = configure_vector_store(args)

    # random for input docs
    random_ids = [str(uuid.uuid4()) for _ in range(len(docs))]

    # adding documents to the vector store
    embedded_ids = vector_store.add_documents(documents=docs, ids=random_ids)

    # saving the vector store
    vector_store.save_local(folder_path=Path(args.output_vector_store_path).resolve(), index_name=args.output_vector_store_index_name)

    return True


def main(args):

    # load and split the summary file
    if Path(args.input_summary_file_path).resolve().exists():
        docs = load_and_split_text(args.input_summary_file_path)
    else:
        print("Input file doesn't exist")
        return 1

    # create the vector store using the given player summary
    try:
        create_vector_store(args, docs)
        print(f"created vector store and stored in {args.output_vector_store_path}")
    except Exception as e:
        print(f"Error while creating Vector Store {e}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Args Parser for creating vector store")
    args_parser.add_argument("--input_summary_file_path", default=".\..\data\IPL_2022_summary.txt")
    args_parser.add_argument("--embedding_model", default="granite-embedding:30m")
    args_parser.add_argument("--output_vector_store_path", default=".\\vector_store")
    args_parser.add_argument("--output_vector_store_index_name", default="ipl_2022")

    # parsing arguments
    args = args_parser.parse_args()

    # calling main function
    main(args)

