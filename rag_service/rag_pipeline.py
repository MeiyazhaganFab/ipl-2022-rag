
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from langchain.retrievers import MultiQueryRetriever
from langchain_community.chat_models import ChatOllama

from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain

import argparse
from pathlib import Path

def set_up_vector_store_retriver(args):
    """
    set up vector store as retriever  
    """
    ollama_embed = OllamaEmbeddings(model=args.embedding_model, show_progress=True)

    vector_store = FAISS.load_local(folder_path=Path(args.output_vector_store_path).resolve(),
                                index_name=args.output_vector_store_index_name,
                                embeddings=ollama_embed,
                                allow_dangerous_deserialization=True)

    vector_store_retriever = vector_store.as_retriever()

    return vector_store_retriever


def set_up_multi_query_retriever(chat_model, retriever):
    """
    set up multi query retreiver with given model and retriever
    """
    
    ollama_model = ChatOllama(model=chat_model, temperature=0.5, num_predict=300)

    multi_query_retriver = MultiQueryRetriever.from_llm(retriever=retriever, llm=ollama_model)

    return multi_query_retriver


def create_chain(chat_model):
    # chain for generation 
    ollama_model = ChatOllama(model=chat_model, temperature=0.5, num_predict=300)
    genaration_prompt = HumanMessagePromptTemplate.from_template(template="""Here are few data for the context to answer question that are asked later
{context_data}


based on the above the context answer below question:
{query}
""")
    prompt_template = ChatPromptTemplate.from_messages([genaration_prompt])

    generation_chain = prompt_template | ollama_model

    return generation_chain


def rag_query(query, mq_retriever, chain):
    """
    Retrieve, Augment and Generate for the given query and retriever
    """
    
    # Retrieve
    context = ""
    relevant_docs = mq_retriever.get_relevant_documents(query=query, k=1)
    context += ''.join(data.page_content for data in relevant_docs) 
    print("Completed Multi Query Retrieval")

    # Augment and Generation
    print("Generating ...")
    result = chain.invoke({'context_data': context,
                         'query':query})
    
    print("Completed!!")

    return result


def main(args):
    """
    main function 
    """

    # set up retriever
    if Path(args.output_vector_store_path).resolve().exists():
        retriever = set_up_vector_store_retriver(args)
    else:
        print("Input file doesn't exist")
        return 1
    

    # set multi query retriever
    multi_query_retriever = set_up_multi_query_retriever(args.chat_model, retriever)

    generation_chain = create_chain(args.chat_model)

    # calling rag
    result = rag_query(args.user_query, multi_query_retriever, generation_chain)

    print(result.content)



if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Args Parser for creating rag pipeline")
    args_parser.add_argument("--user_query", default="what is the total run scored by Jos Buttler?")
    args_parser.add_argument("--embedding_model", default="granite-embedding:30m")
    args_parser.add_argument("--output_vector_store_path", default=".\\vector_store")
    args_parser.add_argument("--output_vector_store_index_name", default="ipl_2022")
    args_parser.add_argument("--chat_model", default="gemma3:4b")

    # parsing arguments
    args = args_parser.parse_args()

    # calling main function
    main(args)







