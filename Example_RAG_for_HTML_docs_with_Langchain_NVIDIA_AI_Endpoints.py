# https://github.com/NVIDIA/GenerativeAIExamples/blob/main/notebooks/05_RAG_for_HTML_docs_with_Langchain_NVIDIA_AI_Endpoints.ipynb

import os
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory

#from langchain.vectorstores import FAISS      # import is deprecated
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

import re
from typing import List, Union

import requests
from bs4 import BeautifulSoup

import datetime

EMBEDDING_MODEL = "NV-Embed-QA"       # "ai-embed-qa-4" is deprecated


# First stage is to load domain documentation from the web, chunkify the data, and generate embeddings using FAISS

# Helper functions for loading html files, which we'll use to generate the embeddings.

def html_document_loader(url: Union[str, bytes]) -> str:
    """
    Loads the HTML content of a document from a given URL and return it's content.

    Args:
        url: The URL of the document.

    Returns:
        The content of the document.

    Raises:
        Exception: If there is an error while making the HTTP request.

    """
    try:
        response = requests.get(url)
        html_content = response.text
    except Exception as e:
        print(f"Failed to load {url} due to exception {e}")
        return ""

    try:
        # Create a Beautiful Soup object to parse html
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style tags
        for script in soup(["script", "style"]):
            script.extract()

        # Get the plain text from the HTML document
        text = soup.get_text()

        # Remove excess whitespace and newlines
        text = re.sub("\s+", " ", text).strip()

        return text
    except Exception as e:
        print(f"Exception {e} while loading document")
        return ""



def create_embeddings(embedding_path: str = "./embed"):

    embedding_path = "./embed"
    now = datetime.datetime.now()

    print(f"{now}: Storing embeddings to {embedding_path}")

    # List of web pages containing NVIDIA Triton technical documentation
    urls = [
         "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html",
         "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html",
         "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html",
         "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_analyzer.html",
         "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html",
    ]

    documents = []
    for url in urls:
        document = html_document_loader(url)
        documents.append(document)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
    )
    texts = text_splitter.create_documents(documents)
    index_docs(url, text_splitter, texts, embedding_path)

    now = datetime.datetime.now()

    print(f"{now}: Generated embedding successfully")

# Generate embeddings using NVIDIA AI Endpoints for LangChain and save embeddings to offline vector store
# in the /embed directory for future re-use
def index_docs(url: Union[str, bytes], splitter, documents: List[str], dest_embed_dir) -> None:
    """
    Split the document into chunks and create embeddings for the document

    Args:
        url: Source url for the document.
        splitter: Splitter used to split the document
        documents: list of documents whose embeddings needs to be created
        dest_embed_dir: destination directory for embeddings

    Returns:
        None
    """
    embeddings = NVIDIAEmbeddings(model=EMBEDDING_MODEL)

    print("The de-serialization relies loading a pickle file. Pickle files can be modified to deliver")
    print("a malicious payload that results in execution of arbitrary code on your machine.")
    print("Check if `allow_dangerous_deserialization` set to `True` is reasonable")
    for document in documents:
        texts = splitter.split_text(document.page_content)

        # metadata to attach to document
        metadatas = [document.metadata]

        # create embeddings and add to vector store
        if os.path.exists(dest_embed_dir):
            update = FAISS.load_local(folder_path=dest_embed_dir, embeddings=embeddings, 
                    allow_dangerous_deserialization=True)
            update.add_texts(texts, metadatas=metadatas)
            update.save_local(folder_path=dest_embed_dir)
        else:
            docsearch = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
            docsearch.save_local(folder_path=dest_embed_dir)    



# Second stage is to load the embeddings from the vector store and build a RAG using NVIDIAEmbeddings

'''
Create the embeddings model using NVIDIA Retrieval QA Embedding endpoint. This model represents
words, phrases, or other entities as vectors of numbers and understands the relation between words
and phrases. See here for reference: https://build.nvidia.com/nvidia/embed-qa-4
'''
create_embeddings()
embedding_model = NVIDIAEmbeddings(model=EMBEDDING_MODEL)


# Load documents from vector database using FAISS
# Embed documents
embedding_path = "embed/"

print("The de-serialization relies loading a pickle file. Pickle files can be modified to deliver")
print("a malicious payload that results in execution of arbitrary code on your machine.")
print("Check if `allow_dangerous_deserialization` set to `True` is reasonable")

docsearch = FAISS.load_local(folder_path=embedding_path, embeddings=embedding_model,
                    allow_dangerous_deserialization=True)

'''
Create a ConversationalRetrievalChain chain using NeMoLLM. In this chain we demonstrate the use
of 2 LLMs: one for summarization and another for chat. This improves the overall result in more 
complicated scenarios. We'll use Llama3 70B for the first LLM and Mixtral for the Chat element 
in the chain. We add a question_generator to generate relevant query prompt. 
See here for reference: https://python.langchain.com/docs/modules/chains/popular/chat_vector_db#conversationalretrievalchain-with-streaming-to-stdout
'''

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

chat = ChatNVIDIA(model="ai-mixtral-8x7b-instruct", temperature=0.1, max_tokens=1000, top_p=1.0)

doc_chain = load_qa_chain(chat , chain_type="stuff", prompt=QA_PROMPT)

qa = ConversationalRetrievalChain(
    retriever=docsearch.as_retriever(),
    combine_docs_chain=doc_chain,
    memory=memory,
    question_generator=question_generator,
)

# ask questions
# the system remembers previous queries
query = "What is Triton?"
result = qa({"question": query})
now = datetime.datetime.now()
answer = result.get("answer")
print(f"{now}: {answer}")

# Now we demonstrate a simpler chain using a single LLM only, a chat LLM
llm = ChatNVIDIA(model="meta/llama3-70b-instruct", temperature=0.1, max_tokens=1000, top_p=1.0)

qa_prompt=QA_PROMPT

doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=QA_PROMPT)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docsearch.as_retriever(),
    chain_type="stuff",
    memory=memory,
    combine_docs_chain_kwargs={'prompt': qa_prompt},
)

# Now try asking a question about Triton with the simpler chain. Compare the answer to the result
# with previous complex chain model

query = "What is Triton?"
result = qa({"question": query})
now = datetime.datetime.now()
answer = result.get("answer")
print(f"{now}: {answer}")
