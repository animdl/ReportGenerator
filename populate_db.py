import os
from dotenv import load_dotenv
import shutil

# langchain imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def main():
    # load env vars
    load_dotenv()
    # process data
    documents = load_pdf_docs()
    # chunking
    chunks = chunk_docs(documents)
    # write to db
    write_to_db(chunks)

def load_pdf_docs():
    document_loader = PyPDFDirectoryLoader(os.getenv("SAMPLE_DATA_PATH"))
    documents = document_loader.load()
    
    # remove path and extension from filename
    for doc in documents:
        filename = os.path.basename(doc.metadata["source"]).replace('.pdf', '')
        doc.metadata["document_name"] = filename

    return documents

def chunk_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100, 
        length_function=len, 
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # remove chunks that are smaller than 50 chars
    filtered_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 50]
    return filtered_chunks

def write_to_db(chunks: list[Document]):
    # delete db if db exists
    if os.path.exists(os.getenv("DATABASE_PATH")):
        shutil.rmtree(os.getenv("DATABASE_PATH"))
    
    # init embedding model
    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL")
    )

    # init db
    vector_db = Chroma(
        collection_name=os.getenv("COLLECTION_NAME"),
        persist_directory=os.getenv("DATABASE_PATH"),
        embedding_function=embeddings,
    )

    # calculate chunk ids
    chunks_with_ids = create_chunk_ids(chunks)

    # set uuids
    uuids = [chunk.metadata["id"] for chunk in chunks_with_ids]

    # add to db in batches
    # done to avoid mem issues
    batch_size = 100
    for i in range(0, len(chunks_with_ids), batch_size):
        batch_chunks = chunks_with_ids[i:i+batch_size]
        batch_uuids = uuids[i:i+batch_size]
        vector_db.add_documents(batch_chunks, ids=batch_uuids)

def create_chunk_ids(chunks):
    # Page Source - Page Number - Chunk Index
    last_page_id = 0
    current_chunk_index = 0

    for chunk in chunks:
        # file path
        source = chunk.metadata.get("source", "unknown") 
        # page number of the doc
        page = chunk.metadata.get("page", 0)

        # set metadata to Page Source - Page Number
        current_page_id = f"{source}-{page}"

        # increment chunk index
        # chunk index is reset to 0 when page id changes
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        # set metadata to Page Source - Page Number - Chunk Index
        chunk_id = f"{current_page_id}-{current_chunk_index}"
        # update page id
        last_page_id = current_page_id
        
        # add extra metadata
        chunk.metadata["id"] = chunk_id

        # debug
        # chunk preview
        chunk.metadata["preview"] = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content

    return chunks

if __name__ == "__main__":
    main()