import os
from dotenv import load_dotenv
import shutil

# langchain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def main():
    # load env vars
    load_dotenv()
    # process data
    documents = load_text_docs()
    # chunking
    chunks = chunk_docs(documents)
    # write to db
    write_to_db(chunks)

def load_text_docs():
    # Load entire directory from file path
    document_loader = DirectoryLoader(
        os.getenv("SAMPLE_DATA_PATH"),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'autodetect_encoding': True},
        show_progress=True
    )
    documents = document_loader.load()
    
    # process metadata
    for doc in documents:
        # remove path and extension from filename
        filename = os.path.basename(doc.metadata["source"]).replace('.txt', '')
        doc.metadata["document_name"] = filename

        # extract country from filename
        country = None
        if '-' in filename:
            parts = filename.split('-')
            country = parts[-1].lower()

        # extract source_type from filename
        source_type = None
        if 'acled' in filename.lower():
            source_type = 'ACLED'
        elif 'reliefweb' in filename.lower():
            source_type = 'ReliefWeb'

        doc.metadata["country"] = country
        doc.metadata["source_type"] = source_type

    print(f"Loaded {len(documents)} text files")
    return documents

def chunk_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Larger chunks for event data
        chunk_overlap=100, 
        length_function=len, 
        is_separator_regex=False,
        separators=[
            "\n================================================================================\n",  # ReliefWeb separator
            "\n--------------------------------------------------------------------------------\n",  # ACLED separator
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    chunks = text_splitter.split_documents(documents)

    # remove chunks that are too small
    filtered_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 100]
    
    print(f"Created {len(filtered_chunks)} chunks from {len(documents)} documents")
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
        print(f"Added batch {i//batch_size + 1}/{(len(chunks_with_ids)-1)//batch_size + 1}")

    print(f"Successfully added {len(chunks_with_ids)} chunks to database")

def create_chunk_ids(chunks):
    """ id: source-country-index """
    chunk_counts = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        country = chunk.metadata.get("country", "unknown")
        
        # Create base id from source and country
        base_id = f"{os.path.basename(source)}-{country}"
        
        # Increment counter for this base_id
        if base_id not in chunk_counts:
            chunk_counts[base_id] = 0
        else:
            chunk_counts[base_id] += 1
        
        # Create unique chunk id
        chunk_id = f"{base_id}-{chunk_counts[base_id]}"
        
        # add metadata
        chunk.metadata["id"] = chunk_id
        chunk.metadata["preview"] = chunk.page_content[:150] + "..." if len(chunk.page_content) > 150 else chunk.page_content

    return chunks

if __name__ == "__main__":
    main()