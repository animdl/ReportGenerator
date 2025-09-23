import os
import argparse
from dotenv import load_dotenv

# langchain
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on board game rulebooks. 

IMPORTANT INSTRUCTIONS:
1. ONLY answer based on the retrieved context from the vector database
2. If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."
3. When you provide an answer, cite the specific source document (Monopoly or Ticket to Ride)
4. Be precise and accurate - don't make assumptions or add information not in the context
5. If multiple pieces of context are relevant, synthesize them clearly
6. Always indicate your confidence level in your answer

Context from documents: {context}"""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

def main():

    # accept params
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="the query text")
    parser.add_argument("--k", type=int, default=7, help="number of chunks to retrieve")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="minimum similarity score")
    args = parser.parse_args()

    query = args.query
    k = args.k
    score_threshold = args.score_threshold

    # load env vars
    load_dotenv()

    # init embedding model
    embeddings = OllamaEmbeddings(
        model = os.getenv("EMBEDDING_MODEL")
    )

    # init db
    vector_db = Chroma(
        collection_name=os.getenv("COLLECTION_NAME"),
        persist_directory=os.getenv("DATABASE_PATH"),
        embedding_function=embeddings,
    )

    # init chat model
    llm = init_chat_model(
        os.getenv("CHAT_MODEL"),
        model_provider=os.getenv("MODEL_PROVIDER"), 
        temperature=0.1
    )

    @tool
    def query_db(query: str) -> str:
        """Search the vector database for relevant information about board game rules.
        
        Args:
            query: The question or topic to search for
            
        Returns:
            str: Formatted context from relevant document chunks
        """

        if score_threshold > 0:
            results = vector_db.similarity_search_with_score(query, k=k)
            # filter by score threshold
            results = [(doc, score) for doc, score in results if score <= score_threshold]
            docs = [doc for doc, score in results]
        else:
            docs = vector_db.similarity_search(query, k=k)

        if not docs:
            return "No relevant information found in the documents."

        # format results
        context_parts = []
        for i, doc in enumerate(docs, 1):
            document_name = doc.metadata.get("document_name", "Unknown Document")
            chunk_id = doc.metadata.get("id", f"chunk_{i}")
            
            context_parts.append(f"""
--- Document Chunk {i} ---
Source: {document_name}
Chunk ID: {chunk_id}
Content: {doc.page_content.strip()}
            """)
        
        formatted_context = "\n".join(context_parts)
        return formatted_context

    @tool
    def get_document_overview(document_name: str = "") -> str:
        """Get an overview of available documents or search for content from a specific document.
        
        Args:
            document_name: Optional name of specific document (monopoly, ticket_to_ride, etc.)
            
        Returns:
            str: Overview of documents or content from specific document
        """

        if document_name.lower() in ["monopoly", "ticket"]:
            # search for content from specific document
            all_docs = vector_db.similarity_search("", k=100)
            filtered_docs = [
                doc for doc in all_docs
                if document_name.lower() in doc.metadata.get("document_name", "").lower()
            ][:10]

            if filtered_docs:
                result = f"Found {len(filtered_docs)} chunks from {document_name}:\n"
                for doc in filtered_docs:
                    preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    result += f"- {preview}\n"
                return result
            else:
                return f"No content found for the document: {document_name}"
        else:
            # get general overview
            all_docs = vector_db.similarity_search("game rules", k=20)
            doc_names = set()
            for doc in all_docs:
                doc_names.add(doc.metadata.get("document_name", "Unknown"))

            overview = f"Available documents: {', '.join(doc_names)}\nTotal chunks available: {len(all_docs)}"
            return overview

    # connect tools to agent
    tools = [query_db, get_document_overview]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate"
    )

    try:
        result = agent_executor.invoke({
            "input": query,
            "context": ""
        })

        ai_response = result["output"]
        print(ai_response)

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Falling back to direct similarity search")
        
        # Fallback: direct search without agent
        docs = vector_db.similarity_search(query, k=7)
        if docs:
            print("Direct search results:")
            for i, doc in enumerate(docs, 1):
                print(f"\n{i}. Source: {doc.metadata.get('document_name', 'Unknown')}")
                print(f"Content: {doc.page_content[:300]}...")
        else:
            print("No relevant documents found.")

if __name__ == "__main__":
    main()
