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
    ("system", """You are an expert analyst creating situation awareness reports based on data from ACLED (conflict and protest events) and ReliefWeb (humanitarian information).

Your task is to analyze the provided context and generate a comprehensive situation report following this structure:

### [Country] Generated Situation Awareness Report [Date Range]

## Important Ongoing Situation:
Identify and describe the most significant ongoing situation based on the data. Focus on major events, conflicts, humanitarian crises, or political developments.

## Key Recent Insights:
List 5-7 key insights from the data. Number them and cite sources (e.g., "ACLED source 1", "ReliefWeb source 2"). Focus on:
- Major political events (coups, elections, government changes)
- Conflict events (armed clashes, attacks, violence)
- Protests and civil unrest
- Humanitarian crises and needs
- Natural disasters and their impacts
- Human rights issues

## Trends:
Identify 2-4 major trends visible in the data. These should be patterns or developments over time, such as:
- Economic indicators and changes
- Escalation or de-escalation of conflicts
- Increasing humanitarian needs
- Political tensions
- Patterns in protest activity

IMPORTANT INSTRUCTIONS:
1. ONLY use information from the retrieved context
2. Cite sources appropriately (ACLED source N, ReliefWeb source N)
3. Be factual and objective - avoid speculation
4. Focus on the most significant and impactful information
5. If data is insufficient for a section, state that clearly
6. Use specific dates, numbers, and details when available
7. Maintain a professional, analytical tone

Context from documents: {context}"""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

def main():

    # accept params
    parser = argparse.ArgumentParser()
    parser.add_argument("country", type=str, help="the country to generate report for")
    parser.add_argument("--k", type=int, default=20, help="number of chunks to retrieve")
    parser.add_argument("--start-date", type=str, default="", help="start date for report (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="", help="end date for report (YYYY-MM-DD)")
    args = parser.parse_args()

    country = args.country.lower()
    k = args.k
    start_date = args.start_date
    end_date = args.end_date

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
        temperature=0.3
    )

    @tool
    def query_db(query: str) -> str:
        """Search the vector database for information about a specific country.
        
        Args:
            query: The question or topic to search for
            
        Returns:
            str: Formatted context from relevant document chunks
        """

        results = vector_db.similarity_search_with_score(query, k=k*2)

        # filter by country
        filtered_results = [
            (doc, score) for doc, score in results 
            if doc.metadata.get("country", "").lower() == country
        ][:k]

        docs = [doc for doc, score in filtered_results]

        if not docs:
            return f"No relevant information found for {country.title()}."

        # format results
        context_parts = []
        acled_count = 0
        reliefweb_count = 0

        for i, doc in enumerate(docs, 1):
            source_type = doc.metadata.get("source_type", "Unknown")
            document_name = doc.metadata.get("document_name", "Unknown Document")

            if source_type == "ACLED":
                acled_count += 1
                source_label = f"ACLED source {acled_count}"
            elif source_type == "ReliefWeb":
                reliefweb_count += 1
                source_label = f"ReliefWeb source {reliefweb_count}"
            else:
                source_label = f"Source {i}"

            context_parts.append(f"""
--- {source_label} ---
Type: {source_type}
Document: {document_name}
Content: {doc.page_content.strip()}
            """)

        formatted_context = "\n".join(context_parts)
        return formatted_context

    @tool
    def get_country_overview(country_name: str = "") -> str:
        """Get an overview of available data for a specific country or all countries.
        
        Args:
            country_name: Name of the country (optional)
            
        Returns:
            str: Overview of available data sources and document counts
        """
        if country_name:
            country_name = country_name.lower()
            # Get sample documents for this country
            all_docs = vector_db.similarity_search("", k=100)
            country_docs = [
                doc for doc in all_docs
                if doc.metadata.get("country", "").lower() == country_name
            ]
            
            if country_docs:
                acled = sum(1 for d in country_docs if d.metadata.get("source_type") == "ACLED")
                reliefweb = sum(1 for d in country_docs if d.metadata.get("source_type") == "ReliefWeb")
                
                result = f"Data available for {country_name.title()}:\n"
                result += f"- ACLED chunks: {acled}\n"
                result += f"- ReliefWeb chunks: {reliefweb}\n"
                result += f"- Total chunks: {len(country_docs)}\n"
                return result
            else:
                return f"No data found for {country_name.title()}"
        else:
            # Get overview of all countries
            all_docs = vector_db.similarity_search("", k=200)
            countries = set()
            for doc in all_docs:
                country_meta = doc.metadata.get("country", "")
                if country_meta:
                    countries.add(country_meta)
            
            overview = f"Available countries: {', '.join(sorted(countries))}\n"
            overview += f"Total documents in database: {len(all_docs)}"
            return overview

    # connect tools to agent
    tools = [query_db, get_country_overview]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        early_stopping_method="generate"
    )

    # create query
    date_range = ""
    if start_date and end_date:
        date_range = f" from {start_date} to {end_date}"
    elif start_date:
        date_range = f" from {start_date}"
    elif end_date:
        date_range = f" up to {end_date}"

    query = f"Generate a comprehensive situation awareness report for {country.title()}{date_range}. Include all major events, conflicts, protests, humanitarian situations, and trends visible in the data."

    try:
        print(f"Generating Situation Report for: {country.title()}")
        if date_range:
            print(f"Date Range: {date_range}\n")

        result = agent_executor.invoke({
            "input": query,
            "context": ""
        })

        response = result["output"]
        print(f"\nSimilarity search results for {country.title()}:\n")
        print(response)

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Falling back to direct similarity search")

        search_query = f"{country} conflict protests humanitarian situation events"
        results = vector_db.similarity_search_with_score(search_query, k=k*2)

        filtered_results = [
            (doc, score) for doc, score in results 
            if doc.metadata.get("country", "").lower() == country
        ][:k]

        if filtered_results:
            print(f"\nFallback Search Results for {country.title()}:\n")
            for i, (doc, score) in enumerate(filtered_results, 1):
                print(f"\n{i}. Source: {doc.metadata.get('source_type', 'Unknown')}")
                print(f"   Score: {score:.4f}")
                print(f"   Content: {doc.page_content[:400]}...")
        else:
            print(f"No relevant documents found for {country.title()}.")

if __name__ == "__main__":
    main()
