
### ReportGenerator.ipynb for Google Colab ###

# install langchain and hugginface dependencies
# !pip install -qU langchain langchain-huggingface sentence_transformers langchain-community faiss-cpu pymupdf4llm

# ------------------------------------------------------------------------------
######################## acled ########################
import os
import json
from pathlib import Path
import requests
import csv
import re

def fetch_acled_data(
    email, password, country, eventStartDate, eventEndDate,
    chosenEventType="", chosenSubEventType="", chosenMinFatalities=1,
    dataFields="event_date|disorder_type|event_type|sub_event_type|actor1|assoc_actor_1|country|location|source|notes|fatalities",
    directoryPath=Path("/content/tempdata"), apiName="ACLED"
):
    tokenURL = "https://acleddata.com/oauth/token"
    baseURL = "https://acleddata.com/api/acled/read"

    def get_access_token(username, password, token_url):
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {'username': username, 'password': password, 'grant_type': 'password', 'client_id': 'acled'}
        response = requests.post(token_url, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()['access_token']
        raise Exception(f"Failed to get access token: {response.status_code} {response.text}")

    def buildURL():
        params = [
            "_format=json",
            "country=" + requests.utils.quote(country),
            f"event_date={eventStartDate}|{eventEndDate}",
            "event_date_where=BETWEEN",
            "fields=" + dataFields
        ]
        if chosenEventType:
            params.append("event_type=" + requests.utils.quote(chosenEventType))
        if chosenSubEventType:
            params.append("sub_event_type=" + requests.utils.quote(chosenSubEventType))
        if chosenMinFatalities is not None:
            params.append("fatalities=" + str(chosenMinFatalities))
            params.append("fatalities_where=>=")
        return baseURL + "?" + "&".join(params)

    token = get_access_token(email, password, tokenURL)
    url = buildURL()

    response = requests.get(url, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", payload if isinstance(payload, list) else [])
    if chosenEventType:
        rows = [r for r in rows if r.get("event_type") == chosenEventType]
    if chosenSubEventType:
        rows = [r for r in rows if r.get("sub_event_type") == chosenSubEventType]
    def to_int(v):
        try: return int(v)
        except: return 0
    if chosenMinFatalities is not None:
        rows = [r for r in rows if to_int(r.get("fatalities", 0)) >= chosenMinFatalities]

    if not directoryPath.exists():
        directoryPath.mkdir(parents=True)

    out_path = directoryPath / f"{apiName}-{country}.csv"
    headers_all = dataFields.split("|")
    omit = {"source", "meta", "assoc_actor_1", "sub_event_type", "country"}
    headers = [h for h in headers_all if h not in omit]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for ev in rows:
            writer.writerow({h: ev.get(h, "") for h in headers})

    print(f"Retrieved {len(rows)} events. Saved to {out_path}")
    return out_path
# ------------------------------------------------------------------------------
######################## worldbank ########################
from pathlib import Path
import requests
import re
import csv
import pymupdf4llm

def fetch_worldbank_briefs(
    country, eventStartDate, docty_exact="Brief",
    disclstat_exact="Disclosed", rowsPerPage=1000, maxTotalRecords=1000,
    directoryPath=Path("/content/tempdata"), apiName="Worldbank", deleteAfterCsv=False,
):
    """
    Fetches World Bank 'Brief' documents for a given country, downloads PDFs,
    extracts text, and saves results into a CSV.
    Returns the path to the resulting CSV file.
    """

    baseURL = "https://search.worldbank.org/api/v3/wds"
    dataFields = (
        "display_title|docdt|count|docty|abstracts|pdfurl|url|repnb|projectid|theme|sectr|disclstat"
    )

    def buildURL(country, os_offset=0):
        params = [
            "format=json",
            "count_exact=" + requests.utils.quote(country),
            "fl=" + requests.utils.quote(dataFields.replace("|", ",")),
            "rows=" + str(rowsPerPage),
            "os=" + str(os_offset),
            "sort=docdt",
            "order=desc",
        ]
        if eventStartDate.strip():
            params.append("strdate=" + requests.utils.quote(eventStartDate.strip()))
        if docty_exact.strip():
            params.append("docty_exact=" + requests.utils.quote(docty_exact))
        return baseURL + "?" + "&".join(params)

    def extractDocuments(payload):
        if not isinstance(payload, dict):
            return []
        return list((payload.get("documents") or {}).values())

    def cleanupTitle(title: str) -> str:
        return re.sub(r"[^\w .()-]+", "", (title or "").strip())[:100] or "document"

    def uniquePDFNames(startingDirectory: Path, startingTitle: str) -> Path:
        p = startingDirectory / f"{startingTitle}.pdf"
        if not p.exists():
            return p
        i = 2
        while True:
            candidate = startingDirectory / f"{startingTitle} ({i}).pdf"
            if not candidate.exists():
                return candidate
            i += 1

    def safe_delete(path: Path):
        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            print(f"Failed to delete {path}: {e}")

    os_offset = 0
    allRows = []
    print(f"Fetching World Bank Briefings for {country} since {eventStartDate}")

    while len(allRows) < maxTotalRecords:
        url = buildURL(country, os_offset=os_offset)
        print(f"Requesting: {url}")
        response = requests.get(url, headers={"Accept": "application/json"}, timeout=60)

        if not response.ok:
            print(f"Request failed ({response.status_code}): {response.text[:200]}")
            break

        payload = response.json()
        docs = extractDocuments(payload)
        if not docs:
            print("No more documents found.")
            break

        allRows.extend(docs)
        if len(docs) < rowsPerPage:
            break
        os_offset += len(docs)

    rows = allRows[:maxTotalRecords]
    print(f"Total documents gathered: {len(rows)}")

    pdfDir = directoryPath / "WorldBank_pdfs"
    textDir = directoryPath / "WorldBank_extracted_text"
    pdfDir.mkdir(parents=True, exist_ok=True)
    textDir.mkdir(parents=True, exist_ok=True)
    directoryPath.mkdir(parents=True, exist_ok=True)

    combinedRows = []
    createdPdfPaths = []
    createdTxtPaths = []

    headers = {"User-Agent": "Mozilla/5.0 (compatible; WorldBankPDFCatcher/1.0)"}

    for idx, doc in enumerate(rows, start=1):
        pdf_url = (doc.get("pdfurl") or "").strip()
        if not pdf_url:
            continue

        title = cleanupTitle(doc.get("display_title") or f"document_{idx}")
        dest = uniquePDFNames(pdfDir, title)

        try:
            resp = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
            if not resp.ok:
                print(f"[{idx}] Failed to download PDF ({resp.status_code}): {pdf_url}")
                continue

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            createdPdfPaths.append(dest)

            textOutputPath = textDir / f"{dest.stem}_text.txt"
            markdownText = pymupdf4llm.to_markdown(str(dest)) or ""
            textOutputPath.write_text(markdownText, encoding="utf-8")
            createdTxtPaths.append(textOutputPath)
            combinedRows.append((title, markdownText))

        except Exception as e:
            print(f"[{idx}] Error: {e}")

    csvPath = directoryPath / f"{apiName}-{country}.csv"
    try:
        with csvPath.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pdf_title", "extracted_text"])
            writer.writerows(combinedRows)
        print(f"CSV created at: {csvPath}")
    except Exception as e:
        print(f"Failed writing CSV: {e}")

    if deleteAfterCsv:
        for path in createdTxtPaths + createdPdfPaths:
            safe_delete(path)
        for subdir in [pdfDir, textDir]:
            if subdir.exists() and not any(subdir.iterdir()):
                subdir.rmdir()

    return csvPath
# ------------------------------------------------------------------------------
######################## gdelt ########################
import json
from pathlib import Path
import requests
import urllib.parse
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import time
import hashlib
import csv

_seen_text_hashes = set()

def fetch_gdelt_data(
    termToSearch,
    sourceCountry="",
    maxNumArticles=250,
    displayType="artlist",
    sortBy="datedesc",
    startDateTime="20251001000000",
    endDateTime="20251022000000",
    wordMentionedNumOfTimes=5,
    directoryPath=Path("/content/tempdata"),
    apiName="GDELT",
):
    """
    Fetches recent news articles related to a keyword or country using the GDELT 2.0 Doc API.
    Cleans HTML, extracts readable text, and outputs results to CSV.
    """

    baseURL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def createGDELTQueryLink(search, sourceCountryCode, maxRecords, modeOfDisplay, sortBy, startTime, endTime):
        gdeltQuery = '"' + search + '" sourcelang:english'

        # repeat operator (single word only)
        firstMatch = re.search(r"[A-Za-z0-9]+", search)
        if firstMatch and wordMentionedNumOfTimes > 0:
            single_word = firstMatch.group(0).lower()
            gdeltQuery += f' repeat{wordMentionedNumOfTimes}:"{single_word}"'

        if sourceCountryCode.strip():
            gdeltQuery += " sourcecountry:" + sourceCountryCode.strip()

        params = {
            "query": gdeltQuery,
            "mode": modeOfDisplay,
            "maxrecords": str(maxRecords),
            "sort": sortBy,
            "format": "json",
            "startdatetime": startTime,
            "enddatetime": endTime,
        }
        return baseURL + "?" + urllib.parse.urlencode(params)

    def downloadHTML(url: str):
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ResearchScraper/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        try:
            r = requests.get(url, headers=headers, timeout=(10,15))
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            if "text/html" not in ctype:
                return None
            return r.content
        except requests.exceptions.RequestException:
            print("Skipping (error):", url)
            return None

    def cleanTags(soup):
        for tag in soup(["script", "style","noscript"]):
            tag.decompose()
        for br in soup.find_all(["br"]):
            br.replace_with("\n")
        return soup

    def getPageTitle(soup):
        ogt = soup.find("meta", property="og:title")
        if ogt and ogt.get("content"):
            return ogt["content"].strip()
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        return h1.get_text(strip=True) if h1 else None

    def getPublishDate(soup):
        t = soup.find("time", attrs={"datetime": True})
        if t and t.get("datetime"):
            return t["datetime"].strip()
        m = soup.find("meta", attrs={"property": "article:published_time"})
        return m.get("content").strip() if m and m.get("content") else None

    def hashText(s):
        return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

    def cleanupSpaces(s: str):
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def throwOutNavigationLines(line: str):
        txt = (line or "").strip().lower()
        if not txt:
            return True
        if len(txt) <= 30:
            if txt in {
                "all news","partners","daily snapshot","video","products and services",
                "subscription","contact us","menu","more","search",
                "history","tourism","culture","sport","our company","web design",
                "subscribers login","terms of use","technical support"
            }:
                return True
            if re.fullmatch(r"\d{1,2}:\d{2}", txt):  # times
                return True
        if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{2,4}", txt):  # dates
            return True
        return False

    def articleSelector(soup):
        selectors = [
            "article",
            "main",
            "div[role='main']",
            "div.article", "div.post", "div.content", "div.text",
            "section.article", "section.post"
        ]
        candidates = [soup.select_one(sel) for sel in selectors if soup.select_one(sel)]
        if candidates:
            return max(candidates, key=lambda n: sum(len(p.get_text(strip=True)) for p in n.find_all("p")))
        return soup.body or soup

    def extractMainText(container, minimumParagraphLength=50):
        paras = []
        for p in container.find_all("p"):
            txt = p.get_text(separator=" ", strip=True)
            if len(txt) < minimumParagraphLength:
                continue
            if len(p.find_all("a")) / max(len(txt), 1) > 0.02:
                continue
            if throwOutNavigationLines(txt):
                continue
            paras.append(txt)
        if len(paras) >= 2:
            return "\n\n".join(paras).strip()
        raw = cleanupSpaces(container.get_text(separator="\n"))
        lines = [ln for ln in raw.split("\n") if not throwOutNavigationLines(ln)]
        body = "\n".join(lines).strip()
        return body if len(body) >= 300 else ""

    # Main body
    url = createGDELTQueryLink(termToSearch, sourceCountry, maxNumArticles, displayType, sortBy, startDateTime, endDateTime)
    print("Your new API Link:", url)

    response = requests.get(url, headers={"Accept": "application/json"}, timeout=120)
    if not response.ok:
        raise Exception(f"GDELT request failed: {response.status_code} {response.text[:200]}")

    payload = response.json()
    articles = payload.get("articles", payload if isinstance(payload, list) else [])
    print(f"Received {len(articles)} articles.")

    # prepare directory and csv
    directoryPath.mkdir(parents=True, exist_ok=True)
    csv_path = directoryPath / f"{apiName}-{termToSearch}.csv"

    csv_rows = []
    accepted = 0
    for i, article in enumerate(articles, 1):
        url = article.get("url")
        if not url:
            continue
        html = downloadHTML(url)
        if not html:
            continue
        soup = cleanTags(BeautifulSoup(html, "html.parser"))
        main = articleSelector(soup)
        title = getPageTitle(soup) or "Untitled"
        pubdate = getPublishDate(soup)
        text = extractMainText(main)
        if not text:
            continue
        h = hashText(text)
        if h in _seen_text_hashes:
            continue
        _seen_text_hashes.add(h)
        accepted += 1
        csv_rows.append({
            "event_number": accepted,
            "title": title,
            "url": url,
            "publish_date": pubdate or "",
            "sourcecountry": article.get("sourcecountry", ""),
            "article_body": text.strip()
        })
        if accepted >= maxNumArticles:
            break
        time.sleep(0.5)

    # save CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "event_number", "title", "url", "publish_date", "sourcecountry", "article_body"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved {accepted} articles to {csv_path}")
    return csv_path
# ------------------------------------------------------------------------------
######################## reliefweb ########################
import os
import json
from pathlib import Path
import requests
import csv
import re
from bs4 import BeautifulSoup

def fetch_reliefweb_data(
    country,
    eventStartDate,
    eventEndDate,
    limit=50,
    maxReports=100,
    directoryPath=Path("/content/tempdata"),
    apiName="ReliefWeb",
    appname="UTD-AIReport-4485"
):
    """
    Fetches humanitarian reports from the ReliefWeb API and saves them in CSV format.
    The output CSV matches the schema used by the ACLED connector.
    """

    BASE_URL = "https://api.reliefweb.int/v2/reports"

    # Handle Russia naming difference
    country_api = "Russian Federation" if country.lower() == "russia" else country

    def clean_html(html_text):
        """Remove HTML tags and return plain text."""
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, "html.parser")
        return soup.get_text(separator="\n").strip()

    def extract_fatalities(text):
        """Extract rough fatality estimates from text if >1."""
        match = re.search(r'(\d+)\s+(?:people|killed|dead|civilians|soldiers|deceased)', text, re.IGNORECASE)
        if match and int(match.group(1)) > 1:
            return int(match.group(1))
        return ""

    def fetch_page(offset):
        """Fetch a single page of reports."""
        filters = [
            {"field": "country.name", "value": country_api},
            {"field": "date.created", "value": {
                "from": f"{eventStartDate}T00:00:00+00:00",
                "to": f"{eventEndDate}T23:59:59+00:00"
            }}
        ]
        payload = {
            "profile": "list",
            "limit": limit,
            "offset": offset,
            "sort": ["date.created:desc"],
            "fields": {
                "include": ["country", "title", "body", "date", "url", "theme", "source", "disaster", "disaster_type"]
            },
            "filter": {"operator": "AND", "conditions": filters}
        }
        try:
            r = requests.post(f"{BASE_URL}?appname={appname}", json=payload)
            r.raise_for_status()
            return r.json().get("data", [])
        except requests.exceptions.RequestException as e:
            print(f"Request failed at offset {offset}: {e}")
            return []

    all_reports = []
    offset = 0

    while True:
        reports = fetch_page(offset)
        if not reports:
            break

        for report in reports:
            fields = report.get("fields", {})
            country_name = fields.get("country", [{}])[0].get("name", "N/A")
            title = fields.get("title", "N/A")
            url = fields.get("url", "N/A")
            event_date = fields.get("date", {}).get("created", "N/A")
            theme = fields.get("theme", [{}])[0].get("name", "")
            actor = fields.get("source", [{}])[0].get("name", "")
            article_body = clean_html(fields.get("body", ""))
            if not article_body:
                continue

            fatalities = extract_fatalities(article_body)

            all_reports.append({
                "event_date": event_date,
                "disorder_type": fields.get("disaster_type", [{}])[0].get("name", ""),
                "event_type": theme,
                "actor1": actor,
                "location": country_name,
                "source": actor,
                "notes": article_body,
                "fatalities": fatalities,
                "title": title,
                "url": url
            })

            if maxReports and len(all_reports) >= maxReports:
                print(f"Reached maxReports: {maxReports}")
                break

        if maxReports and len(all_reports) >= maxReports:
            break

        offset += limit
        print(f"Fetched {len(all_reports)} reports so far...")

    # Ensure output directory exists
    if not directoryPath.exists():
        directoryPath.mkdir(parents=True)

    # Define CSV output path
    out_path = directoryPath / f"{apiName}-{country}.csv"

    # Define headers to align with ACLED format
    headers = [
        "event_date", "disorder_type", "event_type", "actor1",
        "location", "source", "notes", "fatalities", "title", "url"
    ]

    # Write to CSV
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for ev in all_reports:
            writer.writerow({h: ev.get(h, "") for h in headers})

    print(f"Retrieved {len(all_reports)} reports. Saved to {out_path}")
    return out_path
# ------------------------------------------------------------------------------\
######################## runner ########################
from google.colab import userdata, drive
import pandas as pd
from uuid import uuid4

# huggingface
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# faiss vector store
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# init embedding function
embeddings = HuggingFaceEndpointEmbeddings(
    model="google/embeddinggemma-300m",
    huggingfacehub_api_token=userdata.get('HF_TOKEN'),
)

# init index and vector
'''
acled_index = faiss.IndexFlatL2(len(embeddings.embed_query("acled")))
acled_vector_store = FAISS(
    embedding_function=embeddings,
    index=acled_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
'''
'''
worldbank_index = faiss.IndexFlatL2(len(embeddings.embed_query("worldbank")))
worldbank_vector_store = FAISS(
    embedding_function=embeddings,
    index=worldbank_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
'''
'''
gdelt_index = faiss.IndexFlatL2(len(embeddings.embed_query("gdelt")))
gdelt_vector_store = FAISS(
    embedding_function=embeddings,
    index=gdelt_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
'''
'''
reliefweb_index = faiss.IndexFlatL2(len(embeddings.embed_query("reliefweb")))
reliefweb_vector_store = FAISS(
    embedding_function=embeddings,
    index=reliefweb_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
'''
'''
# acled
acled_path = fetch_acled_data(
    email=userdata.get('acled_email'),
    password=userdata.get('acled_password'),
    country="Pakistan",
    eventStartDate="2024-07-17",
    eventEndDate="2024-10-17",
)
acled_data = pd.read_csv(acled_path)
acled_text_col = "notes"
acled_metadata_cols = ["event_date", "disorder_type", "event_type", "actor_1", "location", "fatalities"]
'''
'''
# worldbank
worldbank_path = fetch_worldbank_briefs(
    country="Pakistan",
    eventStartDate="2024-10-20",
    maxTotalRecords=200,
)
worldbank_data = pd.read_csv(worldbank_path)
worldbank_text_col = "extracted_text"
worldbank_metadata_cols = ["pdf_title"]
'''
'''
# gdelt
gdelt_path = fetch_gdelt_data(
    termToSearch="Ukraine",
    startDateTime="20251001000000",
    endDateTime="20251022000000",
    maxNumArticles=15,
)
gdelt_data = pd.read_csv(gdelt_path)
gdelt_text_col = "article_body"
gdelt_metadata_cols = ["title", "url", "publish_date", "sourcecountry"]
'''
'''
# reliefweb
reliefweb_path = fetch_reliefweb_data(
    country="Ukraine",
    eventStartDate="2025-09-01",
    eventEndDate="2025-09-30",
    limit=50,
    maxReports=20,
    directoryPath=Path("/content/tempdata")
)
reliefweb_data = pd.read_csv(reliefweb_path)
reliefweb_text_col = "notes"
reliefweb_metadata_cols = ["title", "url", "event_date", "actor1", "location", "source", "disorder_type", "event_type"]
'''
# ------------
'''
acled_documents = []
for _, row in acled_data.iterrows():
  if pd.notna(row[acled_text_col]):
    metadata = {col: str(row[col]) for col in acled_metadata_cols if col in row}
    acled_documents.append(Document(page_content=row[acled_text_col], metadata=metadata))

print(f"Loaded {len(acled_documents)} documents.")
acled_uuids = [str(uuid4()) for _ in range(len(acled_documents))]
acled_vector_store.add_documents(documents=acled_documents, ids=acled_uuids)
'''
# ------------
'''
worldbank_documents = []
for _, row in worldbank_data.iterrows():
  if pd.notna(row[worldbank_text_col]):
    metadata = {col: str(row[col]) for col in worldbank_metadata_cols if col in row}
    worldbank_documents.append(Document(page_content=row[worldbank_text_col], metadata=metadata))

print(f"Loaded {len(worldbank_documents)} documents.")
worldbank_uuids = [str(uuid4()) for _ in range(len(worldbank_documents))]
worldbank_vector_store.add_documents(documents=worldbank_documents, ids=worldbank_uuids)
'''
# ------------
'''
gdelt_documents = []
for _, row in gdelt_data.iterrows():
    if pd.notna(row[gdelt_text_col]):
        metadata = {col: str(row[col]) for col in gdelt_metadata_cols if col in row}
        gdelt_documents.append(Document(page_content=row[gdelt_text_col], metadata=metadata))

print(f"Loaded {len(gdelt_documents)} documents.")
gdelt_uuids = [str(uuid4()) for _ in range(len(gdelt_documents))]
gdelt_vector_store.add_documents(documents=gdelt_documents, ids=gdelt_uuids)
'''
# ------------
'''
reliefweb_documents = []
for _, row in reliefweb_data.iterrows():
    if pd.notna(row[reliefweb_text_col]) and str(row[reliefweb_text_col]).strip():
        metadata = {col: str(row[col]) for col in reliefweb_metadata_cols if col in row}
        reliefweb_documents.append(Document(page_content=str(row[reliefweb_text_col]), metadata=metadata))

print(f"Loaded {len(reliefweb_documents)} ReliefWeb documents.")
reliefweb_uuids = [str(uuid4()) for _ in range(len(reliefweb_documents))]
reliefweb_vector_store.add_documents(documents=reliefweb_documents, ids=reliefweb_uuids)
'''
# ------------
# query vector db
query = "Where were protests reported?"
# acled_results = acled_vector_store.similarity_search_with_score(query, k=5)
# worldbank_results = worldbank_vector_store.similarity_search_with_score(query, k=2)
# gdelt_results = gdelt_vector_store.similarity_search_with_score(query, k=5)
# reliefweb_results = reliefweb_vector_store.similarity_search_with_score(query, k=5)

# print results
'''
for doc, score in acled_results:
    print(f"Score: {score:.4f}")
    print(f"Text: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")
    print("-" * 10)
'''
'''
for doc, score in worldbank_results:
    print(f"Score: {score:.4f}")
    print(f"Text: {doc.page_content}...")
    print(f"Metadata: {doc.metadata}")
    print("-" * 10)
'''
'''
for doc, score in gdelt_results:
    print(f"Score: {score:.4f}")
    print(f"Text: {doc.page_content}...")
    print(f"Metadata: {doc.metadata}")
    print("-" * 10)
'''
'''
for doc, score in reliefweb_results:
    print(f"Score: {score:.4f}")
    print(f"Text: {doc.page_content}...")
    print(f"Metadata: {doc.metadata}")
    print("-" * 10)
'''
