from __future__ import annotations

import os
import re
import sys
import time
import html
import requests
import pandas as pd
import concurrent.futures
import xml.etree.ElementTree as ET
from threading import Lock
from typing import Any, Dict, List, Optional, Set
from difflib import SequenceMatcher
from urllib.parse import quote

# ----------------------------
# Config: APIs & Rate Limiting
# ----------------------------
APIS = {
    "openalex": {
        "base": "https://api.openalex.org/works",
        "rate_limit": 1.0,  
    },
    "semantic_scholar": {
        "base": "https://api.semanticscholar.org/graph/v1",
        "rate_limit": 1.0, # Slower without key
    },
    "pubmed": {
        "base": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        "rate_limit": 0.34, # 3 req/sec
    },
    "crossref": {
        "base": "https://api.crossref.org/works",
        "rate_limit": 0.5, 
    },
    "arxiv": {
        "base": "http://export.arxiv.org/api/query",
        "rate_limit": 1.0, 
    },
    "core": {
        "base": "https://api.core.ac.uk/v3",
        "rate_limit": 0.5,
    },
    "biorxiv": {
        "base": "https://api.biorxiv.org/details/biorxiv",
        "rate_limit": 1.0,
    }
}

# Load API Keys
API_KEYS = {
    "OPENALEX_EMAIL": os.getenv("OPENALEX_EMAIL", "example@example.com"),
    "S2_API_KEY": os.getenv("S2_API_KEY", ""),
    "CORE_API_KEY": os.getenv("CORE_API_KEY", ""),
}

# User Agent
USER_AGENT = f"AVALANCHE/1.0 (mailto:{API_KEYS['OPENALEX_EMAIL']})"

# Tunables
MAX_WORKERS = 3       
MIN_SCORE_THRESHOLD = 15  # Out of 100
MAX_TOPIC_RESULTS = 20    # Per source
FUZZY_DEDUP_THRESHOLD = 0.90 

# Rate Limiter
class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.last_time = 0.0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_time = time.time()

rate_limiters = {api: RateLimiter(config["rate_limit"]) for api, config in APIS.items()}
print_lock = Lock()

# ----------------------------
# Core Logic: Helpers
# ----------------------------
def safe_print(msg):
    with print_lock:
        print(msg)

def clean_text(s: Any) -> str:
    if not s: return ""
    s = str(s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_doi(doi: str) -> str:
    if not doi: return ""
    # Remove standard prefixes
    d = doi.lower().replace("https://doi.org/", "").replace("doi:", "").replace("http://dx.doi.org/", "").strip()
    return d

def fuzzy_match(s1: str, s2: str) -> float:
    return SequenceMatcher(None, s1, s2).ratio()

def validate_doi(doi: str) -> bool:
    """
    Validates DOI format. A valid DOI contains:
    - Prefix: 10.XXXX (where XXXX is 4+ digits)
    - Suffix: Any characters after /
    Examples: 10.1038/nature12373, 10.1109/5.771073
    """
    if not doi:
        return False

    # Remove common prefixes if present
    clean_doi = doi.lower()
    clean_doi = clean_doi.replace("https://doi.org/", "")
    clean_doi = clean_doi.replace("http://doi.org/", "")
    clean_doi = clean_doi.replace("doi:", "")
    clean_doi = clean_doi.strip()

    # Basic DOI pattern: 10.XXXX/suffix
    doi_pattern = r"^10\.\d{4,}/\S+"
    return bool(re.match(doi_pattern, clean_doi))

# ----------------------------
# Smart Scoring & Exclusion
# ----------------------------
def calculate_smart_score(title: str, abstract: str, keywords: List[str]) -> float:
    """
    Weighted scoring: 
    - Title match: 10 pts
    - Abstract match: 3 pts
    - Normalized by keyword coverage (must match at least some keywords)
    """
    if not keywords: return 100.0
    
    title_lower = title.lower()
    abs_lower = abstract.lower()
    
    score = 0
    hits = 0
    
    for kw in keywords:
        k = kw.lower()
        if k in title_lower:
            score += 10
            hits += 1
        elif k in abs_lower:
            score += 3
            hits += 1
            
    # Boost if high percentage of keywords are present
    coverage = hits / len(keywords)
    final_score = (score * 0.5) + (coverage * 50)
    
    return min(100.0, final_score)

def is_excluded(title: str, abstract: str, excluded_terms: List[str]) -> bool:
    if not excluded_terms: return False
    text = (title + " " + abstract).lower()
    for term in excluded_terms:
        if term.lower() in text:
            return True
    return False

# ----------------------------
# Network Layer
# ----------------------------
def fetch_url(url: str, api_name: str, params: dict = None, headers: dict = None, is_xml=False) -> Any:
    max_retries = 2
    retry_count = 0
    
    while retry_count <= max_retries:
        limiter = rate_limiters.get(api_name)
        if limiter:
            limiter.wait()

        h = {"User-Agent": USER_AGENT}
        if headers:
            h.update(headers)

        if api_name == "semantic_scholar" and API_KEYS["S2_API_KEY"]:
            h["x-api-key"] = API_KEYS["S2_API_KEY"]
        if api_name == "core" and API_KEYS["CORE_API_KEY"]:
            h["Authorization"] = f"Bearer {API_KEYS['CORE_API_KEY']}"

        try:
            r = requests.get(url, params=params, headers=h, timeout=30)

            if r.status_code == 200:
                return r.text if is_xml else r.json()

            if r.status_code == 429:
                safe_print(f"  [⚠️ 429] {api_name} rate-limited. Sleeping 2s...")
                time.sleep(2)
                return None

            # Helpful debugging (don't spam full body)
            safe_print(f"  [⚠️ {api_name}] HTTP {r.status_code} for {r.url}")

        except requests.exceptions.Timeout:
            retry_count += 1
            if retry_count <= max_retries:
                safe_print(f"  [⚠️ {api_name}] Timeout. Retrying... ({retry_count}/{max_retries})")
                time.sleep(1)
                continue
            else:
                safe_print(f"  [⚠️ {api_name}] Request timeout (max retries exceeded)")
                return None
        except Exception as e:
            safe_print(f"  [⚠️ {api_name}] Request error: {e}")
            return None

    return None

# ----------------------------
# Unified Record Factory
# ----------------------------
def make_record(source, title, year, doi, cited, abstract, venue, url, relation, keywords, excluded):
    title = clean_text(title)
    abstract = clean_text(abstract)
    
    if is_excluded(title, abstract, excluded):
        return None

    score = calculate_smart_score(title, abstract, keywords)
    
    # Filter low quality (except Seed)
    if relation != "SEED" and score < MIN_SCORE_THRESHOLD:
        return None

    return {
        "Title": title,
        "Year": int(year) if year else 0,
        "DOI": normalize_doi(doi),
        "Cited_By": int(cited) if cited else 0,
        "Abstract": abstract[:3000], # Truncate massive abstracts
        "Relevance": round(score, 1),
        "Venue": clean_text(venue),
        "URL": url,
        "Source": source,
        "Relation": relation
    }

# ----------------------------
# API Strategies (The "12 APIs" Implementation)
# ----------------------------

# 1. Semantic Scholar
def s2_search(query: str, kw, ex) -> List[dict]:
    safe_print(f"... [S2] Searching '{query}'")
    url = f"{APIS['semantic_scholar']['base']}/paper/search"
    params = {"query": query, "limit": MAX_TOPIC_RESULTS, "fields": "title,year,externalIds,citationCount,abstract,venue,url"}
    data = fetch_url(url, "semantic_scholar", params)
    
    results = []
    if data and "data" in data:
        for item in data["data"]:
            doi = (item.get("externalIds") or {}).get("DOI")
            results.append(make_record(
                "SemanticScholar", item.get("title"), item.get("year"), doi, 
                item.get("citationCount"), item.get("abstract"), item.get("venue"), 
                item.get("url"), "Topic Search", kw, ex
            ))
    return [r for r in results if r]

# 2. Crossref
def crossref_search(query: str, kw, ex) -> List[dict]:
    safe_print(f"... [Crossref] Searching '{query}'")
    params = {"query": query, "rows": MAX_TOPIC_RESULTS}
    data = fetch_url(APIS['crossref']['base'], "crossref", params)
    
    results = []
    if data and "message" in data:
        for item in data["message"].get("items", []):
            results.append(make_record(
                "Crossref", item.get("title", [""])[0], 
                item.get("published-print", {}).get("date-parts", [[0]])[0][0],
                item.get("DOI"), item.get("is-referenced-by-count"), 
                "", # Crossref rarely has abstracts
                item.get("container-title", [""])[0], 
                item.get("URL"), "Topic Search", kw, ex
            ))
    return [r for r in results if r]

# 3. CORE (UK Open Access)
def core_search(query: str, kw, ex) -> List[dict]:
    if not API_KEYS["CORE_API_KEY"]: return []
    safe_print(f"... [CORE] Searching '{query}'")
    
    # POST request is preferred for CORE search
    url = f"{APIS['core']['base']}/search/works"
    headers = {"Authorization": f"Bearer {API_KEYS['CORE_API_KEY']}", "Content-Type": "application/json"}
    payload = {"q": query, "limit": MAX_TOPIC_RESULTS}
    
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            results = []
            for item in data.get("results", []):
                results.append(make_record(
                    "CORE", item.get("title"), item.get("yearPublished"), item.get("doi"),
                    0, item.get("abstract"), "",
                    item.get("downloadUrl"), "Topic Search", kw, ex
                ))
            return [r for r in results if r]
    except Exception as e:
        safe_print(f"  [⚠️ CORE] Request failed: {e}")
    return []

# 4. PubMed (Fixed)
def pubmed_search(query: str, kw, ex) -> List[dict]:
    safe_print(f"... [PubMed] Searching '{query}'")
    # Step 1: Search IDs
    es_params = {"db": "pubmed", "term": f"{query}", "retmode": "json", "retmax": MAX_TOPIC_RESULTS}
    d1 = fetch_url(f"{APIS['pubmed']['base']}/esearch.fcgi", "pubmed", es_params)
    ids = (d1.get("esearchresult", {}).get("idlist", [])) if d1 else []
    
    if not ids: return []
    
    # Step 2: Fetch Details
    sum_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
    d2 = fetch_url(f"{APIS['pubmed']['base']}/esummary.fcgi", "pubmed", sum_params)
    
    results = []
    items = d2.get("result", {}) if d2 else {}
    for uid in ids:
        if uid in items:
            i = items[uid]
            doi = next((x["value"] for x in i.get("articleids", []) if x["idtype"]=="doi"), "")
            # Year hack
            y = re.search(r"\d{4}", i.get("pubdate",""))
            results.append(make_record(
                "PubMed", i.get("title"), y.group(0) if y else 0, doi,
                0, "", i.get("fulljournalname"), 
                f"https://pubmed.ncbi.nlm.nih.gov/{uid}", "Topic Search", kw, ex
            ))
    return [r for r in results if r]

# 5. ArXiv (Fixed)
def arxiv_search(query: str, kw, ex) -> List[dict]:
    safe_print(f"... [ArXiv] Searching '{query}'")
    params = {"search_query": f"all:{query}", "max_results": MAX_TOPIC_RESULTS}
    xml_data = fetch_url(APIS['arxiv']['base'], "arxiv", params, is_xml=True)
    if not xml_data: return []

    results = []
    try:
        root = ET.fromstring(xml_data)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("a:entry", ns):
            doi = entry.findtext("a:doi", "", ns)
            results.append(make_record(
                "ArXiv", entry.findtext("a:title", "", ns), 
                entry.findtext("a:published", "", ns)[:4], doi,
                0, entry.findtext("a:summary", "", ns), "ArXiv",
                entry.findtext("a:id", "", ns), "Topic Search", kw, ex
            ))
    except Exception as e:
        safe_print(f"  [⚠️ ArXiv] XML parsing failed: {e}")
    return [r for r in results if r]

# 6. OpenAlex Helper (Invert Index)
def openalex_invert(inv_idx):
    if not inv_idx: return ""
    tokens = {}
    for word, positions in inv_idx.items():
        for p in positions: tokens[p] = word
    if not tokens: return ""
    return " ".join([tokens[i] for i in sorted(tokens.keys())])

# 7. Robust OpenAlex Seed Lookup
def openalex_get_work_by_doi(seed_doi: str) -> Optional[dict]:
    """
    Robust lookup for a work in OpenAlex using DOI.
    Tries:
      1) direct external-id (raw)
      2) external-id with ':' encoded (safe='/')
      3) external-id fully encoded (safe='')
      4) filter fallback (doi:<canonical>)
    Returns a Work object dict or None.
    """
    d = normalize_doi(seed_doi)
    if not d:
        return None

    ext = f"https://doi.org/{d}"

    # 1) raw (sometimes works)
    url1 = f"{APIS['openalex']['base']}/{ext}"
    data = fetch_url(url1, "openalex")
    if isinstance(data, dict) and data.get("id"):
        return data

    # 2) encode ':' (matches what many clients do: https%3A//doi.org/...)
    url2 = f"{APIS['openalex']['base']}/{quote(ext, safe='/')}"
    data = fetch_url(url2, "openalex")
    if isinstance(data, dict) and data.get("id"):
        return data

    # 3) fully encoded (https%3A%2F%2Fdoi.org%2F...)
    url3 = f"{APIS['openalex']['base']}/{quote(ext, safe='')}"
    data = fetch_url(url3, "openalex")
    if isinstance(data, dict) and data.get("id"):
        return data

    # 4) filter fallback (OpenAlex stores doi as canonical external id)
    params = {"filter": f"doi:{ext}", "per-page": 1}
    data = fetch_url(APIS["openalex"]["base"], "openalex", params=params)
    if isinstance(data, dict) and data.get("results"):
        return data["results"][0]

    return None
# ----------------------------
# OpenAlex Batch Fetch + Recursion Payload
# ----------------------------
def chunked(lst: List[Any], n: int) -> List[List[Any]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def openalex_fetch_works_by_ids(work_ids: List[str]) -> List[dict]:
    """
    Fetch up to ~50 OpenAlex works in one request using filter=openalex_id:
    Example filter: openalex_id:W123|W456|W789
    """
    if not work_ids:
        return []

    # OpenAlex IDs are full URLs like "https://openalex.org/W123..."
    # The filter expects those IDs exactly.
    filt = "openalex_id:" + "|".join(work_ids)
    url = APIS["openalex"]["base"]
    params = {"filter": filt, "per-page": len(work_ids)}  # OpenAlex uses per-page
    data = fetch_url(url, "openalex", params=params)

    if not data or "results" not in data:
        return []
    return data["results"]

def fetch_oa_chunk_recursive(work_ids: List[str], depth_level: int, kw: List[str], ex: List[str]) -> List[dict]:
    """
    Fetch a chunk of OpenAlex work IDs and return standardized records.
    Adds hidden fields:
      - _depth: current depth
      - _refs: referenced_works list (for recursion)
    """
    works = openalex_fetch_works_by_ids(work_ids)
    out = []

    for w in works:
        try:
            oa_abs = openalex_invert(w.get("abstract_inverted_index"))
            rec = make_record(
                "OpenAlex",
                w.get("display_name"),
                w.get("publication_year"),
                w.get("doi"),
                w.get("cited_by_count"),
                oa_abs,
                ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
                w.get("id"),
                "Graph Reference",
                kw,
                ex,
            )
            if rec:
                rec["_depth"] = depth_level
                rec["_refs"] = w.get("referenced_works", []) or []
                out.append(rec)
        except Exception as e:
            safe_print(f"  [⚠️ OpenAlex] Failed to process work: {e}")
            continue

    return out

def openalex_fetch_citations(seed_openalex_id: str, kw: List[str], ex: List[str], per_page: int = 50) -> List[dict]:
    """
    Fetch works that cite the seed using OpenAlex filter=cites:<seed_id>.
    Note: This is only 1 page by default; extend if you want pagination later.
    """
    url = APIS["openalex"]["base"]
    params = {"filter": f"cites:{seed_openalex_id}", "per-page": per_page}
    data = fetch_url(url, "openalex", params=params)

    results = []
    for w in (data or {}).get("results", []):
        oa_abs = openalex_invert(w.get("abstract_inverted_index"))
        rec = make_record(
            "OpenAlex",
            w.get("display_name"),
            w.get("publication_year"),
            w.get("doi"),
            w.get("cited_by_count"),
            oa_abs,
            ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            w.get("id"),
            "Graph Citation",
            kw,
            ex,
        )
        if rec:
            results.append(rec)
    return results


# ----------------------------
# Main Workflow
# ----------------------------
def main(seed_doi: str):
    print(f"--- AVALANCHE | DOI: {seed_doi} ---")

    # Validate DOI format
    if not validate_doi(seed_doi):
        print(f"\n[ERROR] Invalid DOI format: '{seed_doi}'")
        print("Expected format: 10.XXXX/suffix (e.g., 10.1038/s41586-020-2649-2)")
        print("You can also use full URL: https://doi.org/10.XXXX/suffix")
        sys.exit(1)

    # Check API key configuration
    print("\n--- API Configuration ---")
    if API_KEYS["OPENALEX_EMAIL"] == "example@example.com":
        print("  [⚠️] OpenAlex: Using default email (slow rate limits)")
        print("      → Add OPENALEX_EMAIL to .env for faster access")
    else:
        print(f"  [✓] OpenAlex: {API_KEYS['OPENALEX_EMAIL']}")

    if API_KEYS["S2_API_KEY"]:
        print("  [✓] Semantic Scholar: API key configured")
    else:
        print("  [⚠️] Semantic Scholar: No API key (1 req/sec limit)")
        print("      → Add S2_API_KEY to .env for 10x speed boost")

    if API_KEYS["CORE_API_KEY"]:
        print("  [✓] CORE: API key configured")
    else:
        print("  [⚠️] CORE: No API key (CORE search disabled)")

    print()
    kw_input = input(" > Keywords (comma sep): ").strip()
    keywords = [k.strip() for k in kw_input.split(",") if k.strip()]
    
    ex_input = input(" > Exclude terms (comma sep): ").strip()
    excluded = [k.strip() for k in ex_input.split(",") if k.strip()]
    
    print("\n > Citation Depth (how deep to follow references):")
    print("   1 = seed + direct references/citations")
    print("   2 = seed + refs/citations + their refs/citations (default)")
    print("   3-6 = go 3-6 levels deep (slower, more comprehensive)")
    depth_input = input(" > Depth (1-6, default: 2): ").strip()
    depth = int(depth_input) if depth_input.isdigit() and 1 <= int(depth_input) <= 6 else 2
    print(f"   → Using depth: {depth}")

    print("\n > Search Mode Selection:")
    print("   1 = Classical Snowball (Graph Only: Seed + Citations/References)")
    print("   2 = Dual Process (Graph + Multi-Source Keyword Search) [Default]")
    mode_input = input(" > Select Mode (1 or 2): ").strip()
    search_mode = 1 if mode_input == "1" else 2
    print(f"   → Using Mode: {'Classical Snowball' if search_mode == 1 else 'Dual Process'}")

    # --- Phase 1: Federated Graph Search (Recursive) ---
    print("\n--- Phase 1: Federated Graph Search ---")

    # Check OpenAlex for Seed using robust lookup
    seed_data = openalex_get_work_by_doi(seed_doi)
    
    all_records = []
    seen_ids: Set[str] = set()     # OpenAlex work IDs seen (cycle prevention)
    seen_dois: Set[str] = set()    # For dedup later
    seen_ids_lock = Lock()         # Thread-safe access to seen_ids
    records_lock = Lock()          # Thread-safe access to all_records

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures: Set[concurrent.futures.Future] = set()

        # --- A) Process Seed ---
        if seed_data:
            oa_abs = openalex_invert(seed_data.get("abstract_inverted_index"))
            seed_rec = make_record(
                "OpenAlex",
                seed_data.get("display_name"),
                seed_data.get("publication_year"),
                seed_data.get("doi"),
                seed_data.get("cited_by_count"),
                oa_abs,
                ((seed_data.get("primary_location") or {}).get("source") or {}).get("display_name"),
                seed_data.get("id"),
                "SEED",
                keywords,
                [],   # do NOT exclude seed
            )

            if seed_rec:
                print(f"   [Seed Found] {seed_rec['Title'][:60]}...")
                with records_lock:
                    all_records.append(seed_rec)
                with seen_ids_lock:
                    seen_ids.add(seed_data["id"])
                    if seed_rec["DOI"]:
                        seen_dois.add(seed_rec["DOI"])

                    # 1) Seed references (backward)
                    refs = seed_data.get("referenced_works", []) or []
                    # mark as seen immediately (avoids duplicate queueing)
                    new_refs = [r for r in refs if r not in seen_ids]
                    seen_ids.update(new_refs)

                for batch in chunked(new_refs, 50):
                    futures.add(executor.submit(fetch_oa_chunk_recursive, batch, 1, keywords, excluded))

                # 2) Seed citations (forward) - optional but your code hinted at it
                futures.add(executor.submit(openalex_fetch_citations, seed_data["id"], keywords, excluded))

        else:
            print("   [!] Seed not found in OpenAlex (skipping graph expansion).")

        # --- B) Recursive Graph Loop (references only) ---
        while futures:
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

            for f in done:
                try:
                    res = f.result()
                except Exception as e:
                    safe_print(f"   [Error] Worker failed: {e}")
                    continue

                # Some tasks return list[dict] (records)
                if isinstance(res, list):
                    for rec in res:
                        if not rec:
                            continue

                        # If it's a graph-recursion record, it may have hidden fields
                        rec_depth = rec.get("_depth", None)
                        rec_refs = rec.get("_refs", None)

                        # Strip hidden fields before storing (we store after reading them)
                        if "_depth" in rec: rec.pop("_depth", None)
                        if "_refs" in rec: rec.pop("_refs", None)

                        with records_lock:
                            all_records.append(rec)
                            if rec.get("DOI"):
                                seen_dois.add(rec["DOI"])

                        # Recurse if applicable
                        if rec_depth is not None and rec_refs is not None and rec_depth < depth:
                            next_depth = rec_depth + 1
                            with seen_ids_lock:
                                unseen_refs = [r for r in rec_refs if r not in seen_ids]
                                seen_ids.update(unseen_refs)

                            for batch in chunked(unseen_refs, 50):
                                safe_print(f"   [+] Snowballing: {len(batch)} refs from '{rec['Title'][:20]}...' (Depth {next_depth})")
                                futures.add(executor.submit(fetch_oa_chunk_recursive, batch, next_depth, keywords, excluded))

        print(f"   [Graph Phase Complete] Found {len(all_records)} raw records.")

        # --- Phase 2: Multi-Source Topic Search ---
        if search_mode == 2:
            print("\n--- Phase 2: Multi-Source Topic Search ---")
            if keywords:
                query = " ".join(keywords[:3])

                futures2: Set[concurrent.futures.Future] = set()
                futures2.add(executor.submit(s2_search, query, keywords, excluded))
                futures2.add(executor.submit(crossref_search, query, keywords, excluded))
                futures2.add(executor.submit(pubmed_search, query, keywords, excluded))
                futures2.add(executor.submit(arxiv_search, query, keywords, excluded))
                if API_KEYS["CORE_API_KEY"]:
                    futures2.add(executor.submit(core_search, query, keywords, excluded))

                print("   ... Waiting for topic search workers ...")
                for f in concurrent.futures.as_completed(futures2):
                    try:
                        res = f.result()
                        if isinstance(res, list):
                            with records_lock:
                                for r in res:
                                    if r:
                                        all_records.append(r)
                    except Exception as e:
                        safe_print(f"Error: {e}")
        else:
            print("\n--- Phase 2: Skipped (Classical Snowball selected) ---")
    print(f"\n--- Phase 3: Fuzzy Deduplication (Raw: {len(all_records)}) ---")
    unique_records: List[dict] = []

    # Sort by score descending so we keep best record when duplicates exist
    all_records.sort(key=lambda x: x.get("Relevance", 0), reverse=True)

    seen_dois_dedup: Set[str] = set()

    for rec in all_records:
        doi = rec.get("DOI", "")
        title = rec.get("Title", "")

        # 1) Exact DOI match
        if doi and doi in seen_dois_dedup:
            continue

        # 2) Fuzzy title match
        is_duplicate = False
        for existing in unique_records:
            if fuzzy_match(title.lower(), existing["Title"].lower()) > FUZZY_DEDUP_THRESHOLD:
                is_duplicate = True
                # Merge metadata: keep DOI if missing
                if not existing.get("DOI") and doi:
                    existing["DOI"] = doi
                break

        if not is_duplicate:
            unique_records.append(rec)
            if doi:
                seen_dois_dedup.add(doi)

    # --- Phase 4: Export ---
    df = pd.DataFrame(unique_records)
    if not df.empty:
        df = df.sort_values(by="Relevance", ascending=False)
        out_name = "avalanche_results.xlsx"
        df.to_excel(out_name, index=False)
        print(f"\n[Success] Saved {len(df)} unique papers to {out_name}")
        print(f"Top Hit: {df.iloc[0]['Title']} (Score: {df.iloc[0]['Relevance']})")
    else:
        print("\n[!] No papers found matching criteria.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python avalanche.py <DOI>")
    else:
        main(sys.argv[1])