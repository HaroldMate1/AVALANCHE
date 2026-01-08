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

# ----------------------------
# Config: APIs & Rate Limiting
# ----------------------------
APIS = {
    "openalex": {
        "base": "https://api.openalex.org/works",
        "rate_limit": 0.1,  
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
MAX_WORKERS = 16       
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
    limiter = rate_limiters.get(api_name)
    if limiter: limiter.wait()

    h = {"User-Agent": USER_AGENT}
    if headers: h.update(headers)
    
    # Specific Auth Headers
    if api_name == "semantic_scholar" and API_KEYS["S2_API_KEY"]:
        h["x-api-key"] = API_KEYS["S2_API_KEY"]
    if api_name == "core" and API_KEYS["CORE_API_KEY"]:
        h["Authorization"] = f"Bearer {API_KEYS['CORE_API_KEY']}"

    try:
        r = requests.get(url, params=params, headers=h, timeout=15)
        if r.status_code == 200:
            return r.text if is_xml else r.json()
        elif r.status_code == 429:
            safe_print(f"  [⚠️ Limit] {api_name} slowing down...")
            time.sleep(2)
    except Exception:
        pass
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
    except: pass
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
    except: pass
    return [r for r in results if r]

# 6. OpenAlex Helper (Invert Index)
def openalex_invert(inv_idx):
    if not inv_idx: return ""
    tokens = {}
    for word, positions in inv_idx.items():
        for p in positions: tokens[p] = word
    if not tokens: return ""
    return " ".join([tokens[i] for i in sorted(tokens.keys())])

# ----------------------------
# Main Workflow
# ----------------------------
def main(seed_doi: str):
    print(f"--- SOTA-Pro v5.0 (Federated) | DOI: {seed_doi} ---")
    
    kw_input = input(" > Keywords (comma sep): ").strip()
    keywords = [k.strip() for k in kw_input.split(",") if k.strip()]
    
    ex_input = input(" > Exclude terms (comma sep): ").strip()
    excluded = [k.strip() for k in ex_input.split(",") if k.strip()]
    
    print("\n > Citation Depth (how deep to follow references):")
    print("   1 = seed + direct references/citations")
    print("   2 = seed + refs/citations + their refs/citations (default)")
    print("   3 = go 3 levels deep (slower, more comprehensive)")
    depth_input = input(" > Depth (1-3, default: 2): ").strip()
    depth = int(depth_input) if depth_input.isdigit() and 1 <= int(depth_input) <= 3 else 2
    print(f"   → Using depth: {depth}")

    print("\n > Search Mode Selection:")
    print("   1 = Classical Snowball (Graph Only: Seed + Citations/References)")
    print("   2 = Dual Process (Graph + Multi-Source Keyword Search) [Default]")
    mode_input = input(" > Select Mode (1 or 2): ").strip()
    search_mode = 1 if mode_input == "1" else 2
    print(f"   → Using Mode: {'Classical Snowball' if search_mode == 1 else 'Dual Process'}")

    # --- Phase 1: Seed & Snowball (OpenAlex + Fallback) ---
    print("\n--- Phase 1: Federated Graph Search ---")
    
    # Check OpenAlex first
    seed_url = f"{APIS['openalex']['base']}/https://doi.org/{normalize_doi(seed_doi)}"
    seed_data = fetch_url(seed_url, "openalex")
    
    all_records = []
    seen_dois = set()
    queue = [] # (id, type, depth)

    if seed_data:
        # OpenAlex Success
        oa_abs = openalex_invert(seed_data.get("abstract_inverted_index"))
        seed_rec = make_record("OpenAlex", seed_data["display_name"], seed_data["publication_year"], 
                               seed_data["doi"], seed_data["cited_by_count"], oa_abs, 
                               (seed_data.get("primary_location") or {}).get("source", {}).get("display_name"),
                               seed_data["id"], "SEED", keywords, [])
        if seed_rec:
            all_records.append(seed_rec)
            seen_dois.add(normalize_doi(seed_rec["DOI"]))
            print(f"   [Seed Found] {seed_rec['Title'][:60]}...")
            
            # Add refs to queue
            refs = seed_data.get("referenced_works", [])
            queue.append((refs, "refs", 1))
            queue.append((seed_data["id"], "cites", 1))

    else:
        # OpenAlex Failed -> Try Semantic Scholar
        print("   [!] Seed not in OpenAlex. Trying Semantic Scholar...")
        s2_url = f"{APIS['semantic_scholar']['base']}/paper/DOI:{normalize_doi(seed_doi)}?fields=title,references,citations"
        s2_data = fetch_url(s2_url, "semantic_scholar")
        if s2_data:
             # Add refs from S2
             refs = [r["paperId"] for r in s2_data.get("references", []) if r.get("paperId")]
             queue.append((refs, "s2_refs", 1))

    # Processing Graph Queue
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # A. Process Snowball
        futures = []
        
        # Helper to fetch OpenAlex chunks
        def fetch_oa_chunk(ids, d):
            u = f"{APIS['openalex']['base']}?filter=openalex_id:{'|'.join(ids)}&per-page=50"
            res = fetch_url(u, "openalex")
            ret = []
            if res and "results" in res:
                for w in res["results"]:
                    r = make_record("OpenAlex", w["display_name"], w["publication_year"], w["doi"],
                                    w["cited_by_count"], openalex_invert(w.get("abstract_inverted_index")),
                                    "", w["id"], f"Graph (Depth {d})", keywords, excluded)
                    if r: ret.append(r)
            return ret

        while queue:
            item, q_type, q_depth = queue.pop(0)
            if q_depth >= depth: continue
            
            if q_type == "refs":
                # Chunk into 50s
                for i in range(0, len(item), 50):
                    futures.append(executor.submit(fetch_oa_chunk, item[i:i+50], q_depth))
            
            elif q_type == "cites":
                # Fetch citations
                u = f"{APIS['openalex']['base']}?filter=cites:{item}&per-page=50"
                futures.append(executor.submit(lambda url=u: fetch_url(url, "openalex")))

        # B. Process Topic Search (Parallel)
        # ONLY IF DUAL PROCESS IS SELECTED
        if search_mode == 2:
            print("\n--- Phase 2: Multi-Source Topic Search ---")
            if keywords:
                query = " ".join(keywords[:3])
                futures.append(executor.submit(s2_search, query, keywords, excluded))
                futures.append(executor.submit(crossref_search, query, keywords, excluded))
                futures.append(executor.submit(pubmed_search, query, keywords, excluded))
                futures.append(executor.submit(arxiv_search, query, keywords, excluded))
                if API_KEYS["CORE_API_KEY"]:
                    futures.append(executor.submit(core_search, query, keywords, excluded))
        else:
            print("\n--- Phase 2: Skipped (Classical Snowball selected) ---")

        # Collect Results
        print("   ... Waiting for workers ...")
        for f in concurrent.futures.as_completed(futures):
            try:
                res = f.result()
                if isinstance(res, list): # From Search or OA Chunk
                    for r in res:
                        all_records.append(r)
                elif isinstance(res, dict) and "results" in res: # From Cites
                    for w in res["results"]:
                          r = make_record("OpenAlex", w["display_name"], w["publication_year"], w["doi"],
                                    w["cited_by_count"], openalex_invert(w.get("abstract_inverted_index")),
                                    "", w["id"], "Graph Citation", keywords, excluded)
                          if r: all_records.append(r)
            except Exception as e:
                safe_print(f"Error: {e}")

    # --- Phase 3: Fuzzy Deduplication ---
    print(f"\n--- Phase 3: Fuzzy Deduplication (Raw: {len(all_records)}) ---")
    unique_records = []
    
    # Sort by Score (desc) so we keep best matches
    all_records.sort(key=lambda x: x["Relevance"], reverse=True)
    
    for rec in all_records:
        doi = rec["DOI"]
        title = rec["Title"]
        
        # 1. Exact DOI Match
        if doi and doi in seen_dois:
            continue
            
        # 2. Fuzzy Title Match
        is_duplicate = False
        for existing in unique_records:
            # If titles are 90% similar, treat as same paper
            if fuzzy_match(title.lower(), existing["Title"].lower()) > FUZZY_DEDUP_THRESHOLD:
                is_duplicate = True
                # Merge metadata (keep DOI if missing)
                if not existing["DOI"] and doi: existing["DOI"] = doi
                break
        
        if not is_duplicate:
            unique_records.append(rec)
            if doi: seen_dois.add(doi)

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