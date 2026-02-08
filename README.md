# ❄️ Avalanche  
**Automated Federated Literature Discovery Tool**

Avalanche is a Python-based tool that automates the **snowballing method** for systematic literature reviews. Starting from a single seed paper, it traverses citation graphs and queries multiple academic databases concurrently to produce a **deduplicated, ranked, and review-ready bibliography**.

Stop opening dozens of tabs. Start an Avalanche. 🚀

---

## 📌 Why Avalanche?

Have you ever tried to understand the true *state of the art* in a research area?

You find one excellent paper… then spend hours jumping between PubMed, arXiv, Semantic Scholar, and Google Scholar—manually checking references, chasing citations, fighting duplicates, and filtering noise.

It’s slow, exhausting, and error-prone.

**Avalanche changes the game.**

You provide **one seed paper (DOI)**, and Avalanche triggers a controlled landslide of relevant research—automatically.

---

## ⚙️ How It Works

### One Seed, Full Harvest
Avalanche traverses the citation graph in both directions:
- **Backward snowballing**: references cited by the seed paper  
- **Forward snowballing**: papers that cite the seed paper  

This captures both the *foundations* and the *evolution* of an idea.

### Unified Intelligence
Queries **6+ academic databases concurrently**, including:
- OpenAlex  
- PubMed  
- Semantic Scholar  
- arXiv  
- Crossref  
- CORE  

### Zero Duplicates
A fuzzy deduplication engine merges equivalent records across sources  
(e.g. *“Smith et al.”* vs *“J. Smith”*, preprints vs published versions).

### Ranked & Ready
Results are **smart-scored** by relevance to your keywords and exported as a **clean, sorted Excel file**.

---

## 🧠 Under the Hood

### 1. Snowballing Method
Snowballing is the gold standard in systematic literature reviews for uncovering connections that keyword searches miss.

Avalanche automates:
- **Backward snowballing (References)** – older foundational work  
- **Forward snowballing (Citations)** – newer research building on the idea  

What normally takes hours of manual effort happens in seconds.

---

### 2. Graph Theory
Avalanche models academic literature as a **graph**:

- **Nodes** → individual papers  
- **Edges** → citation relationships  

Providing a seed DOI identifies a starting node. Avalanche then traverses connected edges to map the local research cluster, revealing hidden relationships that traditional search engines often overlook.

---

## ✨ Features

- **Dual Search Modes**
  - **Mode 1 – Classical Snowball**  
    Pure citation traversal (references + citations) for high-precision mapping
  - **Mode 2 – Dual Process**  
    Combines citation traversal with federated keyword search for broad scoping

- **Federated Search**  
  Concurrent queries across 6+ academic databases

- **Smart Scoring**  
  Automatically ranks papers by relevance to your keywords

- **Fuzzy Deduplication**  
  Intelligently merges duplicate entries across sources

- **Excel Export**  
  Outputs a clean `.xlsx` file ready for immediate review

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/HaroldMate1/avalanche.git
   cd avalanche
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys** (recommended for best performance)
   ```bash
   cp .env.example .env
   # Edit .env and add your email/API keys
   ```

   At minimum, add your email for faster OpenAlex access:
   ```
   OPENALEX_EMAIL=your.email@university.edu
   ```

   Optional but recommended for 10x speed:
   - Get Semantic Scholar API key: https://www.semanticscholar.org/product/api
   - Get CORE API key: https://core.ac.uk/services/api

---

## 📖 Usage

Run the script providing the DOI of your "Seed Paper":

```bash
python avalanche.py 10.1016/j.addr.2017.09.009
```

You'll be prompted for:
- **Keywords**: Terms to search for (e.g., "machine learning, transformers")
- **Exclude terms**: Terms to filter out (e.g., "survey, review")
- **Depth**: How many levels deep to follow citations (1-6, default: 2)
- **Mode**:
  - Mode 1: Classical Snowball (citation graph only)
  - Mode 2: Dual Process (graph + keyword search across databases)

---

## 🔧 Recent Updates (v1.1 - 2026-01-19)

### Critical Bug Fixes
- ✅ Fixed duplicate code execution bug that caused double processing
- ✅ Added thread-safe operations to prevent race conditions
- ✅ Added DOI validation with clear error messages
- ✅ Improved error logging (no more silent failures)
- ✅ Added API configuration checker at startup

### Improvements
- Better user feedback during execution
- Clearer warnings for missing API keys
- Enhanced error handling throughout
- Added `.env.example` template for easy setup

---

## 📝 Citation

If you use AVALANCHE in your research, please cite:

```
AVALANCHE: Automated Federated Literature Discovery Tool
Harold Mateo Mojica Urrego, University of Navarra-TECNUN, 2026
GitHub: https://github.com/HaroldMate1/AVALANCHE
