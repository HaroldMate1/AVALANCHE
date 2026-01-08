# ❄️ Avalanche

**Avalanche** is a federated literature discovery tool that automates the "snowballing" method of research. It starts with a single seed paper (DOI) and triggers an avalanche of relevant literature by traversing citation graphs and querying multiple academic APIs simultaneously.

## ✨ Features

* **Dual Search Modes:**
    * **Mode 1 (Classical Snowball):** Strictly traverses the citation graph (References & Citations) of the seed paper.
    * **Mode 2 (Dual Process):** Combines the citation graph with a federated keyword search across Semantic Scholar, PubMed, ArXiv, Crossref, and CORE.
* **Federated Search:** Queries 6+ academic databases concurrently.
* **Smart Scoring:** Auto-ranks papers based on title/abstract relevance to your keywords.
* **Fuzzy Deduplication:** Merges duplicate entries from different sources automatically.
* **Excel Export:** Outputs a clean, sorted `.xlsx` file.

## 🚀 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/HaroldMate1/avalanche.git](https://github.com/HaroldMate1/avalanche.git)
    cd avalanche
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys (Optional but Recommended)**
    Create a `.env` file or export these variables in your terminal. While the script runs without them, having them increases rate limits.
    * `OPENALEX_EMAIL` (Highly recommended for polite pooling)
    * `S2_API_KEY` (Semantic Scholar)
    * `CORE_API_KEY` (CORE.ac.uk)

## 📖 Usage

Run the script providing the DOI of your "Seed Paper":

```bash
python avalanche.py 10.1038/s41586-020-2649-2