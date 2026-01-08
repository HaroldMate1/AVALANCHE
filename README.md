# ❄️ Avalanche

Automated Federated Literature Discovery Tool
Avalanche is a powerful Python-based tool designed to automate the "snowballing" method of systematic literature review. By traversing citation graphs and querying multiple academic APIs simultaneously, it turns a single seed paper into a comprehensive, deduplicated, and ranked bibliography.
🚀 Why Avalanche?
Have you ever struggled to grasp the true "State of the Art" of a research topic?
We've all been there: You find one great paper, and then spend the next six hours opening fifty tabs, manually checking who cited it and who they cited. You end up bouncing between PubMed, ArXiv, and Semantic Scholar, fighting through a mess of duplicate entries, paywalls, and irrelevant noise. It is tedious, prone to human error, and exhausting.
Avalanche changes the game.
Instead of manual "tab-hopping," Avalanche automates the rigorous literature discovery process. You simply feed it a single Seed Paper (DOI), and it triggers a controlled landslide of relevant research.
How it works for you:
• One Seed, Full Harvest: It traverses the citation graph (both backward references and forward citations) to map the entire lineage of an idea.
• Unified Intelligence: It queries 6+ academic databases concurrently (including OpenAlex, PubMed, and Crossref).
• Zero Duplicates: Its fuzzy deduplication engine automatically merges the "Smith et al." from PubMed with the "J. Smith" from ArXiv, leaving you with a clean, single entry.
• Ranked & Ready: Instead of a chaotic list, you get a smart-scored Excel sheet sorted by relevance to your specific keywords.
Stop searching paper-by-paper. Start an Avalanche.
🧠 Under the Hood: Snowballing & Graphs
Avalanche isn't just a keyword search; it is built on two fundamental concepts of bibliometrics and network science.
1. The Snowballing Method
In systematic literature reviews, "Snowballing" is the gold standard for finding connections that keyword searches miss. It works in two directions:
• Backward Snowballing (References): Looking at the bibliography of your seed paper to see the older research that built the foundation.
• Forward Snowballing (Citations): Looking at newer papers that have cited your seed paper to see how the idea has evolved over time.
Manually, this is incredibly slow—you have to open every paper to check its references. Avalanche automates this instantly, pulling both the past (references) and future (citations) of your topic in seconds.
2. Graph Theory
Avalanche treats the world of academic literature as a massive, interconnected network (a Graph).
• Nodes (The Dots): Every paper is a "node" in the network.
• Edges (The Lines): Every citation is a directional "edge" connecting two nodes.
When you provide a Seed DOI, you are identifying the starting Node. Avalanche then traverses the edges connected to that node, hopping from paper to paper to map out the local cluster of relevant research. By treating literature as a graph rather than a list, Avalanche finds the hidden relationships between papers that standard search engines often overlook.
✨ Features
• Dual Search Modes:
• Mode 1 (Classical Snowball): Strictly traverses the citation graph (References & Citations) of the seed paper for high-precision mapping.
• Mode 2 (Dual Process): Combines the citation graph with a federated keyword search across Semantic Scholar, PubMed, ArXiv, Crossref, and CORE for broad scoping.
• Federated Search: Queries 6+ academic databases concurrently to ensure no stone is left unturned.
• Smart Scoring: Auto-ranks papers based on title/abstract relevance to your specific keywords.
• Fuzzy Deduplication: Intelligently merges duplicate entries from different sources automatically (e.g., matching pre-prints with published versions).
• Excel Export: Outputs a clean, sorted .xlsx file ready for immediate review.

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