# 📊 CPI Item Co-Movement Network Analysis

## 🧠 Project Overview
This project models and analyzes **consumer price movement similarity across multiple cities in Pakistan** using official CPI data from the Pakistan Bureau of Statistics (PBS).

Instead of treating cities as nodes, this project treats **consumer items (e.g., rice, milk, oil)** as nodes in a graph. Two items are connected if they show **similar price movement patterns across cities and time**.

The goal is to build a **temporal item–item network** and analyze it using graph theory, similarity measures, and centrality metrics.

---

## 🎯 Objective

- Model items as graph nodes (not cities)
- Measure similarity in price movements across cities
- Build item–item networks using cosine similarity
- Study:
  - Centrality of essential goods
  - Temporal changes across years
  - Category-level interactions
- Apply graph theory + data analysis to real CPI data

---

## 📦 Dataset

- Source: Pakistan Bureau of Statistics (PBS)
- Type: Monthly CPI reports (Annexures)
- Coverage: 3 years of monthly data
- Dimensions:
  - Items (consumer goods)
  - Cities (e.g., Karachi, Lahore, Islamabad)
  - Time (monthly observations)

---

## 🏗️ Project Structure
```bash
project/
├── data/
│   └── cpi_data.csv          # scraped/downloaded data
├── src/
│   ├── scraper.py            # Step 1: data collection
│   ├── preprocess.py         # Step 2: price change vectors
│   ├── similarity.py         # Steps 3-4: cosine similarity + city count
│   ├── graph_builder.py      # Step 5: graph construction
│   ├── analysis.py           # Steps 6-7: centrality + temporal
│   └── visualize.py          # Step 8: plots
├── main.py                   # runs full pipeline
└── report/
```

## ⚙️ Pipeline Workflow

The project runs in the following stages:

1. **Data Collection**
   - Download CPI PDFs from PBS
   - Extract Annexure tables

2. **Preprocessing**
   - Convert prices into time-series vectors
   - Normalize item names

3. **Similarity Computation**
   - Compute cosine similarity between item vectors
   - Identify strongly related items

4. **Graph Construction**
   - Build item–item network
   - Apply thresholds (τ, K)

5. **Network Analysis**
   - Degree centrality
   - Betweenness centrality
   - Temporal stability

6. **Visualization**
   - Network graphs per year
   - Heatmaps
   - Centrality plots

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Download data
```bash
python download_pdfs.py
```
3. Run full pipeline
```bash
python main.py
```
4. Run specific stage
```bash
python main.py --stage preprocess
python main.py --stage similarity
python main.py --stage analysis
python main.py --stage visualize
```
📊 Key Features
✔ Item-based graph modeling
✔ Cross-city price movement analysis
✔ Temporal network evolution (3 years)
✔ Cosine similarity-based edge creation
✔ Centrality ranking of essential goods
✔ Category-level network structure
📈 Outputs

The pipeline generates:

Clean CPI dataset (cpi_data.csv)
Similarity matrices
Graph objects per year
Centrality rankings
Temporal stability analysis
Visual network graphs

🧰 Tech Stack
Python
Pandas
NumPy
NetworkX
pdfplumber
Matplotlib
