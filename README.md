# breast-cancer-ppi-analysis
Topological analysis and functional module mining of breast cancer Protein-Protein Interaction (PPI) networks using graph theory, Louvain community detection, and Graph Neural Networks (GNN).

# Breast Cancer PPI Network Analysis & Functional Module Mining

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![NetworkX](https://img.shields.io/badge/NetworkX-2.x-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

## 📖 Overview
[cite_start]This repository contains the source code and comprehensive report for a computational biology project focusing on the **Protein-Protein Interaction (PPI) network** of breast cancer[cite: 80, 94]. [cite_start]By abstracting biological entities into a graph structure, this project applies advanced graph theory, community detection algorithms (Louvain), and Graph Neural Networks (GNN) to decode the molecular mechanisms of breast cancer[cite: 80, 94, 96, 97]. 

[cite_start]The research aims to identify critical hub proteins, uncover functional modules (e.g., DNA repair, cell cycle regulation), and propose multi-target therapeutic strategies through network robustness simulations[cite: 80, 96, 98].

## ✨ Key Features
* [cite_start]**Topological Network Analysis:** Quantitative assessment of global network properties, validating scale-free and small-world characteristics[cite: 95]. [cite_start]Hubs and bottlenecks are identified using Degree and Betweenness Centrality[cite: 218].
* [cite_start]**Mesoscopic Module Mining:** Unsupervised community detection using the **Louvain algorithm** (modularity optimization) to discover densely connected functional subgraphs[cite: 316, 317].
* [cite_start]**Deep Feature Learning (GNN):** Implementation of a **Graph Auto-Encoder (GAE)** to capture high-order, non-linear protein interactions in a latent space, revealing hidden metabolic modules[cite: 97, 222, 566, 567].
* [cite_start]**Global Structure Validation:** Application of **Spectral Biclustering** to verify the hierarchical modularity and inter-module communication[cite: 588, 590].
* [cite_start]**Robustness Simulation:** Targeted attack vs. random failure simulations to analyze network redundancy and drug resistance mechanisms[cite: 98, 488, 489].

## 📊 Data Sources
* [cite_start]**PPI Data:** High-confidence interaction data (Combined Score > 0.7) extracted from the **STRING Database**[cite: 80, 117].
* [cite_start]**Seed Genes:** 10 breast cancer driver genes (e.g., TP53, BRCA1, EGFR) sourced from **OMIM / DisGeNET**[cite: 80, 104, 203].
* [cite_start]**Functional Annotation:** Gene Ontology (GO) and KEGG pathways enriched via the `gseapy` library (Enrichr API)[cite: 152, 163].

## 🔬 Core Findings
1.  **Three Major Pathogenic Modules:** The network is structurally driven by three main engines: 
    * [cite_start]*Genome Guardian & DNA Repair* (TP53/BRCA1/ATM) [cite: 437, 438]
    * [cite_start]*Growth & Survival Signaling* (EGFR/PIK3CA/PTEN) [cite: 440, 441]
    * [cite_start]*Cell Cycle Driving* (MYC/CCND1) [cite: 443, 444]
2.  [cite_start]**High Functional Redundancy:** Node-removal simulations reveal that attacking top hubs only partially collapses the network, providing a topological basis for the necessity of "cocktail therapy" in treating breast cancer[cite: 489, 492, 494].
3.  [cite_start]**Hidden Sub-modules:** Latent space visualization from the GNN successfully isolated a hidden mitochondrial energy metabolism module (ATP Synthase Complex)[cite: 583, 584].

## ⚙️ Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed. Install the required dependencies using:
```bash
pip install -r requirements.txt
