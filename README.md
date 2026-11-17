# NEUROLEX 3D: Interactive NLP Tokenization & Word Embedding Visualizer

> A modern, client-side, interactive NLP research and educational laboratory contained entirely within a single standalone file: [index.html](index.html).

---

## 📖 Table of Contents
1. [Overview & Philosophy](#overview--philosophy)
2. [Key Features](#key-features)
   - [1. Interactive Vocabulary Builder](#1-interactive-vocabulary-builder)
   - [2. Token Creation & 5-Step Pipeline](#2-token-creation--5-step-pipeline)
   - [3. Sentence Similarity & Closest-Pair Analysis](#3-sentence-similarity--closest-pair-analysis)
   - [4. Interactive 3D Embedding Space (Three.js)](#4-interactive-3d-embedding-space-threejs)
   - [5. Tokenizer Playground & Plotly Analytics](#5-tokenizer-playground--plotly-analytics)
   - [6. Theory Knowledge Base Modal](#6-theory-knowledge-base-modal)
3. [Mathematical Foundations](#mathematical-foundations)
   - [Jacobi Eigenvalue Algorithm & Real PCA](#jacobi-eigenvalue-algorithm--real-pca)
   - [High-Dimensional Semantic Feature Basis (16-D)](#high-dimensional-semantic-feature-basis-16-d)
   - [Similarity Metrics: Cosine vs. Jaccard](#similarity-metrics-cosine-vs-jaccard)
4. [Technology Stack](#technology-stack)
5. [How to Run](#how-to-run)

---

## Overview & Philosophy

**NEUROLEX 3D** deconstructs the inner workings of natural language processing (NLP) models. It demonstrates how raw text transforms into discrete tokens, integer vocabulary indices, and high-dimensional vector embeddings, which are then projected into interactive 2D and 3D spaces using genuine mathematical **Principal Component Analysis (PCA)**.

- **Zero Build Tools / Zero Backend**: Fully functional by double-clicking `index.html`.
- **Pure Client-Side**: No `node_modules`, no Python runtime, and no external API keys needed.
- **Mathematically Authentic**: No fake or random 3D coordinates. Points are projected using eigenvalue decomposition on high-dimensional semantic vectors.

---

## Key Features

### 1. Interactive Vocabulary Builder
- **Text Analysis**: Parses any sentence into unique vocabulary types and token instances.
- **Lexical Diversity Metrics**:
  - Vocabulary Size ($|\mathcal{V}|$)
  - Total Token Count ($N$)
  - Unique Words (single occurrence) vs. Repeated Words (frequency $> 1$)
  - Type-Token Ratio ($\text{TTR} = \frac{|\mathcal{V}|}{N} \times 100\%$)
- **Interactive Sentence Highlighting**: Click any vocabulary card to immediately highlight and spotlight all occurrences of that word in the raw sentence.
- **Heaps' Law Curve**: An interactive Plotly chart visualizing vocabulary discovery rate $V(N) = k \cdot N^\beta$ across token sequence length.
- **Searchable Table**: Filter and sort vocabulary entries by frequency, length, alphabetical order, or first occurrence.

### 2. Token Creation & 5-Step Pipeline
Visualizes the 5 sequential transformation stages of modern tokenizers:
1. **Raw Input Text**: The unformatted input stream.
2. **Normalization & Cleaning**: Unicode NFKC folding, casing transformation, and whitespace consolidation.
3. **Token Segmentation**: Splitting text along boundary delimiters into colorful, interactive blocks.
4. **Vocabulary Lookup**: Mapping string tokens to unique integer Token IDs.
5. **Model Tensor Sequence**: The final integer vector (e.g. `[101, 2043, 881, 102]`) ready for the embedding matrix.
- **Live Token Inspector**: Click on any token to display its Token ID, positional index, character offsets `[start : end]`, and byte length.

### 3. Sentence Similarity & Closest-Pair Analysis
- **Dual Sentence Inputs**: Compare two sentences with presets (Paraphrase, High Overlap, Distinct).
- **Circular Animated Radial Gauge**: Real-time animated SVG gauge displaying Cosine Similarity percentage.
- **Metrics Breakdown**: Compares Cosine Similarity against Jaccard Similarity ($|A \cap B| / |A \cup B|$).
- **3-Column Set Breakdown**: Visual categorization of words unique to Sentence A, shared in both ($A \cap B$), and unique to Sentence B.
- **Closest-Pair Semantic Analysis**: Cross-compares every word in Sentence A with Sentence B using dense vector representations, ranking matching pairs with similarity bars and distinguishing exact lexical matches from semantic analogies.

### 4. Interactive 3D Embedding Space (Three.js)
- **60+ Curated Tokens Across 7 Semantic Clusters**:
  - *AI & Computing* (Cyan)
  - *Nature & Animals* (Emerald)
  - *Science & Physics* (Blue)
  - *Royalty & Society* (Amber)
  - *Emotions & Mind* (Magenta)
  - *Actions & Verbs* (Purple)
  - *Tools & Artifacts* (Violet)
- **Interactive WebGL Scene**:
  - **OrbitControls**: Drag to rotate, right-click to pan, scroll to zoom.
  - **Billboard Sprites**: Sharp, dynamic canvas text labels that always face the camera.
  - **Dual-Layer Glowing Spheres**: Additive blending glow halos around nodes.
  - **Nearest Neighbor Graph**: Glowing 3D connection lines linking semantically related words.
  - **Mouse Raycasting**: Hovering over a node displays a glass tooltip with 3D coordinates $(x, y, z)$ and nearest neighbors.
  - **Click-to-Inspect**: Focuses camera on the selected word, highlights its nearest connections, and opens a docked inspector showing its raw 16-D feature vector.
  - **Dynamic Word Injector**: Type any custom word or sentence; its embedding is synthesized on the fly and projected into 3D space with real-time PCA re-calculation!

### 5. Tokenizer Playground & Plotly Analytics
Experiment with three distinct tokenization algorithms:
- **Word-Level**: Splits on whitespace and punctuation.
- **Subword-Style**: Simulates Byte-Pair Encoding (BPE) / WordPiece, extracting root morphemes and `##` continuation subwords (e.g. `un` + `##break` + `##able`).
- **Character-Level**: Deconstructs text into individual atomic character tokens.

Generates **5 distinct interactive Plotly charts**:
1. **2D PCA Embedding Scatter**: Principal Components 1 & 2 with token length color gradients.
2. **3D PCA Embedding Scatter**: 3D orbital scatter plot with spatial coordinates.
3. **Token Frequency Distribution**: Ranked bar chart with frequency counts.
4. **Token Position Timeline**: Horizontal Gantt-style span chart displaying character index offsets `[start : end]`.
5. **Pairwise Cosine Similarity Heatmap**: Full $N \times N$ token similarity correlation matrix.

### 6. Theory Knowledge Base Modal
An in-app modal explaining:
- What is a Vocabulary & Out-of-Vocabulary (OOV) handling?
- What are Tokens and Token IDs?
- Sparse One-Hot vs. Dense Continuous Embeddings.
- Cosine Similarity vs. Jaccard Index.
- Dimensionality Reduction & Step-by-Step PCA.

---

## Mathematical Foundations

### Jacobi Eigenvalue Algorithm & Real PCA
To avoid artificial or arbitrary coordinates, dimensionality reduction uses a self-contained pure JavaScript **Jacobi Eigenvalue Algorithm**:

1. **Centering**:
   $$\mathbf{X}_c = \mathbf{X} - \boldsymbol{\mu}, \quad \boldsymbol{\mu} = \frac{1}{N}\sum_{i=1}^N \mathbf{x}_i$$
2. **Sample Covariance Matrix**:
   $$\mathbf{C} = \frac{1}{N-1} \mathbf{X}_c^T \mathbf{X}_c \in \mathbb{R}^{d \times d}$$
3. **Jacobi Rotation**: Iteratively eliminates off-diagonal elements $C_{pq}$ by rotating the coordinate frame with orthogonal transformation $\mathbf{R}(\theta)$:
   $$\theta = \frac{1}{2}\arctan\left(\frac{2 C_{pq}}{C_{qq} - C_{pp}}\right)$$
4. **Projection**: Centered vectors are projected onto the top eigenvectors corresponding to the largest eigenvalues:
   $$\mathbf{z}_i = \mathbf{x}_{c,i} \cdot [\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3]$$

### High-Dimensional Semantic Feature Basis (16-D)
Each word vector exists in a 16-dimensional semantic basis representing:
$$\mathbf{v} = [v_{\text{animate}}, v_{\text{tech}}, v_{\text{science}}, v_{\text{royalty}}, v_{\text{emotion}}, v_{\text{action}}, v_{\text{cognition}}, v_{\text{nature}}, v_{\text{magnitude}}, v_{\text{creative}}, v_{\text{tools}}, v_{\text{speed}}, v_{\text{light}}, v_{\text{solid}}, v_{\text{language}}, v_{\text{grammar}}]$$

For out-of-vocabulary words entered by the user, an embedding synthesizer combines FastText-style character $n$-gram hashing, morphological affix rules (`-tion`, `-ing`, `micro-`, `cyber-`), and contextual co-occurrence window averaging.

### Similarity Metrics: Cosine vs. Jaccard
- **Cosine Similarity (Directional angle in vector space)**:
  $$\cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$
- **Jaccard Similarity (Exact lexical set overlap)**:
  $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

---

## Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Structure & Logic** | HTML5 / Vanilla ES6+ | Single-file application (`index.html`) |
| **Styling** | Modern CSS3 | Glassmorphism, CSS grid, dark futuristic theme |
| **3D Graphics** | Three.js (r128) + OrbitControls | WebGL scene, canvas text sprites, raycaster |
| **2D / 3D Charts** | Plotly.js (2.29.1) | Scatter 2D/3D, frequency bars, timelines, heatmaps |
| **Typography** | Google Fonts | Inter, Space Grotesk, JetBrains Mono |
| **Icons** | FontAwesome 6 | Minimal, clean vector iconography |

---

## How to Run

1. **Directly in your browser**:
   Double click [index.html](index.html) or run in terminal:
   ```bash
   open index.html
   ```

2. **Via any local HTTP server** (optional):
   ```bash
   python3 -m http.server 8000
   ```
   Then open `http://localhost:8000` in your web browser.

