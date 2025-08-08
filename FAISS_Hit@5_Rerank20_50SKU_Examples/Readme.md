
# 🔍 Semantic Product Matching using FAISS and Re-Ranking

This folder contains the core notebook, results, and related outputs for the SKU matching model built using dense vector search (FAISS) and cross-encoder re-ranking. The objective is to build a **scalable, high-precision product-matching pipeline** that works well even with large-scale unstructured catalog data.

---

## 🎯 Objective

The primary aim is to **retrieve and rank semantically similar SKUs** using embeddings rather than simple lexical matching. We use FAISS for fast nearest-neighbor retrieval and then apply a transformer-based cross-encoder to refine and re-rank the top candidates.

---

## 📊 Why Filter for SKUs with ≥ 50 Examples?

Many SKUs in the dataset have **very few examples**, resulting in a **long-tail data skew**. To maintain data density and ensure effective training, we applied a threshold filter:

- **Total mapped rows**: 2,996,970  
- **Distinct SKUs**: 95,010  
- **SKUs with ≥ 50 examples**: 12,249  
- **Rows with SKUs ≥ 50 examples**: 2,312,153  
- This subset accounts for **~77.15%** of all mapped rows.

> 🔹 Hence, we focus our modeling on this dense portion of the data.  
> 🔹 For the long-tail SKUs (< 50 examples), future work may include **text-based data augmentation**.
<img width="624" height="139" alt="image" src="https://github.com/user-attachments/assets/c9193d4c-3109-41d8-86a0-2e455fb03762" />
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/a57e3132-c1eb-48d2-83b8-5953cfdb6b77" />




---

## 🔁 End-to-End Pipeline Flow

Our end-to-end flow 

  Clean & filter: keep only mapped rows; parse sizes; validate barcodes.
  
  Build KB: one canonical text per master product (brand, product name, category, size…).
  
  Embed:
  
  KB texts → vectors (once).
  
  Source queries → vectors (per record).
  
  Index & retrieve: FAISS index over KB vectors (optionally per category shard); top-K (e.g., 50) per query.
  
  Re-rank: logistic regression uses features (fuzzy text sim, category/brand exact, size ratio, barcode hit) to score candidates.
  
  Decide + metrics: choose top-1 (Acc@1), also report Hit@5; apply confidence thresholds for production.
<img width="530" height="1447" alt="image" src="https://github.com/user-attachments/assets/ef26666b-daaa-49b3-bdfd-6d9b49f19fcb" />

---

## 🧠 Core Components

### 🔹 Embedding Generation (Knowledge Base)

- Embeddings for SKUs are generated using:  
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Each SKU is represented as a **384-dimensional dense vector**
- FAISS Index: FlatL2 (no quantization for maximum accuracy)
- Final index size: 12,000 entries

### 🔹 Query Embedding & Candidate Retrieval

- 600,000 total queries were embedded (100k test)
- Top-20 candidates retrieved for each using FAISS

<img width="958" height="558" alt="image" src="https://github.com/user-attachments/assets/66972676-58c9-4a65-abd6-879720f51a68" />


### 🔹 Re-Ranking (Fine-Grained Match)

- Retrieved candidates are re-scored using:  
  `cross-encoder/ms-marco-MiniLM-L-6-v2`
- From 20 candidates, **top-5 are selected as final predictions**
- Re-ranker trained using 480k rows, tested on 120k rows
- **Accuracy@1 (for single-ID cases)**: 99.81%  
- **Overall Hit@5**: 55.09%

---

## 🔬 Multi-ID Threshold Tuning

To analyze performance trade-offs, we evaluated false-positive rates across confidence thresholds.

<img width="671" height="289" alt="image" src="https://github.com/user-attachments/assets/ea9b9059-ae5a-4e90-9915-8ed7c81b3418" />


Observations:
- Threshold ≥ 0.30 → 17.37% FP rate
- Threshold ≥ 0.50 → 7.98% FP rate
- Threshold ≥ 0.95 → 6.74% FP rate  
- Lowering threshold increases recall, but also FP rate — a **precision-recall trade-off**

---

## 🧮 Final Threshold-wise Performance Table

<!--<img width="789" height="348" alt="image" src="https://github.com/user-attachments/assets/4543b097-f11e-4e58-b624-9d764994c4a3" />-->

<img width="1061" height="324" alt="image" src="https://github.com/user-attachments/assets/c7e109b4-4c49-4fbb-8fb6-de962fd7adf2" />

<img width="1089" height="485" alt="image" src="https://github.com/user-attachments/assets/e5d282d5-7df9-4393-8ddb-b6ccb71f4a68" />


## ✅ Conclusion

- Model performs well in **high-confidence zones** (≥ 0.95), showing **<1% FP rate**, making it suitable for production pipelines.
- The **FAISS + Cross-Encoder** stack offers a balance between **speed and semantic depth**.
- Further improvements can be achieved by:
  - Training on more representative augmented data for low-frequency SKUs
  - Using vector quantization for larger-scale inference
  
---

# Experimentation With Augmentation

For SKUs with <50 train rows, synthesize new rows up to 50 using:

Synonym swaps / word order shuffles / light typos in SourceDescription.

Brand/alias swaps from master-side names.

Size normalization & unit conversions (e.g., 500 ml ↔ 0.5 L) with consistent numeric source_ml.

Safe noise (punctuation/spacing).
<img width="764" height="393" alt="image" src="https://github.com/user-attachments/assets/03bb6636-3536-47da-ba79-36967d86cae2" />
<img width="768" height="393" alt="image" src="https://github.com/user-attachments/assets/ceac8e51-8ad1-4ded-b0bf-e1eb384d8a6b" />

<img width="1021" height="112" alt="image" src="https://github.com/user-attachments/assets/a0fac1a4-5aaf-48b4-a0e3-948764c922d1" />
<img width="1076" height="320" alt="image" src="https://github.com/user-attachments/assets/b90a562d-fc19-4276-a88f-5acd08d81e54" />
<img width="1089" height="484" alt="image" src="https://github.com/user-attachments/assets/37923ef1-6d72-4141-baa4-070c2ec22788" />


De-duplicate and cap at 50.

📁 **Files in this folder:**
- `faiss_notebook.ipynb`: Complete pipeline code  
- `faiss_top5_results_55.09pct_120000rows.csv`: Final predictions  
- `README.md`: This documentation

---

<!--<img width="925" height="1170" alt="image" src="https://github.com/user-attachments/assets/842103c5-cdba-4587-9e09-b73d7f36f8b6" />-->


