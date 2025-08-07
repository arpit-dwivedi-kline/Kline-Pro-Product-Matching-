
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

---

## 🔁 End-to-End Pipeline Flow


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

<img width="789" height="348" alt="image" src="https://github.com/user-attachments/assets/4543b097-f11e-4e58-b624-9d764994c4a3" />

| Threshold | Rows | False Positives | Accuracy |
|----------:|------:|----------------:|---------:|
| ≥ 1.00    | 16,093 | 30              | 99.81%   |
| ≥ 0.95    | 18,616 | 200             | 98.93%   |
| ≥ 0.70    | 25,851 | 811             | 96.86%   |
| ≥ 0.50    | 26,009 | 821             | 96.84%   |
| ≥ 0.30    | 34,532 | 3,232           | 96.38%   |
| ≥ 0.20    | 65,641 | 21,922          | 90.64%   |
| ≥ 0.10    | 120,000| 53,889          | 55.09%   |

---

## ✅ Conclusion

- Model performs well in **high-confidence zones** (≥ 0.95), showing **<1% FP rate**, making it suitable for production pipelines.
- The **FAISS + Cross-Encoder** stack offers a balance between **speed and semantic depth**.
- Further improvements can be achieved by:
  - Training on more representative augmented data for low-frequency SKUs
  - Using vector quantization for larger-scale inference
  
---

📁 **Files in this folder:**
- `faiss_notebook.ipynb`: Complete pipeline code  
- `faiss_top5_results_55.09pct_120000rows.csv`: Final predictions  
- `README.md`: This documentation

---

<!--<img width="925" height="1170" alt="image" src="https://github.com/user-attachments/assets/842103c5-cdba-4587-9e09-b73d7f36f8b6" />-->


