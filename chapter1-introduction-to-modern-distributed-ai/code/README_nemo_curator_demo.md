# NeMo Curator Quick Demo

This demo script demonstrates NeMo Curator's core functionality: **data cleaning, deduplication, and quality filtering**, completing in under 1 minute.

## Features Demonstrated

This example demonstrates the following features:

1. **Quality Filtering**:
   - Word count filter: Keep documents with 50-10000 words
   - Symbol ratio filter: Symbols cannot exceed 25%
   - Repeated lines filter: Repeated lines cannot exceed 70%

2. **Exact Deduplication**:
   - Use MD5 hashing to identify identical documents
   - Automatically assign unique IDs

3. **Data Reduction Effects**:
   - Show the transformation process from raw data to high-quality data
   - Statistics on data reduction rate

## Requirements

- Python 3.10+
- NeMo Curator installed (`nemo-curator[text]` or `nemo-curator[text_cuda12]`)
- Ray (automatically managed by NeMo Curator)

## How to Run

```bash
cd code/chapter1
python nemo_curator_demo.py
```

## Expected Output

```
NeMo Curator Quick Demo
============================================================
This example demonstrates:
  1. Quality filtering (word count, symbol ratio, repeated lines)
  2. Exact deduplication
  3. Data reduction effects
============================================================
✓ Created 10 sample documents
  - High-quality documents: 4
  - Low-quality documents: 4 (too short, too many symbols, repeated lines, too long)
  - Duplicate documents: 2 (identical to high-quality documents)

============================================================
Starting data curation pipeline...
============================================================

[Stage 1] Quality filtering...
  ✓ Quality filtering completed (time: X.XXs)
  ✓ Documents retained: 6

[Stage 2] Exact deduplication...
  ✓ Deduplication completed (time: X.XXs)
  ✓ Documents after deduplication: 4

============================================================
Data Curation Results Statistics
============================================================
Input documents:     10
After filtering:     6  (60.0%)
After deduplication: 4  (40.0%)

Processing time:
  Quality filtering: X.XX seconds
  Deduplication:     X.XX seconds
  Total:             X.XX seconds

Data reduction rate: 60.0%

Output file:         /tmp/.../curated_data.jsonl
============================================================

✓ Demo completed!
```

## Sample Data Description

The sample data contains:
- **4 high-quality documents**: Complete paragraphs about machine learning, distributed systems, deep learning, and NLP
- **4 low-quality documents**:
  - Too short (< 50 words)
  - Too many symbols (> 25%)
  - Too many repeated lines (> 70%)
  - Too long (> 10000 words)
- **2 duplicate documents**: Identical to high-quality documents

## Comparison with Real-World Applications

| Feature | This Demo | Real Application (RedPajama v2) |
|---------|----------|--------------------------------|
| Data Scale | 10 documents | 8TB, 1.78 trillion tokens |
| Processing Time | < 1 minute | 0.5 hours (32 H100 GPUs) |
| Data Reduction | 60% | 80% |
| Processing Capacity | Single machine | Multi-node cluster |

## Extended Usage

In real-world applications, you can:

1. **Replace data source**: Replace `create_sample_data()` with reading from files
2. **Add more filters**: Add language detection, classifiers, etc.
3. **Use GPU acceleration**: Install `nemo-curator[text_cuda12]` and use GPU-accelerated classifiers
4. **Distributed processing**: Run on multi-node clusters to process TB-scale data

## Related Documentation

- NeMo Curator official documentation: https://docs.nvidia.com/nemo/curator/
- Full tutorials: `resources/nemo-curator/tutorials/`
- Architecture documentation: `ref/ray-architecture.md` and `ref/nemo-curator-problems-solved.md`
