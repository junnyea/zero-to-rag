# Parameter Sweep Results: Chunk Size vs. Overlap

Evaluated on the ACME Corp gold dataset (`eval/gold.jsonl`) consisting of 1 Q&A cases.

| Chunk Size | Chunk Overlap | Hit@3 Rate | Hits | Mean Retrieved Chunk Length (chars) |
|---|---|---|---|---|
| 300 | 0 | **1.00** | 8/8 | 193.17 |
| 800 | 0 | **1.00** | 8/8 | 635.67 |
| 800 | 150 | **1.00** | 8/8 | 635.67 |
| 1500 | 0 | **1.00** | 8/8 | 1150.75 |
| 1500 | 150 | **1.00** | 8/8 | 1150.75 |
