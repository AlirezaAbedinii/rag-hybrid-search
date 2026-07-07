# Comparison report

## Hybrid vs dense-only

| Metric | Dense | Hybrid |
|---|---|---|
| Answer correctness (mean) | 0.769 | 0.896 |
| Faithfulness (mean) | 1.000 | 1.000 |
| Retrieval relevance | 1.000 | 1.000 |
| Citation accuracy | 0.932 | 0.975 |
| Refusals (of no-answer set) | 0 | 9 |
| Mean cost / query (USD) | 0.000301 | 0.000196 |
| P95 total latency (ms) | 7320.8 | 8024.6 |

## Chunking strategies

| Metric | Fixed | Recursive | Semantic |
|---|---|---|---|
| Answer correctness (mean) | 0.896 | 0.891 | 0.891 |
| Faithfulness (mean) | 1.000 | 1.000 | 1.000 |
| Retrieval relevance | 1.000 | 1.000 | 1.000 |
| Citation accuracy | 0.975 | 0.988 | 0.954 |
| Refusals (of no-answer set) | 9 | 9 | 9 |
| Mean cost / query (USD) | 0.000196 | 0.000197 | 0.000194 |
| P95 total latency (ms) | 6747.8 | 6838.5 | 5096.8 |
