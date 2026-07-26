# TraceRAG Retrieval Guide

Hybrid retrieval combines dense vector search with sparse BM25 keyword search. Dense
retrieval is useful when a question and a passage express similar meanings with different
words. BM25 is useful for exact technical terms, product names, identifiers, and error
codes.

TraceRAG combines the ranked result lists using reciprocal rank fusion. Reciprocal rank
fusion rewards passages that rank highly in either retrieval system without requiring
their raw scores to use the same scale.

Every indexed chunk retains its source filename and page number. The answer generator
uses this metadata to display citations. If the retrieved evidence is insufficient, the
system should abstain instead of inventing an answer.
