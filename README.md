Papers referred: 
- [Kudo](https://arxiv.org/abs/1804.10959)
- [Land](https://arxiv.org/abs/2606.27019)

Hardware used for training

- TPU: v5e-1 TPU

## Benchmark: MinGram-PP vs SentencePiece

WikiText-2 validation set (1.1M bytes, vocab=8000)

| Metric | SentencePiece | MinGram-PP |
|---|---|---|
| Bytes/Token | 3.70 | 3.51 |
| Total Tokens | 300,255 | 316,484 |
| Tokenize Time | 0.1s | 2.3s |
| Tokens/sec | 3.25M | 135K |

- SentencePiece trained in 1.1s (C++, suffix array seed generation)
- MinGram-PP trained in 547s (pure Python, brute-force substring mining)
- Both achieve comparable compression quality
- Speed gap is in seed generation, not the core Viterbi + EM algorithm

