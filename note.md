[Notes are refined using Gemini]

The Unigram algorithm is a subword segmentation method that treats tokenization as a probabilistic inference problem. It operates on the fundamental assumption that each subword in a sequence occurs independently of the others. Because of this independence assumption, the probability of a specific subword sequence is calculated simply as the product of the individual probabilities of each subword in that sequence.

The mathematical framework of the algorithm relies on a core probability formula:

$$P(x) = \prod_{i=1}^M p(x_i)$$

Subject to the constraint that for all subwords in the vocabulary $V$:

$$\sum_{x \in V} p(x) = 1$$

**The Constraint ($\sum p(x) = 1$)** ensures that the probabilities of all tokens within the vocabulary $V$ sum up to exactly 1, making it a mathematically valid probability distribution.    

To build its vocabulary, Unigram estimates subword probabilities using the Expectation-Maximization (EM) algorithm, which maximizes the overall likelihood of the training text corpus.

The training process itself uses iterative pruning. It typically starts with an excessively large "seed" vocabulary generated via methods like suffix-arrays or Byte-Pair Encoding (BPE). It then systematically removes the tokens that contribute the least to the overall likelihood of the corpus until it reaches the target vocabulary size.

When it comes to processing new text, the algorithm needs a way to find the most likely segmentation. It achieves this by using Viterbi decoding, an algorithm designed to efficiently find the path with the highest total probability through all possible subword combinations.

#### **Unigram vs. Traditional Tokenization (BPE)**

Standard subword tokenization, most notably Byte-Pair Encoding (BPE), differs fundamentally from Unigram across several dimensions.

**Deterministic vs. Probabilistic Approaches**

BPE is a greedy, deterministic algorithm. It builds words from the bottom up by merging the most frequent adjacent character pairs, resulting in exactly one unique subword sequence for any given input. Unigram, by contrast, is fully probabilistic. It looks at a sentence and can see multiple valid ways to split it, calculating an explicit probability score for each variation.

**Encoding Principles and Structure**

BPE acts as a "dictionary encoder" designed to minimize the total number of unique symbols. Unigram operates as an "entropy encoder," aiming to minimize total code length based on Shannon’s coding theorem. This creates distinct structural differences: a Unigram vocabulary is a simple, flat list of tokens paired with probability scores, whereas BPE vocabularies are chained to a strict, ordered merge list. If you remove a single token from a BPE vocabulary, you break the downstream merge dependencies; removing a token from Unigram requires no such recalculation.

**Inference Flexibility**

While BPE is locked into fixed merge rules during inference, Unigram can utilize various inference methods. It can use standard marginal EM or Hard EM on a minimum-token path, depending on the specific variant being deployed.

#### Problems Solved by Unigram

The Unigram algorithm directly addresses the rigid limitations inherent to deterministic models.

**Subword Regularization**

Because Unigram inherently calculates multiple valid segmentations for a single sentence, it enables a technique called subword regularization. During training, the model can sample different random segmentations for the identical input phrase. This introduces beneficial noise and acts as data augmentation, making the final machine learning model far more robust against typos, noise, and segmentation errors.

**Morphological Alignment and Vocabulary Efficiency**

Unigram-family tokenizers frequently produce tokens that align better with actual human linguistics and word roots. Deterministic models like BPE are prone to "fragmented," nonsensical splits based purely on character frequency. Furthermore, Unigram offers superior vocabulary management; you can easily prune low-value entries or merge different vocabularies because you are not bound by a rigid, historical merge-order.

#### Examples

**1. Multiple Segmentations for One Input**

Consider the phrase "Hello World." A standard BPE tokenizer will always output a single sequence. Unigram recognizes that a vocabulary can encode this text in several valid ways:

- **Variant A:** Hell / o / world
- **Variant B:** H / ello / world
- **Variant C:** He / llo / world

By exposing a neural network to these different variations during training, the model learns to recognize the core text regardless of how it is sliced.

**2. Learning Word Compositionality**

This probabilistic splitting helps models naturally grasp how words are built. If a model only ever sees the word "books" as a single token, it treats it as completely distinct from "book." However, if Unigram occasionally segments it as "book" + "s," a translation or language model quickly learns that the trailing "s" is a distinct unit signifying a plural.

**3. Alignment Comparison**

The structural differences become obvious when observing how different algorithms split common English words:

|**Word**|**Gold Standard**|**BPE (Deterministic)**|**Unigram (Probabilistic)**|
|---|---|---|---|
|**outspoken**|out / spoken|out / spoken|outs / po / ken|
|**spaceship**|space / ship|sp / aces / hip|space / ship|
|**southeast**|south / east|s / outheast|south / east|
|**coloured**|colour / ed|col / oured|colour / ed|

In the "spaceship" example, BPE's greedy approach results in a linguistically messy split (_sp / aces / hip_), while Unigram naturally retains the morphologically clean and logical tokens (_space / ship_).

**4. Graceful Handling of Rare Terms**

Unigram prevents rare compound words from being completely shattered into meaningless character fragments. For a complex string like "ArticlePubMedGoogle," a standard Unigram model will likely divide it into three clean, recognizable components: "Article," "PubMed," and "Google." This preserves the semantic integrity of the words far better than breaking them down into raw, isolated letters.
