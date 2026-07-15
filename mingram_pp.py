# -*- coding: utf-8 -*-
"""
MinGram-PP: A Unigram / PathPiece-style subword tokenizer.

References:
  - Kudo (2018): Subword Regularization (https://arxiv.org/abs/1804.10959)
  - Land (2026): PathPiece (https://arxiv.org/abs/2606.27019)

Two tokenizers are provided:
  - MinGramPPTokenizer – operates on string-level tokens
  - ByteMinGramPPTokenizer – operates on UTF-8 byte-level tokens with a
    FlatMatrixTrie for cache-friendly prefix matching.
"""

import math
import re
import collections
from typing import Dict, List, Optional, Set, Tuple

# =====================================================================
# STRING-LEVEL TOKENIZER
# =====================================================================


class MinGramPPTokenizer:
    """MinGram-PP tokenizer operating on string-level vocabulary tokens.

    Uses a character-level prefix trie for fast lattice construction and a
    Viterbi decoder that minimises token count, breaking ties by maximising
    joint log-probability.
    """

    def __init__(self) -> None:
        # Token string -> log probability
        self.vocab: Dict[str, float] = {}
        # Reverse mapping for decoding
        self.id_to_token: List[str] = []
        self.token_to_id: Dict[str, int] = {}

        # Character prefix trie for fast matching during lattice construction
        self.trie: dict = {}
        # Guaranteed atomic characters (every unique char seen in the corpus)
        self.atomics: Set[str] = set()

    # ------------------------------------------------------------------
    # Trie helpers
    # ------------------------------------------------------------------

    def _add_to_trie(self, token: str) -> None:
        """Insert *token* into the character prefix trie."""
        node = self.trie
        for char in token:
            if char not in node:
                node[char] = {}
            node = node[char]
        node["<EOS>"] = True  # marks a valid vocabulary token endpoint

    def _find_matches(self, text: str, start_idx: int) -> List[Tuple[str, int]]:
        """Return all vocabulary tokens that match *text* at *start_idx*.

        Each result is ``(matched_token_string, token_length)``.  If no
        vocabulary token matches, the single atomic character at *start_idx*
        is returned as a fallback.
        """
        matches: List[Tuple[str, int]] = []
        node = self.trie
        curr_idx = start_idx

        while curr_idx < len(text):
            char = text[curr_idx]
            if char not in node:
                break
            node = node[char]
            curr_idx += 1
            if "<EOS>" in node:
                matched_token = text[start_idx:curr_idx]
                matches.append((matched_token, curr_idx - start_idx))

        # Fallback to the atomic character
        if not matches and start_idx < len(text):
            atomic_char = text[start_idx]
            if atomic_char in self.vocab:
                matches.append((atomic_char, 1))

        return matches

    # ------------------------------------------------------------------
    # Vocabulary initialisation
    # ------------------------------------------------------------------

    def load_seed_vocabulary(
        self, candidate_counts: Dict[str, int], corpus_text: str
    ) -> None:
        """Initialise the vocabulary from an overshot BPE candidate set.

        Every unique character in *corpus_text* is registered as an atomic
        token, guaranteeing that any input can be segmented.
        """
        self.atomics = set(corpus_text)
        all_tokens = set(candidate_counts.keys()) | self.atomics

        total_count = (
            sum(candidate_counts.get(t, 0) for t in all_tokens) + len(self.atomics)
        )

        self.vocab = {}
        self.trie = {}

        for token in all_tokens:
            count = candidate_counts.get(token, 0)
            if token in self.atomics and count == 0:
                count = 1
            self.vocab[token] = math.log(count / total_count)
            self._add_to_trie(token)

        self.id_to_token = sorted(self.vocab.keys())
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

    # ------------------------------------------------------------------
    # Viterbi decoder
    # ------------------------------------------------------------------

    def _tokenize_chunk(self, chunk: str) -> List[str]:
        """Run the Viterbi lattice decoder on a short text *chunk*."""
        n = len(chunk)
        # dp[i] = (min_tokens, joint_log_prob, backpointer, matched_token)
        dp = [(float("inf"), float("-inf"), -1, "")] * (n + 1)
        dp[0] = (0, 0.0, -1, "")

        for i in range(n):
            if dp[i][0] == float("inf"):
                continue

            matches = self._find_matches(chunk, i)
            for token, length in matches:
                next_idx = i + length
                token_score = self.vocab[token]

                potential_tokens = dp[i][0] + 1
                potential_prob = dp[i][1] + token_score

                if potential_tokens < dp[next_idx][0]:
                    dp[next_idx] = (potential_tokens, potential_prob, i, token)
                elif potential_tokens == dp[next_idx][0]:
                    # Tie-break with joint log-probability
                    if potential_prob > dp[next_idx][1]:
                        dp[next_idx] = (potential_tokens, potential_prob, i, token)

        # Backward reconstruction
        tokens: List[str] = []
        curr = n
        while curr > 0:
            _, _, prev_idx, token = dp[curr]
            tokens.append(token)
            curr = prev_idx

        return tokens[::-1]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize *text* into the best Viterbi path.

        The input is first split into whitespace-delimited chunks to keep the
        DP array small.
        """
        if not text:
            return []

        chunks = re.findall(r"\s*\w+|\s+|[^\w\s]", text)
        final_tokens: List[str] = []
        for chunk in chunks:
            final_tokens.extend(self._tokenize_chunk(chunk))
        return final_tokens

    def encode(self, text: str) -> List[int]:
        """Convert *text* to a list of integer token IDs."""
        tokens = self.tokenize(text)
        return [self.token_to_id[t] for t in tokens if t in self.token_to_id]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back to a string."""
        return "".join(self.id_to_token[i] for i in ids)

    # ------------------------------------------------------------------
    # Hard-EM optimisation
    # ------------------------------------------------------------------

    def fit_em_step(self, corpus_sequences: List[str]) -> None:
        """Single Expectation-Maximisation step.

        E-step: tokenize the corpus with the current decoder and count token
        usage along the winning paths.
        M-step: re-estimate log-probabilities from the accumulated counts.
        Atomic tokens are given a count floor of 1.
        """
        token_counts: Dict[str, int] = collections.Counter()

        for text in corpus_sequences:
            if not text:
                continue
            chosen_tokens = self.tokenize(text)
            for token in chosen_tokens:
                token_counts[token] += 1

        all_tokens = set(self.vocab.keys())
        total_count = sum(token_counts.values()) + len(self.atomics)

        new_vocab: Dict[str, float] = {}
        self.trie = {}

        for token in all_tokens:
            count = token_counts.get(token, 0)
            if token in self.atomics and count == 0:
                count = 1

            if count > 0:
                new_vocab[token] = math.log(count / total_count)
                self._add_to_trie(token)
            else:
                new_vocab[token] = float("-inf")

        self.vocab = new_vocab
        self.id_to_token = sorted(self.vocab.keys())
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

    # ------------------------------------------------------------------
    # Iterative pruning  (PathPiece-style)
    # ------------------------------------------------------------------

    def train_prune(
        self,
        corpus_sequences: List[str],
        target_vocab_size: int,
        prune_factor: float = 0.2,
        em_steps_per_iter: int = 2,
    ) -> None:
        """Iteratively shrink the vocabulary down to *target_vocab_size*.

        Each iteration runs a few Hard-EM steps, sorts tokens by their
        log-probability, and prunes the lowest-scoring fraction
        (*prune_factor*).  Atomic tokens are never removed.
        """
        print(f"Starting training loop. Initial Seed Vocab Size: {len(self.vocab)}")

        while len(self.vocab) > target_vocab_size:
            for _ in range(em_steps_per_iter):
                self.fit_em_step(corpus_sequences)

            current_size = len(self.vocab)
            num_to_drop = int(current_size * prune_factor)

            if current_size - num_to_drop < target_vocab_size:
                num_to_drop = current_size - target_vocab_size

            if num_to_drop <= 0:
                break

            # Only non-atomic tokens are eligible for removal
            non_atomic_tokens = [
                t for t in self.vocab.keys() if t not in self.atomics
            ]
            non_atomic_tokens.sort(key=lambda t: self.vocab[t])  # type: ignore[arg-type]

            tokens_to_evict = set(non_atomic_tokens[:num_to_drop])

            pruned_vocab: Dict[str, float] = {}
            self.trie = {}

            for token, score in self.vocab.items():
                if token not in tokens_to_evict:
                    pruned_vocab[token] = score
                    self._add_to_trie(token)

            self.vocab = pruned_vocab
            self.id_to_token = sorted(self.vocab.keys())
            self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}

            print(f" Pruned {num_to_drop} tokens. Current Vocab Size: {len(self.vocab)}")

        # Final EM step to lock in stable probabilities
        self.fit_em_step(corpus_sequences)
        print(f"Training Complete! Final Vocab Size: {len(self.vocab)}")


# =====================================================================
# FLAT MATRIX TRIE  (byte-level, cache-friendly)
# =====================================================================


class FlatMatrixTrie:
    """A flattened 256-ary trie stored as a contiguous integer matrix.

    Each row corresponds to a trie state; columns are byte values (0–255).
    ``matrix[state * 256 + byte_val]`` gives the next state, or ``-1`` if no
    transition exists.  This layout is cache-friendly for Viterbi-style
    traversal.
    """

    ALPHABET_SIZE = 256

    def __init__(self, vocabulary_list: List[bytes]) -> None:
        # 1. Build a list of dictionary-based states first
        states: List[Dict[int, int]] = [{}]
        self.is_terminal: List[bool] = [False]

        for token_bytes in vocabulary_list:
            curr_state = 0
            for byte_val in token_bytes:
                if byte_val not in states[curr_state]:
                    states[curr_state][byte_val] = len(states)
                    states.append({})
                    self.is_terminal.append(False)
                curr_state = states[curr_state][byte_val]
            self.is_terminal[curr_state] = True

        # 2. Flatten into a contiguous 1-D matrix
        num_states = len(states)
        self.matrix: List[int] = [-1] * (num_states * self.ALPHABET_SIZE)

        for state_idx in range(num_states):
            for byte_val, next_state_idx in states[state_idx].items():
                flat_idx = state_idx * self.ALPHABET_SIZE + byte_val
                self.matrix[flat_idx] = next_state_idx

    def find_all_matches(self, byte_seq: bytes, start_idx: int) -> List[int]:
        """Return a list of token *lengths* that match at *start_idx*."""
        matches: List[int] = []
        curr_state = 0
        curr_idx = start_idx
        seq_len = len(byte_seq)

        while curr_idx < seq_len:
            byte_val = byte_seq[curr_idx]
            flat_idx = curr_state * self.ALPHABET_SIZE + byte_val
            next_state = self.matrix[flat_idx]

            if next_state == -1:
                break

            curr_state = next_state
            curr_idx += 1

            if self.is_terminal[curr_state]:
                matches.append(curr_idx - start_idx)

        return matches


# =====================================================================
# BYTE-LEVEL TOKENIZER
# =====================================================================


class ByteMinGramPPTokenizer:
    """MinGram-PP tokenizer operating on UTF-8 byte tokens.

    All 256 individual byte values are registered as atomic tokens, so the
    tokenizer can handle *any* input including non-ASCII / non-printable
    characters.  A :class:`FlatMatrixTrie` is used internally for fast prefix
    matching.
    """

    def __init__(self) -> None:
        # Byte token -> log probability
        self.vocab: Dict[bytes, float] = {}
        self.id_to_token: List[bytes] = []
        self.token_to_id: Dict[bytes, int] = {}

        self.trie: dict = {}
        # All 256 individual byte values are atomic safety nets
        self.atomics: Set[bytes] = {bytes([i]) for i in range(256)}

        # Lazily initialised flat matrix trie
        self.flat_trie: Optional[FlatMatrixTrie] = None

    def _add_to_trie(self, token_bytes: bytes) -> None:
        """Insert *token_bytes* into the byte-level prefix trie."""
        node = self.trie
        for byte_val in token_bytes:
            if byte_val not in node:
                node[byte_val] = {}
            node = node[byte_val]
        node["<EOS>"] = True

    def load_seed_vocabulary(self, candidate_counts: Dict[bytes, int]) -> None:
        """Initialise the vocabulary from a set of byte-token candidates.

        All 256 single-byte values are unconditionally protected as atomics.
        """
        all_tokens = set(candidate_counts.keys()) | self.atomics
        total_count = sum(candidate_counts.get(t, 0) for t in all_tokens) + 256

        self.vocab = {}
        self.trie = {}

        for token in all_tokens:
            count = candidate_counts.get(token, 0)
            if token in self.atomics and count == 0:
                count = 1
            self.vocab[token] = math.log(count / total_count)
            self._add_to_trie(token)

        self.id_to_token = sorted(self.vocab.keys())
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.flat_trie = None

    # ------------------------------------------------------------------
    # Flat matrix trie matching
    # ------------------------------------------------------------------

    def _find_matches(self, byte_seq: bytes, start_idx: int) -> List[Tuple[bytes, int]]:
        """Find all vocabulary tokens matching *byte_seq* at *start_idx*.

        Uses the :class:`FlatMatrixTrie` internally and lazily builds it on
        first access.
        """
        if self.flat_trie is None:
            self.flat_trie = FlatMatrixTrie(self.id_to_token)

        lengths = self.flat_trie.find_all_matches(byte_seq, start_idx)
        matches = [(byte_seq[start_idx : start_idx + l], l) for l in lengths]

        if not matches and start_idx < len(byte_seq):
            matches.append((byte_seq[start_idx : start_idx + 1], 1))

        return matches

    # ------------------------------------------------------------------
    # Viterbi decoder
    # ------------------------------------------------------------------

    def _tokenize_byte_chunk(self, chunk_bytes: bytes) -> List[bytes]:
        """Run the Viterbi lattice decoder on a byte chunk."""
        n = len(chunk_bytes)
        if n == 0:
            return []

        # dp[i] = (min_tokens, joint_log_prob, backpointer, matched_token)
        dp = [(float("inf"), float("-inf"), -1, b"")] * (n + 1)
        dp[0] = (0, 0.0, -1, b"")

        for i in range(n):
            if dp[i][0] == float("inf"):
                continue

            matches = self._find_matches(chunk_bytes, i)
            for token_bytes, length in matches:
                next_idx = i + length
                token_score = self.vocab[token_bytes]

                potential_tokens = dp[i][0] + 1
                potential_prob = dp[i][1] + token_score

                if potential_tokens < dp[next_idx][0]:
                    dp[next_idx] = (potential_tokens, potential_prob, i, token_bytes)
                elif potential_tokens == dp[next_idx][0]:
                    if potential_prob > dp[next_idx][1]:
                        dp[next_idx] = (potential_tokens, potential_prob, i, token_bytes)

        tokens: List[bytes] = []
        curr = n
        while curr > 0:
            _, _, prev_idx, token_bytes = dp[curr]
            tokens.append(token_bytes)
            curr = prev_idx

        return tokens[::-1]

    def tokenize(self, text: str) -> List[bytes]:
        """Tokenize *text* into byte tokens via Viterbi decoding.

        The text is first split into chunks so that the DP array for each
        chunk stays small.
        """
        if not text:
            return []

        chunks = re.findall(r"\s*\w+|\s+|[^\w\s]", text)
        final_tokens: List[bytes] = []
        for chunk in chunks:
            chunk_bytes = chunk.encode("utf-8")
            final_tokens.extend(self._tokenize_byte_chunk(chunk_bytes))
        return final_tokens

    def encode(self, text: str) -> List[int]:
        """Convert *text* to a list of integer token IDs."""
        tokens = self.tokenize(text)
        return [self.token_to_id[t] for t in tokens if t in self.token_to_id]

    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back to a string."""
        byte_stream = b"".join(self.id_to_token[i] for i in ids)
        return byte_stream.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Special-token support
    # ------------------------------------------------------------------

    def encode_with_special_tokens(
        self, text: str, allowed_special: Optional[Set[str]] = None
    ) -> List[int]:
        """Encode *text*, preserving protected special tokens as single IDs.

        Special tokens (e.g. ``<|endoftext|>``) are matched via regex and
        looked up directly, bypassing the byte-level Viterbi decoder.
        """
        if not text:
            return []

        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}

        special_pattern = "|".join(re.escape(tok) for tok in allowed_special)
        if not special_pattern:
            return self.encode(text)

        parts = re.split(f"({special_pattern})", text)

        final_ids: List[int] = []
        for part in parts:
            if part in allowed_special:
                final_ids.append(self.token_to_id[part.encode("utf-8")])
            else:
                final_ids.extend(self.encode(part))

        return final_ids

    # ------------------------------------------------------------------
    # Hard-EM optimisation
    # ------------------------------------------------------------------

    def fit_em_step_bytes(self, corpus_sequences: List[str]) -> None:
        """Single Hard-EM step for the byte-level tokenizer.

        Token usage is accumulated along winning Viterbi paths, then
        log-probabilities are re-estimated.  The flat matrix trie cache is
        cleared so it is rebuilt on the next tokenization call.
        """
        token_counts: Dict[bytes, int] = collections.Counter()

        for text in corpus_sequences:
            chosen_tokens = self.tokenize(text)
            for token in chosen_tokens:
                token_counts[token] += 1

        all_tokens = set(self.vocab.keys())
        total_count = sum(token_counts.values()) + len(self.atomics)

        new_vocab: Dict[bytes, float] = {}
        for token in all_tokens:
            count = token_counts.get(token, 0)
            if token in self.atomics and count == 0:
                count = 1

            if count > 0:
                new_vocab[token] = math.log(count / total_count)
            else:
                new_vocab[token] = float("-inf")

        self.vocab = new_vocab
        self.id_to_token = sorted(self.vocab.keys())
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.flat_trie = None

    # ------------------------------------------------------------------
    # Iterative pruning (byte-level)
    # ------------------------------------------------------------------

    def train_prune_bytes(
        self,
        corpus_sequences: List[str],
        target_vocab_size: int,
        prune_factor: float = 0.2,
        em_steps_per_iter: int = 2,
    ) -> None:
        """Iteratively shrink the byte vocabulary to *target_vocab_size*."""
        print(
            f"Starting Byte Training. Initial Vocabulary Candidates: {len(self.vocab)}"
        )

        while len(self.vocab) > target_vocab_size:
            for _ in range(em_steps_per_iter):
                self.fit_em_step_bytes(corpus_sequences)

            current_size = len(self.vocab)
            num_to_drop = int(current_size * prune_factor)

            if current_size - num_to_drop < target_vocab_size:
                num_to_drop = current_size - target_vocab_size

            if num_to_drop <= 0:
                break

            non_atomic_tokens = [
                t for t in self.vocab.keys() if t not in self.atomics
            ]
            non_atomic_tokens.sort(key=lambda t: self.vocab[t])  # type: ignore[arg-type]

            tokens_to_evict = set(non_atomic_tokens[:num_to_drop])

            pruned_vocab: Dict[bytes, float] = {}
            for token, score in self.vocab.items():
                if token not in tokens_to_evict:
                    pruned_vocab[token] = score

            self.vocab = pruned_vocab
            self.id_to_token = sorted(self.vocab.keys())
            self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
            self.flat_trie = None

            print(f" Pruned {num_to_drop} byte tokens. Current Size: {len(self.vocab)}")

        self.fit_em_step_bytes(corpus_sequences)
        print(f"Training Complete! Final Clean Matrix Size: {len(self.vocab)}")
