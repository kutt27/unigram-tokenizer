/*
 * mingram_c.c — C extension for MinGram-PP hot loops.
 *
 * Implements:
 *   1. FlatMatrixTrie construction and prefix matching
 *   2. Viterbi byte-chunk decoder
 *
 * Interface via ctypes from Python.
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

#define ALPHABET_SIZE 256

/* ── FlatMatrixTrie ─────────────────────────────────────────────── */

/*
 * Build a flat 256-ary trie from a list of byte tokens.
 *
 * Parameters:
 *   tokens_flat    – all token bytes concatenated
 *   token_offsets  – start index of each token in tokens_flat (length num_tokens+1,
 *                    last entry = total bytes length)
 *   num_tokens     – number of tokens
 *   matrix_out     – pre-allocated output: num_states * 256 (caller must guess max;
 *                    function writes actual num_states via num_states_out)
 *   is_terminal_out – pre-allocated output: num_states booleans
 *   max_states     – capacity of matrix_out / 256 and is_terminal_out
 *   num_states_out – [out] actual number of states used
 *
 * Returns 0 on success, -1 if max_states exceeded.
 */
int trie_build(
    const uint8_t *tokens_flat,
    const int *token_offsets,
    int num_tokens,
    int32_t *matrix_out,
    uint8_t *is_terminal_out,
    int max_states,
    int *num_states_out
) {
    /* Phase 1: build dictionary-based trie to count states */
    /* We use a compact adjacency representation:
       states[state][byte_val] -> next_state, stored as flat arrays. */
    int cap = max_states;
    int *states = (int *)malloc(cap * ALPHABET_SIZE * sizeof(int));
    uint8_t *terminal = (uint8_t *)malloc(cap * sizeof(uint8_t));
    if (!states || !terminal) {
        free(states); free(terminal);
        return -1;
    }
    memset(states, -1, cap * ALPHABET_SIZE * sizeof(int));
    memset(terminal, 0, cap * sizeof(uint8_t));

    int num_states = 1; /* state 0 is root */

    for (int t = 0; t < num_tokens; t++) {
        int start = token_offsets[t];
        int end = token_offsets[t + 1];
        int curr = 0;
        for (int i = start; i < end; i++) {
            uint8_t b = tokens_flat[i];
            int idx = curr * ALPHABET_SIZE + b;
            if (states[idx] == -1) {
                if (num_states >= cap) {
                    /* Realloc — rare, but needed for large vocabs */
                    int new_cap = cap * 2;
                    int *new_s = (int *)malloc(new_cap * ALPHABET_SIZE * sizeof(int));
                    uint8_t *new_t = (uint8_t *)malloc(new_cap * sizeof(uint8_t));
                    if (!new_s || !new_t) {
                        free(new_s); free(new_t);
                        free(states); free(terminal);
                        return -1;
                    }
                    memcpy(new_s, states, cap * ALPHABET_SIZE * sizeof(int));
                    memset(new_s + cap * ALPHABET_SIZE, -1,
                           (new_cap - cap) * ALPHABET_SIZE * sizeof(int));
                    memcpy(new_t, terminal, cap * sizeof(uint8_t));
                    memset(new_t + cap, 0, (new_cap - cap) * sizeof(uint8_t));
                    free(states); free(terminal);
                    states = new_s;
                    terminal = new_t;
                    cap = new_cap;
                    idx = curr * ALPHABET_SIZE + b;
                }
                states[idx] = num_states;
                num_states++;
            }
            curr = states[idx];
        }
        terminal[curr] = 1;
    }

    /* Copy to output arrays */
    memcpy(matrix_out, states, num_states * ALPHABET_SIZE * sizeof(int));
    memcpy(is_terminal_out, terminal, num_states * sizeof(uint8_t));
    *num_states_out = num_states;

    free(states);
    free(terminal);
    return 0;
}

/*
 * Find all token lengths matching at start_idx in byte_seq.
 *
 * Returns number of matches written to lengths_out.
 */
int trie_find_matches(
    const int32_t *matrix,
    const uint8_t *is_terminal,
    int num_states,
    const uint8_t *byte_seq,
    int seq_len,
    int start_idx,
    int *lengths_out,
    int max_matches
) {
    int num_matches = 0;
    int curr_state = 0;
    int idx = start_idx;

    while (idx < seq_len) {
        uint8_t b = byte_seq[idx];
        int flat = curr_state * ALPHABET_SIZE + b;
        if (flat >= num_states * ALPHABET_SIZE) break;
        int next = matrix[flat];
        if (next == -1) break;
        curr_state = next;
        idx++;
        if (is_terminal[curr_state]) {
            if (num_matches < max_matches) {
                lengths_out[num_matches] = idx - start_idx;
            }
            num_matches++;
        }
    }
    return num_matches;
}

/* ── Viterbi Decoder ────────────────────────────────────────────── */

/*
 * Tokenize a byte chunk using Viterbi DP.
 *
 * Parameters:
 *   byte_seq, seq_len   – input byte sequence
 *   matrix, is_terminal, num_states – trie
 *   vocab_scores        – log-prob for each token (indexed by token id)
 *   token_ids_by_length – for each trie match length, the token id
 *                         (built during trie construction; see trie_build_with_ids)
 *   tokens_out          – output array of token start positions and lengths
 *   max_tokens          – capacity of tokens_out
 *   num_tokens_out      – [out] number of tokens decoded
 *
 * tokens_out[i*2] = start offset in byte_seq
 * tokens_out[i*2+1] = length
 *
 * Returns 0 on success.
 */
int viterbi_tokenize(
    const uint8_t *byte_seq,
    int seq_len,
    const int32_t *matrix,
    const uint8_t *is_terminal,
    int num_states,
    const float *vocab_scores,
    /* Map from (state, byte) -> token_id for terminal states.
       We encode this as: for each terminal state, store the token_id.
       But multiple tokens can end at the same state? No — in a trie,
       each state corresponds to at most one token (the path to it is unique).
       So terminal_token_id[state] = token_id or -1. */
    const int *terminal_token_id,
    int *tokens_out,
    int max_tokens,
    int *num_tokens_out
) {
    /* dp[i] = (min_tokens, max_log_prob, backpointer, token_len) */
    struct DPState {
        int min_tokens;
        float max_logprob;
        int backptr;
        int tok_len;    /* length of token that led here */
        int tok_id;     /* token id */
    };

    struct DPState *dp = (struct DPState *)malloc((seq_len + 1) * sizeof(struct DPState));
    if (!dp) return -1;

    for (int i = 0; i <= seq_len; i++) {
        dp[i].min_tokens = INT32_MAX;
        dp[i].max_logprob = -FLT_MAX;
        dp[i].backptr = -1;
        dp[i].tok_len = 0;
        dp[i].tok_id = -1;
    }
    dp[0].min_tokens = 0;
    dp[0].max_logprob = 0.0f;

    /* Match buffer */
    int max_matches = 256;
    int *lengths = (int *)malloc(max_matches * sizeof(int));
    if (!lengths) { free(dp); return -1; }

    for (int i = 0; i < seq_len; i++) {
        if (dp[i].min_tokens == INT32_MAX) continue;

        /* Find all matches at position i */
        int nmatches = trie_find_matches(
            matrix, is_terminal, num_states,
            byte_seq, seq_len, i,
            lengths, max_matches
        );
        if (nmatches > max_matches) nmatches = max_matches;

        for (int m = 0; m < nmatches; m++) {
            int length = lengths[m];
            int next = i + length;

            /* Find the token_id for this match by walking the trie to the
               terminal state at this length. We re-walk to get the state. */
            int state = 0;
            int tok_id = -1;
            for (int k = i; k < next; k++) {
                state = matrix[state * ALPHABET_SIZE + byte_seq[k]];
            }
            if (state >= 0 && state < num_states && is_terminal[state]) {
                tok_id = terminal_token_id[state];
            }
            if (tok_id < 0) continue;

            float score = vocab_scores[tok_id];
            int pot_tokens = dp[i].min_tokens + 1;
            float pot_prob = dp[i].max_logprob + score;

            if (pot_tokens < dp[next].min_tokens ||
                (pot_tokens == dp[next].min_tokens && pot_prob > dp[next].max_logprob)) {
                dp[next].min_tokens = pot_tokens;
                dp[next].max_logprob = pot_prob;
                dp[next].backptr = i;
                dp[next].tok_len = length;
                dp[next].tok_id = tok_id;
            }
        }
    }

    /* Backtrack */
    int stack[4096]; /* max tokens */
    int ntok = 0;
    int pos = seq_len;
    while (pos > 0 && ntok < 4096) {
        stack[ntok] = pos;
        pos = dp[pos].backptr;
        ntok++;
    }

    /* Write output (reversed) */
    int out = 0;
    for (int i = ntok - 1; i >= 0 && out < max_tokens; i--) {
        int end = stack[i];
        int start = dp[end].backptr;
        tokens_out[out * 2] = start;
        tokens_out[out * 2 + 1] = end - start;
        out++;
    }
    *num_tokens_out = out;

    free(dp);
    free(lengths);
    return 0;
}

/* ── Batch Viterbi (for EM) ─────────────────────────────────────── */

/*
 * Tokenize multiple chunks in one call, returning token IDs directly.
 * This avoids Python loop overhead during EM steps.
 *
 * Parameters:
 *   chunks_flat         – all chunk bytes concatenated
 *   chunk_offsets       – start index of each chunk (length num_chunks+1)
 *   num_chunks          – number of chunks
 *   matrix, is_terminal, num_states – trie
 *   vocab_scores        – log-prob per token id
 *   terminal_token_id   – token id per trie state
 *   token_counts_out    – [out] count per token id (length = vocab_size)
 *   vocab_size          – size of vocabulary
 *   total_tokens_out    – [out] total tokens emitted
 *
 * Returns 0 on success.
 */
int viterbi_batch_count(
    const uint8_t *chunks_flat,
    const int *chunk_offsets,
    int num_chunks,
    const int32_t *matrix,
    const uint8_t *is_terminal,
    int num_states,
    const float *vocab_scores,
    const int *terminal_token_id,
    int *token_counts_out,
    int vocab_size,
    int *total_tokens_out
) {
    memset(token_counts_out, 0, vocab_size * sizeof(int));
    int total = 0;

    int max_tok_per_chunk = 4096;
    int *tok_buf = (int *)malloc(max_tok_per_chunk * 2 * sizeof(int));
    if (!tok_buf) return -1;

    for (int c = 0; c < num_chunks; c++) {
        int start = chunk_offsets[c];
        int end = chunk_offsets[c + 1];
        int len = end - start;
        if (len <= 0) continue;

        int ntok = 0;
        int rc = viterbi_tokenize(
            chunks_flat + start, len,
            matrix, is_terminal, num_states,
            vocab_scores, terminal_token_id,
            tok_buf, max_tok_per_chunk, &ntok
        );
        if (rc != 0) continue;

        /* Count tokens */
        for (int t = 0; t < ntok; t++) {
            /* Re-derive token id by walking trie */
            int tstart = tok_buf[t * 2];
            int tlen = tok_buf[t * 2 + 1];
            int state = 0;
            for (int k = tstart; k < tstart + tlen; k++) {
                state = matrix[state * ALPHABET_SIZE + (chunks_flat + start)[k]];
            }
            if (state >= 0 && state < num_states && is_terminal[state]) {
                int tid = terminal_token_id[state];
                if (tid >= 0 && tid < vocab_size) {
                    token_counts_out[tid]++;
                    total++;
                }
            }
        }
    }

    *total_tokens_out = total;
    free(tok_buf);
    return 0;
}