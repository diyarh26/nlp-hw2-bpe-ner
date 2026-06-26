---
phase: 02-fast-encode-training-efficiency
created: 2026-06-25
source: inline-gsd-plan-phase
status: complete
---

# Phase 2 Research

## Existing Code Findings

- `BPETokenizer.train(texts)` currently converts each capped input line to a character/space-marker sequence, repeatedly recounts all legal adjacent pairs with `_count_pairs`, then applies one merge globally with `_apply_pair_to_sequences`.
- `BPETokenizer.encode(text)` converts text to base symbols, then `_apply_ranked_merges` repeatedly scans all adjacent pairs to find the lowest learned merge rank. This is correct and deterministic, but it has no cache and rebuilds lists with slicing on each applied merge.
- `__getstate__` already drops `_encode_cache` and compiled regex state, so adding a real runtime cache can preserve pickle safety if it remains runtime-only.
- `generate_tokenizers.py` reads the full training files into memory and passes lists into `BPETokenizer.train`; Phase 2 cannot stream from the generator without editing a forbidden file.

## Recommended Encode Approach

- Keep `merge_ranks: Dict[Tuple[str, str], int]` as the source of truth.
- Add an uncached helper for behavior:
  - Convert text to encode symbols.
  - Apply learned merges by rank.
  - Convert final symbols to ids, using byte fallback for unseen symbols.
- Have public `encode(text)` check `_encode_cache` first and store a tuple/list copy of ids after the uncached helper.
- Preserve current cross-space behavior by caching whole input strings first. A future refinement can add per-word/chunk partial caching, but whole-string caching is the safe Phase 2 slice because it cannot suppress legal cross-word merges.
- Add measurable repeated-call benchmark evidence. Even full-string caching is valuable for repeated NER/dev/checker calls and proves the runtime cache path works without changing outputs.

## Recommended Training Approach

- Convert input texts to weighted unique sequences:
  - Count duplicate text lines with `Counter(texts)`.
  - Convert each unique non-empty line to one mutable symbol sequence.
  - Store a parallel integer weight per sequence.
- Build pair indexes once:
  - `pair_counts[pair] += weight * occurrences_in_sequence`.
  - `pair_to_sequences[pair].add(sequence_id)` for sequences containing the pair.
- For each merge:
  - Select the legal pair with highest weighted count and deterministic tie-break `(count desc, left asc, right asc)`.
  - Only update sequences from `pair_to_sequences[pair]`.
  - For each impacted sequence, subtract all old weighted pair contributions, apply the merge, then add new weighted pair contributions.
  - Drop zero/negative pair counts and empty index sets lazily.
- This is incremental because each merge updates only impacted sequences instead of full recounting every sequence.

## Verification Risks

- Cache must not be serialized. Re-run a pickle probe and inspect the pickled state does not include `_encode_cache`.
- Incremental pair indexes are easy to corrupt. Keep a small-sample oracle check that compares final vocab/merges against the old full-recount trainer for a deterministic fixture.
- Removing the training cap may increase memory. The generator already holds text lists, so the trainer should avoid unnecessary duplicate line copies beyond weighted unique sequences.
- Full checker runs are slow but mandatory before closing Phase 2 because training and encode changes can break alignment silently.

