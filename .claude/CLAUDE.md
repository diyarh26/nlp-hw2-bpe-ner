<!-- GSD:project-start source:PROJECT.md -->

## Project

**BPE Tokenizer for Cross-Domain NER (NLP HW2)**

A from-scratch Byte Pair Encoding (BPE) tokenizer, implemented in `code/bpe_tokenizer.py` as a subclass of the course-provided `BaseTokenizer`, that feeds a **fixed** binary NER model (entity vs. non-entity, word-level F1). We deliver **three** trained tokenizers — one tuned for a Twitter/social domain, one for a news/newswire domain, and one designed to generalize to a **hidden** third domain — and a ≤1-page report. This is a graded university NLP homework (HW2); the deliverable is a single `HW2_<id>.zip` submission.

**Core Value:** Three correct, reproducible BPE tokenizers that (a) pass every hard format/spec gate so the submission is not zeroed or disqualified, and (b) score well on the competition's three equally-weighted axes — **tokenizer efficiency (tokens/sentence), tokenization speed, and NER F1** — including on the hidden domain.

### Constraints

- **Tech stack**: Python 3.11–3.12, `uv` environment; libs limited to numpy, regex, torch, tqdm (no additions). Pure-Python/numpy BPE.
- **Compute**: Develop & unit-test the tokenizer locally on Windows (CPU). Run NER training / F1 / timing on the **course VM** (Tesla M60 GPU) — **10-hour total runtime budget**, so debug locally first and only use the VM when GPU is needed (`time_left` shows remaining budget).
- **Reproducibility**: Tokenizers must be byte-reproducible from `code/` + `train_commands.txt` via `generate_tokenizers.py` (seeded).
- **Format**: Any file-structure mismatch = grade 0. Report format rules are strict.
- **Integrity**: AI tools allowed, but code is the student's responsibility; no sharing of prompts/AI snippets or code between students.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->

## Technology Stack

## Recommended Stack

### Core Technologies (algorithms / data structures)

| Technology | "Version" / variant | Purpose | Why Recommended |
|------------|---------------------|---------|-----------------|
| **Word-frequency dict + symbol-list representation** | dedup chunks, count multiplier | TRAINING corpus representation | Identical chunks ("the", "in the") collapse to one entry with a weight. Turns a 95MB corpus into a few-hundred-K unique chunks. All pair counting then scales with *unique* chunks × weight, not raw tokens. This is the single biggest training win. |
| **Incremental pair-count deltas + inverted index (pair → chunk ids)** | max-count selection per round | TRAINING merge loop | After each merge of `(b,c)→e`, only re-touch chunks that contained `(b,c)`. Decrement the 3 disappearing pairs, increment the 2 new pairs. Avoids O(corpus) recount per merge → O(chunks_containing_pair). ~100–1000× faster than naive recount in pure Python. |
| **Merge-rank dict `{(tok_a, tok_b): rank}` + greedy lowest-rank apply** | tiktoken/`_educational.py` style | ENCODE | Encode is "apply learned merges in training order." Store each merge's rank (= creation order). At encode time, repeatedly merge the adjacent pair with the **lowest rank** until none qualify. This is exactly what tiktoken's reference encoder does. Deterministic, matches training, and the rank dict is a plain picklable `dict`. |
| **Per-word (per-chunk) encode cache** `{chunk_str: List[int]}` | plain dict, lazily filled | ENCODE | Real text is Zipfian — a handful of chunks dominate. Memoizing the token-id list per pre-tokenized chunk turns most of a sentence into dict hits. Often the largest real-world encode speedup, larger than micro-optimizing the merge loop. Must be an instance attribute that pickles cleanly (a normal dict does). |
| **`regex`-based pre-tokenization into chunks (incl. an explicit space marker)** | GPT-2-like split, space-aware | TRAIN + ENCODE | Split text into chunks once, encode each chunk independently, concatenate. Bounds the inner merge loop to short chunks (fast, near-linear in practice) instead of whole-sentence O(n²) rescans. The space marker (e.g. `▁`) is emitted as a real symbol so merges may legally cross it (see bigram section). |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `regex` (third-party `regex`, not stdlib `re`) | env-pinned | Pre-tokenization pattern (handles Unicode, contractions, URLs/@mentions for the Twitter domain) | Always — it's the only allowed regex engine and is needed for robust splitting across the three domains. |
| `collections.Counter` | stdlib | Initial bulk pair count of the whole corpus before the incremental loop starts | Use ONCE to seed pair counts; do NOT use it to recount every round. |
| `heapq` | stdlib | Optional max-heap (push negative counts) to pick the top pair without scanning the whole pair dict each round | Use if the pair dict is large and per-round `max()` becomes the bottleneck. Lazy-deletion pattern (re-validate popped entry against the live count) avoids the "stale heap entry" bug. |
| `numpy` | env-pinned | Optional: vectorize bulk initial pair counting, or store flat int arrays for chunk symbol sequences | Use for the initial count / heavy bulk ops; the incremental merge loop stays in Python dicts (numpy doesn't help with the irregular per-merge updates). |
| `tqdm` | env-pinned | Progress bars over the merge loop / corpus pass | Dev ergonomics only; wrap the merge-count loop so long trainings are observable. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `pickle` round-trip test | Verify the trained tokenizer unpickles from `code/` alone | Add a unit test: train tiny → `pickle.dumps` → `pickle.loads` in a fresh process → encode/decode still works. Catches lambda/closure leaks early. |
| Decode length-faithfulness test | Verify NER alignment invariant | Assert `len(decode(ids[:i+1])) - len(decode(ids[:i]))` reconstructs original char spans for sample sentences. This is the exact mechanism the NER pipeline uses. |
| Bigram-presence assertion | Avoid disqualification | After training, assert `any(space_token in t and t.replace(space_token,' ').strip().count(' ')==1 for t in vocab)` and assert NO token has 2+ internal spaces. |

## How each piece fits — TRAINING (efficient)

## How each piece fits — ENCODE (fast, no O(n²) full rescan)

## The cross-word BIGRAM merge (assignment-specific) — explicit design

- **Spaces are mergeable symbols.** Represent each space as `space_token` (`▁`). Because it is an ordinary symbol, pairs like `(in, ▁)` and then `(in▁, the)` are eligible to merge → producing a token `in▁the` that `check_submission.py` reads as a one-internal-space **bigram**. (`check_submission.py` flags a bigram via `tok.replace(space_token,' ').strip()` containing an internal space — so the marker must survive into vocab tokens.)
- **Cap at one internal space.** Enforce a hard rule in BOTH train and encode: a merge is **forbidden** if the resulting token would contain **2+ `space_token` markers**. Filter such pairs out of `pair_counts` (training) and out of the rank dict / skip them in the greedy scan (encode). This guarantees ≥... never a 3-word token while still allowing bigrams.
- **Guarantee ≥1 bigram.** Don't rely on it emerging. After normal training, if no bigram exists, explicitly force-merge the highest-count legal cross-space pair so the vocab contains at least one. (Absence = disqualification, so make it deterministic.)
- **Decode must be length-faithful.** `decode([id])` replaces `space_token`→`' '` and returns exact characters; cumulative `decode(ids[:i+1])` lengths must match original text offsets (the NER aligner depends on this). Keep decode a pure string substitution with no normalization that changes length (no stripping, no case-fold, no NFC that alters char count).

## Vocab size — the efficiency / speed / F1 trade-off

| Vocab size | Tokens/sentence (efficiency) | Encode speed | F1 effect | Net guidance |
|------------|------------------------------|--------------|-----------|--------------|
| **Small** (~1–4k) | Worse (more tokens/word) | Fast train, but encode does MORE merges per chunk + more tokens to align | More UNK-ish fragmentation; NER sees noisier subwords | Avoid as floor; too fragmented. |
| **Medium** (~8–16k) | Good | Balanced | Usually best F1 for word-level NER (subwords still informative, first-subtoken labels stable) | **Recommended sweet spot to sweep first.** |
| **Large** (~32k+) | Best raw token count | Slower training; encode cache hit-rate rises (helps) but rank-dict scan per chunk slightly larger | Risk of over-merging rare/entity words into long tokens → fewer alignment points, can hurt entity F1; worse on the **hidden** domain (overfit vocab) | Use cautiously; watch hidden-domain generalization. |

- Efficiency (tokens/sentence) **monotonically improves** with larger vocab and especially with aggressive bigram merges — but bigram merges that swallow entity words can **cost F1** because NER labels the first subtoken per word; a bigram that fuses two words removes a labeling boundary. So tune bigram aggressiveness, not just vocab count.
- Speed is dominated more by **pre-tokenization + cache hit rate** than by raw vocab size; a good cache makes medium and large vocab similar in throughput.
- For the **hidden domain** (tokenizer 3), prefer **medium vocab + byte/char fallback** so unseen words degrade to short legal subwords instead of UNK. Generalizing vocab beats a large domain-overfit vocab here.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| List-of-symbols chunks + inverted index | True doubly-linked list of symbol nodes | Only if profiling shows merge splicing dominates; gives O(1) splice but heavier in pure Python and trickier to pickle. |
| Per-round `max()` over pair dict | `heapq` max-heap with lazy deletion | When the pair dict is very large and selection becomes the hot spot; mind stale entries. |
| Greedy lowest-rank encode (tiktoken style) | Priority-queue / DP "optimal" BPE encode | Not needed — greedy-by-rank reproduces training and is standard; DP adds complexity for no F1/efficiency gain here. |
| `Counter` ONLY for the initial seed count | `Counter` recomputed every merge | Never — that's the naive recount anti-pattern. |
| Encode cache as plain `dict` | `functools.lru_cache` on a method | Avoid — `lru_cache` wrappers and bound-method caches complicate pickling; a plain instance `dict` is picklable and explicit. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| GPT-2 pre-tokenization that **never merges across spaces** | Produces **zero bigrams** → disqualification, and leaves token efficiency on the table | Space-as-symbol pre-tokenization with cross-space merges capped at one space |
| Naive recount training (recount all pairs every merge) | O(num_merges × corpus); hours on 95MB in pure Python | Incremental deltas + inverted index |
| Whole-string repeated-rescan encode (`for merge: replace in full text`) | O(vocab × n) — loses the speed axis | Per-chunk greedy lowest-rank merge + per-chunk cache |
| `lambda` / `defaultdict(lambda: ...)` / closures / local classes in tokenizer state | **Unpicklable** → fails `BaseTokenizer.load` → disqualification | Module-level functions, plain `dict`/`Counter` (convert `defaultdict` to `dict` before pickling, or use a module-level default factory), `__getstate__`/`__setstate__` if needed |
| Decode that strips/normalizes/case-folds | Breaks length-faithful char-span alignment the NER pipeline relies on | Pure `space_token`→`' '` substitution, length-preserving |
| Over-large vocab for tokenizer 3 | Overfits training domains; bigrams swallow unseen words → UNK/poor alignment on hidden domain | Medium vocab + byte/char fallback |

## Pickle-safety dos and don'ts (hard gate)

- Keep all learned state in plain, picklable containers: `dict`, `list`, `tuple`, `set`, `int`, `str`. Merge ranks `{(str,str): int}`, vocab `{str: int}` / `{int: str}`, `space_token: str`.
- Define any helper used as a default/sort key at **module level** (so it pickles by qualified name).
- If you use `defaultdict`, either convert to a plain `dict` before pickling **or** give it a module-level (named) default factory — never `lambda`.
- If state needs custom handling (e.g. dropping the runtime cache or a compiled regex), implement `__getstate__`/`__setstate__` to serialize only primitives and recompile/rebuild on load.
- No `lambda`, no nested/closure functions, no local classes stored on the instance.
- Don't store a compiled `regex` Pattern if it risks non-portability — store the **pattern string** and compile in `__init__`/`__setstate__`.
- Don't rely on `lru_cache`-decorated methods for the encode cache (wrapper state is awkward to pickle).
- Don't store open file handles, generators, or torch tensors/devices in tokenizer state.

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| `regex` (3rd-party) | Python 3.11–3.12 | Use `import regex` — distinct from stdlib `re`; the only allowed advanced regex engine. Store pattern as string for pickle portability. |
| `numpy` (env-pinned) | Python 3.11–3.12 | Optional for bulk initial counting; keep learned state as Python primitives, not numpy arrays, to keep pickles small/portable. |
| `pickle` protocol | default (highest) | Trained tokenizer must load from `code/` alone — module path of the tokenizer class must be importable at unpickle time; keep the class in `code/bpe_tokenizer.py`. |

## Sources

- [tiktoken `_educational.py` (openai/tiktoken)](https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py) — verified the greedy lowest-merge-rank encode loop and list-of-parts representation. **HIGH**
- [Core BPE Implementation — DeepWiki / tiktoken](https://deepwiki.com/openai/tiktoken/2.1-byte-pair-encoding-(bpe)) — merge-rank priority + cache-locality merge order. **HIGH**
- ["From Hours to Seconds: Optimising BPE Tokeniser Training" — L. Thomson (Medium)](https://medium.com/@logan_16888/from-hours-to-seconds-optimising-bpe-tokeniser-training-f4234300d03e) — weighted-chunk dedup, pair-count deltas, inverted pair→chunk index, ~2000× speedup. **HIGH** (cross-checked against HF + tiktoken).
- [HuggingFace LLM Course ch.6 — BPE tokenization](https://huggingface.co/learn/llm-course/chapter6/5) — canonical train/encode reference (merge rules applied in learned order). **HIGH**
- [Sennrich et al. 2016, "Neural Machine Translation of Rare Words with Subword Units" (arXiv 1508.07909)](https://arxiv.org/pdf/1508.07909) — original BPE-for-NLP algorithm. **HIGH**
- Pickle-safety guidance — Python stdlib pickle semantics (no lambdas/closures; pickle by qualified name). **HIGH** (well-established stdlib behavior).
- Bigram cross-word adaptation, decode length-faithfulness, vocab/F1 trade-off — reasoned from `.planning/PROJECT.md` spec (`check_submission.py` bigram rule, NER decode-length alignment). **MEDIUM-HIGH** (assignment-specific, no external reference exists).

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
