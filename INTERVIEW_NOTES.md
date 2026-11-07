# Interview notes: finrag-eval

All numbers come from `reports/` (test split unless marked dev). See the README for the full tables.

## Design choices and why

**FiQA-2018 (BEIR) as the benchmark.** It is a real financial-domain retrieval task with public qrels
and a published BEIR BM25 reference (nDCG@10 = 0.236; my tuned bm25s gets 0.2502). So the baseline
can be sanity-checked against the literature. Weaknesses:
- The passages are StackExchange-style opinion answers, not filings.
- The qrels are sparse: about 2.6 judged passages per test query (`data_stats.json`). That matters
  for how citation precision should be read (see weaknesses).

**Protocol: tune on dev, report test once.**
- Every choice is selected on the 500 dev queries: BM25 k1/b, embedder, RRF k and dense weight,
  reranker model, depth and fusion, and the generation prompt.
- `experiments.common.require_tuning_split` raises if a tuning function is pointed at test.
- `evaluate_split` reads only the frozen selections in `reports/dev_tuning.json`.

**Tuned BM25 baseline, not a strawman.** The lift claims are against BM25 with dev-tuned k1/b
(a 3x3 grid) and Snowball stemming. It scores 0.2502 on test, above the BEIR reference.

**Qdrant in embedded mode.** No server, no Docker; the index is persisted on disk.
- Embedded mode does exact (brute-force) cosine search and ignores HNSW parameters, so the dense
  numbers carry no ANN approximation error.
- The collection is still created with an HNSW config, so `QdrantDenseIndex.remote` gets HNSW
  against a server.
- The cost is latency. Embedded search is about 300-440 ms per query at 57.6K x 768. A NumPy exact
  search over the same vectors takes 10.7 ms and agrees on 99.66% of top-100 ids
  (`latency.json: numpy_exact_dense_reference`).
- Writes were also slow at first: one SQLite commit per point. Setting WAL + synchronous=NORMAL on
  the local storage cut the bge-base upsert to 62.5 s.

**RRF instead of score interpolation.** BM25 scores and cosine similarities live on different scales,
and their distributions shift per query. RRF needs only ranks, so there is no normalisation, and it has
two knobs: k and a per-list weight. Both are tuned on dev.

**Rerank only the head, and fuse rather than replace.**
- A cross-encoder costs one forward pass per (query, passage) pair, so it scores only the top-N fused
  candidates; the tail keeps its order.
- CE scores are computed once at the deepest N (50) on dev and reused, so the depth, fusion-k and
  weight sweep needs no extra GPU.
- The final design RRF-fuses the CE order with the first-stage order. Pure reordering by CE score did
  not help (see below).

**Generator behind a protocol.** `Generator.generate_batch(items) -> list[GenerationResult]` is the
whole interface. The local Qwen2.5-1.5B-Instruct backend and the test fake both implement it. A vLLM
or API backend only needs that one method; the API and evaluation code never import transformers.

**Faithfulness via NLI, not an LLM judge.** A DeBERTa-v3 NLI model is cheap, deterministic and local,
and its behaviour is known. An LLM judge would need a much bigger model or an API key.
- The metric is sentence level: each answer sentence must be entailed (p >= 0.5) by a passage it
  cites.
- I also report "context faithfulness", entailed by any passage in the context. The gap between the
  two shows whether the problem is bad grounding or bad citing.

**Citation precision against qrels, reported next to context precision.** A cited passage being
judged relevant is a stricter check than "the id exists". Comparing it to the share of relevant
passages in the context shows whether the model picks relevant passages better than chance.

## What didn't work (and what I did about it)

1. **Textbook hybrid hurt.**
   - RRF with k=60 and equal weights scored 0.3743, against 0.4058 for dense alone: -0.0316, p=0.0005.
   - BM25 is 0.16 nDCG behind dense on this corpus, so equal votes promote BM25's mistakes.
   - Tuning on dev fixed it (k=5, dense weight 3.0: 0.4128, +0.0069 over dense, p=0.015).
   - The first dense-weight grid was too narrow and the optimum sat on its edge, so I extended it
     (up to 10) before tuning the reranker on top.
2. **Pure cross-encoder reranking did not help.**
   - On dev, both MiniLM-L6 and bge-reranker-base scored below the tuned hybrid at every depth
     (5-50), and got worse with depth (bge at depth 50: 0.380 vs 0.430).
   - On test, pure bge reranking is +0.0005 (p=0.92).
   - These MS MARCO-style rerankers disagree with FiQA's notion of relevance (opinionated, long
     answers), and the dense model is already strong.
   - My first selection code always switched a reranker on. I changed it so the reranker must beat the
     tuned hybrid on dev to be selected.
   - RRF-fusing the CE order with the first-stage order (depth 20, k=5, CE weight 0.5) did beat it:
     test +0.0086 [0.0022, 0.0152]. This is small, and I report it as small.
3. **The few-shot prompt was worse than plain instructions** (dev, 24 queries, selected_top5).
   - v1 had rules plus a closing reminder with an example id; 45% of answers cited something.
   - v2 used a one-shot demonstration and an abstention demonstration but no closing reminder.
     Citation rate fell to 17% and sentence faithfulness from 0.22 to 0.04.
   - The 1.5B model follows the instruction nearest the question more than the demonstration. v1 went
     to test; both dev reports are kept (`generation_eval_dev_prompt_v1.json`, `_v2.json`).
4. **Embedded Qdrant write speed.** The first build took about 10 min per collection because of
   per-point commits. Fixed with SQLite pragmas (see above), not by switching tools.

5. **Batched generation ate host RAM on Windows.**
   - The first test generation run grew past 5 GB RSS and ran about 2x slower than on dev. My RAM guard
     killed it before the machine ran out of memory.
   - I found two causes:
     - Loading on CPU and then calling `.to("cuda")` left about 3 GB resident. I measured 3.6 GB vs
       0.8 GB RSS with `device_map="cuda"`.
     - With the driver's sysmem fallback, VRAM allocations spill into host memory instead of failing.
       So PyTorch's caching allocator never frees, and length-bucketed batches keep fragmenting it.
   - Fix: load with `device_map`, call `set_per_process_memory_fraction(0.8)`, and run `empty_cache()`
     after every batch. I also checkpoint each config to disk.
   - The rerun peaked at 1.83 GB host RSS (`generation_runs_test.json`). The cost was about 23 wasted
     GPU-minutes and a smaller test generation study (two configs, not three).

## Generation results and failure modes (test, 150 queries, read from the outputs)

Numbers are from `reports/generation_eval.json`. The counts come from reading
`reports/generation_judgments_test.jsonl` and `generations_test.jsonl` (selected_top5 unless stated).

| | top-3 | top-5 |
|---|---|---|
| answered / 150 | 126 | 131 |
| answers citing anything | 0.381 | 0.389 |
| citation validity | 0.975 | 0.970 |
| citation precision vs qrels (base rate) | 0.364 (0.247) | 0.237 (0.184) |
| sentence faithfulness / any-context support | 0.232 / 0.764 | 0.241 / 0.791 |
| abstention (relevant in ctx / none) | 0.148 / 0.174 | 0.122 / 0.133 |

Failure modes, from most to least important:

1. **Grounded but uncited.** 80 of 131 answers have no `[docID]` at all. Of their sentences, 366 are
   entailed by some context passage and 113 are not. The model mostly paraphrases the passages
   correctly but drops the citation format. That is why sentence faithfulness (0.24) sits so far below
   context support (0.79). The fix is format control (constrained decoding, a larger model, or
   post-hoc attribution by NLI), not better retrieval.
2. **Copying the prompt's example id.** 3 answers (2 for top-3) cite `[doc12345]` from the
   instruction, as in "Based on the passage [doc12345], ...". They are scored as invalid citations.
   The few-shot prompt that removed the example id was worse overall (dev: 17% vs 45% citing).
3. **Misattribution with more context.** Sentences that cite a passage which does not entail them,
   while another context passage does, rose from 3 (top-3) to 20 (top-5). Citation precision vs qrels
   falls from 0.364 to 0.237. More passages give the model more ways to point at the wrong one.
4. **Citation as a preamble.** Many cited sentences put the id first, as in "According to [docX], ...".
   Several then say something the passage does not support. One example cites a passage for "feeling
   comfortable with an 8% gain suggests selling" with NLI p=0.00. The passage is about something else,
   while another passage in context does support the sentence.
5. **Abstention is uninformative.** The abstention rate is about the same whether or not a
   judged-relevant passage is in context (0.122 vs 0.133). Some abstentions are on clearly answerable
   questions (bond ETF capital gains with a relevant passage in context). Meanwhile 52 of 60 queries
   with no judged-relevant passage are answered anyway, though sparse qrels mean some of those
   contexts are in fact useful.

What helps and what doesn't: top-5 vs top-3 makes no difference to answer-level faithfulness
(+0.0004, p=0.99) and lowers citation precision. So top-3 is the better default for this model.

## Weaknesses (what I'd say before the interviewer does)

- **Sparse qrels.** About 2.6 judged passages per query. Retrieval metrics and citation precision vs
  qrels are lower bounds. The BEIR qrels were built from the original answers, which can favour
  systems that retrieve the same kind of passages as the annotated ones.
- **The reranking gain is small** and costs about 10 s per query on CPU. On CPU I would serve the tuned
  hybrid.
- **The generator is small.** Qwen2.5-1.5B follows the citation format inconsistently: 61% of top-5
  answers cite nothing. The prompt was chosen from two variants on 24 dev queries, which is a small
  sample. The BM25-context generation config was only run on dev.
- **The NLI judge is unvalidated.** No human agreement study. It truncates at 512 tokens, and numeric
  or multi-hop claims can be misjudged.
- **Latency is not production-like.** Embedded Qdrant dominates. No HNSW recall/latency trade-off was
  measured, the GPU was shared, and thermals were not controlled.
- **One dataset, one seed, greedy decoding.** No variance across seeds for generation. Retrieval is
  deterministic.
- **CI is written but not yet run** (the repo has not been pushed).

## Likely interview questions

**1. Why is your BM25 baseline 0.250 when BEIR reports 0.236?**
Different implementation and tuning. bm25s uses Lucene scoring with Snowball stemming and stop words,
and k1/b were tuned on dev (k1=0.9, b=0.75). BEIR used Elasticsearch defaults. A stronger baseline
makes the lift claim more conservative.

**2. How do you know your metric code is right?**
- Unit tests on hand-computed cases: ideal and inverted rankings, duplicates, no relevant documents,
  k larger than the list.
- Every reported variant is cross-checked against `ir_measures` (pytrec_eval), with a max absolute
  difference of 4.4e-16 across the 9 variants.

**3. Is +0.0086 nDCG from the reranker real?**
- A paired bootstrap over the 648 test queries gives CI [0.0022, 0.0152] and p=0.012, so it is probably
  not noise.
- It is also about 2% relative and costs 147 ms on GPU (10 s on CPU). I'd only ship it on GPU.
- It was selected on dev, and test was scored once.

**4. Why did plain RRF hurt?**
RRF with equal weights assumes both lists are comparably good. Here BM25 is 0.16 nDCG behind dense, so
its top ranks push relevant dense results down. A smaller k (sharper rank discounting) and a 3x dense
weight let BM25 contribute mainly when the dense list is unsure.

**5. Why did pure reranking hurt but fusion help?**
- The cross-encoders were trained on MS MARCO, which has short factoid passages. FiQA relevance is
  long opinionated answers.
- Replacing the order throws away the bi-encoder's in-domain signal. Fusing keeps it and lets the CE
  break ties.
- The dev curve supports this: pure reranking gets worse as depth grows, because the CE has more room
  to promote its own mistakes.

**6. How is faithfulness measured, and what are its failure modes?**
- Split the answer into sentences, keeping citations attached.
- For each sentence, take the max NLI entailment probability over the passages it cites.
- A sentence with no citation of its own is judged against the answer's other valid citations; the
  own-citation rate is reported separately. An answer with no valid citation scores 0.
- A sentence is faithful if that probability is >= 0.5.
- Failure modes: truncation at 512 tokens, paraphrase that NLI misses, and numeric claims.
- "Context faithfulness" (any passage in the context) separates "made it up" from "forgot to cite".

**7. Why not use an LLM as the judge?**
It needs either an API key (ruled out) or a model much larger than the generator, which would not fit
the GPU budget. NLI is cheap and deterministic. The cost is that it is a coarser judge; I state that as
a limitation and would add a small human-labelled agreement check next.

**8. How would you scale this to 10M passages?**
- Use a Qdrant server (or similar) with HNSW, and tune ef/m for a recall@100 target against exact
  search on a sample.
- Shard BM25, or move it to a Lucene-based engine.
- Batch query embedding, cache frequent queries, and keep the reranker on GPU with a small depth (20).
- The harness stays the same; only `QdrantDenseIndex.remote` and the BM25 backend change.

**9. How did you avoid test leakage?**
- Every tuning function calls `require_tuning_split`, which raises on test.
- Selections are written to `dev_tuning.json`, and the test evaluation only reads them.
- Test retrieval was evaluated once with all variants, and the generation test run was done once
  after the prompt was frozen on dev.
- Dev-only probes, like the CE-fusion idea, were run on dev before being added to the grid.

**10. What would you do next?**
- Validate the NLI judge against about 100 human labels.
- Try a 3-7B generator or constrained decoding for the citation format, since format compliance is the
  main generation weakness.
- Measure HNSW with a Qdrant server.
- Fine-tune the reranker on FiQA train (5,500 queries), because the off-the-shelf MS MARCO
  cross-encoders were the weak link.
