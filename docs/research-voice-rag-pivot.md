# Research Synthesis — Voice Fine-Tune + RAG Pivot

**Date:** 2026-05-11
**Status:** Research complete; awaiting user approval to start coding.
**Predecessor:** `docs/training-run-001.md` (knowledge-fine-tune attempt — underperformed Claude Sonnet 4.7).

## 1. Decision in one paragraph

Drop knowledge fine-tuning. Adopt the **2024–2025 consensus pattern: fine-tune for behaviour/voice, retrieve for facts**. The sku-food repo already implements a strong RAG layer (hybrid BM25 + dense, Voyage rerank-2.5, lost-in-middle reorder, Anthropic Contextual-Retrieval-style 4:1 RRF weighting). What is missing is (a) podcast/YouTube ingestion into the knowledge base, and (b) a voice-tuned model that lets us replace Claude Sonnet 4.5 at the generation step without losing Peter Chapman's voice. We build both inside this Python repo and expose the voice model as an OpenAI-compatible endpoint that sku-food can call by flipping `LLM_MODEL_ID`.

## 2. sku-food repo audit (what already exists)

| Concern | State | Notes |
|---|---|---|
| Stack | Next.js 15 + Supabase pgvector + Vercel AI SDK | TypeScript, pnpm |
| Embeddings | Voyage `voyage-4-lite`, 1024 dims | HNSW index |
| Generation | Claude Sonnet 4.5 via Vercel AI SDK provider registry | `LLM_MODEL_ID` env-driven — **pluggable** |
| Hybrid search | Dense + BM25, RRF fusion, 4:1 dense:sparse | Migration 016/017; matches Anthropic recipe |
| Reranking | Voyage `rerank-2.5`, top-20 candidates → top-N | Graceful fallback to vector-only on failure |
| Context assembly | Lost-in-the-middle reorder, `[Source N]` citation contract | `lib/rag/context.ts`, invariant tied to `source-order.ts` |
| Intent gate | Abuse / injection / greeting / identity short-circuit | `lib/rag/intent-router.ts` |
| Eval & telemetry | `query_logs` + `eval_feedback` (thumbs up/down) | Migrations 012, 018 |
| Query cache | Prompt-version-keyed | Migration 019 |
| System prompt | Detailed Peter Chapman voice spec, CART framework, voice rules, example exchanges | `config/projects/skufood/system_prompt.md` |
| Source types declared | `markdown, docx, pdf, transcript` | `transcript` already in the schema — **podcast/YT ingestion is anticipated but not implemented** |
| Last shipped commit | 2026-04-30 "pilot ship — citations, security, ui redesign, title migration" | Active production |
| Tests / evals | Live in sibling `genAI-template` repo (not in sku-food) | Out of scope for our work |

**Strategic implication:** sku-food does *not* need RAG re-architecture. Our job is (1) feed it more knowledge (transcripts) and (2) give it a voice model to call.

## 3. Architecture decision: where the boundary lives

Three viable architectures considered:

**A. Voice model replaces Sonnet 4.5 inside sku-food** (single-model RAG)
- Pros: Smallest change to sku-food (env var flip). Lowest latency.
- Cons: Voice model sees raw chunk prose — flattening risk per Stream 4.

**B. Two-stage decoupled** (retriever-summarizer → voice paraphrase)
- Pros: Strongest voice preservation; voice LoRA's job collapses to pure style transfer.
- Cons: Extra LLM call (Haiku/cheap model summarises chunks into a fact card before voice model paraphrases). More moving parts.

**C. Hybrid** — start with (A); add (B) only if eval shows voice flattening.
- **Recommended.** Lets us ship faster, then upgrade only if measured.

**Why this works:** sku-food's `app/api/chat/route.ts` already separates retrieve → assemble → generate. Replacing the generate step (or adding a summarise-then-generate step) is a localized change behind the existing `LLM_MODEL_ID` abstraction.

## 4. Voice fine-tuning recipe (Stream 2 synthesis)

**Base model:** Qwen 3.5 9B (keep — strongest 7–9B instruction follower; we have an established Unsloth pipeline; tokenizer handles English idiom well).

**Adapter config (starting point):**
- LoRA rank **16**, alpha **16** (alpha = rank, not 2×; lower effective scale → less knowledge overwrite)
- Target modules: **`q_proj, k_proj, v_proj, o_proj` only — exclude MLP** (interpretability evidence — Geva et al., Rimsky et al., MEMIT — places factual recall in mid MLPs; restricting LoRA to attention preserves the world model)
- Dropout 0.05, weight decay 0.01, NEFTune noise α=5
- Optimizer: AdamW, LR 5e-5, cosine schedule, 3% warmup
- Effective batch 32 (per-device 2 × grad-accum 16), seq 2048
- **Epochs 1–2** (prior runs overfit at 3)
- Embeddings + lm_head **frozen**
- Optional: L2-on-deltas or KL-to-base regularizer (LoRA-FA, "anchored LoRA")

**Training method:** SFT primary, optional ORPO polish.
- ORPO (arXiv:2403.07691) is monolithic (SFT + preference in one loss, no reference model) — preferred over DPO for low-data stylistic tasks. Use only if the SFT pass leaves measurable style gaps.

**Data mix (60/30/10) on top of diarized Peter-only transcripts:**
1. **60% — Persona continuation.** Long monologue chunks (512–2048 tokens) as assistant turns, with a generic eliciting user turn ("Walk me through how you think about X"). Highest voice signal, lowest knowledge contamination.
2. **30% — Style-rewrite pairs.** Generic competent answer → Peter-voice answer. The facts in both sides are neutral so the LoRA learns *transformation*, not facts. Generate with Claude using 3–5 Peter excerpts as few-shot.
3. **10% — RAG-grounded Q&A.** `(question + <context> block, peter_voice_answer_that_cites)`. Teaches the runtime shape — this is the format the model sees in production.

**Entity scrubbing (CRITICAL):** in continuation data, replace specific numbers, dates, brand names, retailer-specific data with `[number]`, `[year]`, `[retailer]` placeholders. This is the single biggest lever for keeping facts out of weights — Stream 2's headline opinion.

**Adversarial training samples (5–10%):** examples where `<context>` contradicts a Peter-ism and gold answer follows the context. Prevents "model trusts itself" failure mode.

**General-replay (5–10%):** small slice of generic SFT (OpenHermes/Alpaca-cleaned) to anchor general capability.

## 5. Audio → SFT data pipeline (Stream 3 synthesis)

Single Python pipeline producing two outputs from the same processed transcripts: **(a)** RAG ingestion JSON for sku-food, **(b)** SFT training JSONL for the voice fine-tune.

```
yt-dlp + podcast feed
       │
       ▼
faster-whisper large-v3-turbo  ──(WhisperX wraps for VAD + batched + word align)
       │                          fallback to large-v3 if WER critical
       ▼
pyannote.audio 3.1 diarization  (CPU on Mac, CUDA on Lambda)
       │
       ▼
ECAPA-TDNN speaker embedding match  (30–60s Peter reference clip; cosine ≥ 0.55–0.65; majority-vote on cluster centroids ≥3s)
       │
       ▼
WhisperX forced alignment (wav2vec2 word-level confidences)
       │
       ▼
Quality filters:  Peter-only • duration 3–90s • mean word-conf ≥ 0.6 • no-speech-prob < 0.3 •
                  filler ratio < 0.15 • crosstalk < 10% • SponsorBlock ad strip
                  Topic gate: BGE-large-en-v1.5 cosine to Food-&-CPG seed prompts, keep ≥ p40
       │
       ├──────────────────────────────┐
       ▼                              ▼
SFT JSONL (voice training)     RAG documents (sku-food ingestion)
   60% continuation               document_type=transcript
   30% style-rewrite              chunks per episode/segment
   10% RAG-grounded               Voyage-4-lite embeddings
   adversarial 5–10%              insert into existing Supabase schema
```

**Tooling (concrete):**
- Ingest: `yt-dlp` (latest 2025.x) with `--write-info-json --write-chapters --sub-langs en`
- ASR: `faster-whisper` + `WhisperX` (default `large-v3-turbo`; `large-v3` for high-stakes WER)
- Diarize: `pyannote.audio` 3.1 (HF token; CPU-only on Apple Silicon — no MPS)
- Speaker ID: SpeechBrain `spkrec-ecapa-voxceleb` OR pyannote `wespeaker-voxceleb-resnet34-LM`
- Embed for topic gate: `BAAI/bge-large-en-v1.5`
- Synthetic Q&A / style-rewrites: Claude Sonnet 4.6 via Anthropic SDK (Self-Instruct / Humpback pattern — arXiv:2308.06259; LongForm — arXiv:2304.08460)
- Ad strip: SponsorBlock API + chapter metadata + small ad-text seed embedding similarity

**Compute estimates (~100h audio):**
| Stage | M4 Pro 48GB | A100 80GB (Lambda) |
|---|---|---|
| yt-dlp + ffmpeg | 2–4h (network-bound) | same |
| WhisperX large-v3-turbo | 6–10h | 0.5–1h |
| pyannote diarization | 15–25h (CPU only) | 2–3h |
| Speaker ID + filtering | 1–2h | <1h |
| Claude synthesis | 2–5h wall, ~$15–40 tokens | same |

→ End-to-end on A100: **~4–6 hours, ~$5–12 on Lambda.** Validate the pipeline on 1h locally on the M4 Pro first.

**JSONL schema (Unsloth/TRL chat-template compatible):**
```json
{
  "id": "ep042-seg017",
  "source": {"type":"youtube|podcast","video_id":"...","url":"...","timestamp":[932.4, 978.1]},
  "speaker": "peter",
  "speaker_confidence": 0.81,
  "asr_confidence": 0.74,
  "duration_sec": 45.7,
  "topic_tags": ["pricing","listings"],
  "generation_mode": "continuation|natural_qa|reverse_qa|style_rewrite|rag_grounded|adversarial",
  "scrubbed": true,
  "messages": [
    {"role":"system","content":"You are Peter Chapman, founder of SKUFood..."},
    {"role":"user","content":"<eliciting question>"},
    {"role":"assistant","content":"<Peter's verbatim or scrubbed transcribed answer>"}
  ]
}
```
**Always keep the raw audio span pointer.** Future TTS work will need it.

## 6. RAG architecture validation (Stream 1 synthesis)

sku-food already matches the recommended stack. Confirmed:

| Layer | Stream 1 recommendation | sku-food state |
|---|---|---|
| Embeddings | Voyage-3-large or BGE-M3 | Voyage-4-lite ✅ (newer, smaller dims — fine) |
| Sparse | BM25 over contextualised chunks | BM25 via Postgres `tsvector` ✅ |
| Fusion | RRF, top-150 candidates | RRF 4:1 dense:sparse, top-30 ✅ (smaller pool — fine for current corpus) |
| Reranker | Cohere Rerank 3.5 or BGE-reranker-v2-m3 | Voyage `rerank-2.5` ✅ |
| Lost-in-middle | recommended | implemented in `context.ts` ✅ |
| Citation contract | recommended | `[Source N]` invariant ✅ |
| Eval | RAGAS + Phoenix + golden set | thumbs feedback exists; **golden set not built** ❌ |

**Gaps to close:**
1. **Anthropic-style Contextual Retrieval prefix** — sku-food does plain chunks; adding a 50–100 token LLM-generated context prefix at ingestion time gives a ~50% retrieval-failure reduction (Anthropic blog, Sept 2024). Cheap with prompt caching. Done at ingestion, not retrieval — minimal app change.
2. **Golden eval set** (150–300 questions, Peter-validated). The single highest-ROI work on the eval side. Stream 1 is explicit: "matters more than any framework choice."
3. **Phoenix/Arize tracing** for retrieval observability.

**What to skip:** GraphRAG (over-engineered for this corpus size), Self-RAG (training fragile), Adaptive-RAG routing (marginal vs prompt-based classifier), long-context dump (lost-in-the-middle is real per NoLiMa benchmark arXiv:2502.05167).

## 7. System composition (Stream 4 synthesis)

**Recommended end-to-end** (architecture C from §3, evolved):

```
USER QUERY
   │
   ▼
[sku-food TypeScript] intent gate → query cache → query processor →
                      retriever (hybrid + rerank) → context assembler ([Source N], lost-in-middle)
   │
   │  full chat-completion request via Vercel AI SDK provider registry
   │  LLM_MODEL_ID = openai-compat:peter-voice-qwen
   ▼
[python food-cpg-intelligence] FastAPI OpenAI-compatible endpoint
                                  serving QLoRA-merged Qwen 9B (vLLM or llama.cpp)
   │
   ▼
STREAMED ANSWER (citations preserved by upstream context)
```

**Voice-preservation rules baked into the runtime system prompt:**
- Role-then-evidence ordering ("You ARE Peter Chapman" → rules → `<documents>`) — already correct in sku-food
- Explicit recitation prohibition ("Treat documents as private notes — never quote verbatim, paraphrase in your voice")
- `<documents>` block stays in system role, not user role (already correct in sku-food's `buildSystemMessage`)

**If voice flattens under eval (escalation path):**
- Insert a Haiku/Qwen-base summarisation step between retrieval and voice model
- Voice model receives a JSON `tool_result` fact card `[{claim, source_id}, ...]` instead of raw chunks
- Strongest defense per literature (Anthropic "Building effective agents" prompt chaining pattern)
- Cost: +1 LLM call, +~300–800 summary tokens

## 8. Evaluation framework

Three layers, all required before declaring success:

**A. Retrieval (sku-food side, but rerun with new transcript content)**
- Golden set 150–300 questions; recall@10, MRR@10
- Built collaboratively with Peter — non-negotiable

**B. Voice fidelity (this repo)**
- Stylometric: function-word JSD vs held-out Peter corpus; mean sentence length; LIWC categories
- **Peter classifier**: fine-tune DistilRoBERTa to discriminate Peter vs other Canadian food-retail experts; report P(Peter) on every output. Gate releases on P(Peter) ≥ 0.75 under RAG load.
- Perplexity on held-out Peter transcripts vs base model (should drop monotonically per epoch)
- LLM-as-judge (Claude Sonnet 4.7) on PersonaGym-style rubric: *consistency, distinctiveness, knowledge-deferral, voice fidelity, factual restraint*. Pairwise A/B vs (base + system prompt only) and vs (Claude Sonnet 4.5 + system prompt)
- Human A/B with 3–5 people who actually know Peter — gold standard, no substitute

**C. End-to-end (RAGAS + tracing)**
- RAGAS: faithfulness, answer relevancy, context precision, context recall — wire into a CLI under `src/food_cpg_intelligence/evaluation/`
- Phoenix/Arize for trace-level observability
- Skip TruLens (stagnated) and ARES (research-only)

**Guard metrics that block release:**
- P(Peter) < 0.75 → voice flattening — escalate to two-stage architecture (§7 escalation)
- RAGAS faithfulness drop > 5pp vs Sonnet 4.5 baseline → factual regression — abort
- Base-model MMLU/TruthfulQA drop > 2pp → catastrophic forgetting — reduce rank or epochs

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Style + facts co-learned despite scrubbing | Aggressive entity scrubbing in continuation; rely on rewrite-pair format; eval factual-restraint dimension |
| Adapter overfits to podcast-specific topics | Deduplicate near-identical monologues; cap any single topic at ≤5% of corpus |
| Model ignores RAG context, trusts internal weights | 5–10% adversarial training samples where context contradicts a Peter-ism |
| LoRA underfits voice (too conservative) | Tiered rollout: rank 8 → 16 → 32 if early eval flat; ORPO polish stage |
| Claude judge biased toward Claude-flavoured outputs | Always pair with stylometric + human eval; never accept LLM-judge alone |
| pyannote MPS unsupported | Force CPU on Mac; full pass on Lambda CUDA |
| Whisper hallucination on silence/music | Pre-VAD via WhisperX; drop repetitive-token segments (`compression_ratio_threshold` ~2.4) |
| Unsloth deps pinned | Per project memory: don't bump transformers/datasets/torch independently. ASR pipeline lives in separate venv/container |
| Lambda cost overrun | 1h local test on M4 Pro before any Lambda spend; full pipeline ≈4–6 A100 hours = $5–12 |

## 10. Proposed phased plan (post-approval)

### Phase 1 — Audio pipeline (local M4 Pro, 1 week)
- `src/food_cpg_intelligence/ingest/audio/` package: yt-dlp + faster-whisper + pyannote + WhisperX
- Smoke test on 1h of one Peter podcast end-to-end
- Output: validated JSONL schema + a single processed episode

### Phase 2 — Data generation (Lambda A100, 1 day)
- Full ~100h pass: ASR + diarize + speaker-ID + filter
- Claude synthesis of style-rewrite pairs + reverse Q&A
- Output: training JSONL (~10–30k examples after filtering) + RAG-ingest JSONL

### Phase 3 — Voice fine-tune (Lambda A100, 1–2 days)
- Unsloth QLoRA on Qwen 3.5 9B with the §4 recipe
- Save LoRA + merged 16-bit + GGUF (per existing pipeline)
- Run §8 eval suite; gate on P(Peter) and forgetting metrics

### Phase 4 — Serve & integrate (local, then Vercel preview, 2–3 days)
- vLLM or `llama-cpp-python` OpenAI-compatible FastAPI endpoint
- Add provider entry to sku-food's Vercel AI SDK registry; flip `LLM_MODEL_ID` on a preview branch
- Side-by-side eval vs Sonnet 4.5 with the golden set

### Phase 5 — Ingest transcripts into sku-food RAG (parallel to Phase 4, 1–2 days)
- Use existing sku-food ingestion (`pnpm re-ingest:local`); transcript already declared as a `source_type`
- Add Anthropic-style Contextual Retrieval prefix at chunk time (~50–100 tok LLM-generated context per chunk)

### Phase 6 — Optional two-stage decoupled (only if Phase 4 eval shows flattening)
- Add Haiku-based fact-card summariser between retrieval and voice model

## 11. What we are NOT doing

- ❌ Re-architecting sku-food's RAG (it's already correct)
- ❌ Building a new vector DB (Supabase pgvector stays)
- ❌ GraphRAG / HippoRAG / RAPTOR / Self-RAG (overkill for corpus size; revisit if eval shows multi-hop failures)
- ❌ Knowledge fine-tuning (the canonical anti-pattern; was Run 001/002)
- ❌ Long-context-no-RAG (lost-in-the-middle real per NoLiMa)
- ❌ Replacing Claude end-to-end before measuring (Sonnet 4.5 is the baseline, not the enemy)

## 12. Open questions for the user

1. **Voice model deployment target**: vLLM (better latency, GPU required) vs llama.cpp (CPU-viable, slower) for the inference service?
2. **Inference host**: keep on Lambda permanently (cost), self-host on a Mac mini / Studio (latency variance), or run on Modal/RunPod serverless GPU?
3. **Podcast/YouTube source list**: where does the master list live? Do we have a reference audio clip of Peter ready for speaker-ID?
4. **Access to sku-food Supabase**: do we get a service-role key for re-ingestion of transcripts, or do we hand off ingestion artifacts and let the sku-food team ingest?
5. **Eval Peter-validation**: 150–300 question golden set requires Peter's time. Realistic to schedule?

## 13. Source notes (subagents had web access blocked)

All four research subagents ran with WebSearch/WebFetch/Bash disabled in their sandboxes and produced reports from Jan-2026 training knowledge. URLs are recall-based and should be spot-verified before publishing. Substance has been cross-checked against the actual sku-food repo (which Stream 4 couldn't access) — and the sku-food architecture independently validates Stream 1's recommendations (it implements them). High confidence in the overall direction; medium confidence in any specific paper/arxiv ID.

**Key references to verify:**
- Anthropic Contextual Retrieval (Sept 2024)
- Biderman et al., "LoRA Learns Less and Forgets Less," arXiv:2405.09673
- Hong et al., "ORPO," arXiv:2403.07691
- Shao et al., "Character-LLM," arXiv:2310.10158
- Li et al., "Humpback / Instruction Backtranslation," arXiv:2308.06259
- Köksal et al., "LongForm," arXiv:2304.08460
- WhisperX — Bain et al., arXiv:2303.00747
- pyannote.audio 3.1 — Bredin, Interspeech 2023
- NoLiMa long-context degradation, arXiv:2502.05167
- HippoRAG 2, arXiv:2502.14802
