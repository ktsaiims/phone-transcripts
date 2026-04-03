# CLAUDE.md — Topic Extraction from JSON Conversation Transcripts (Top 5)

## Objective
Given a year of conversation transcripts stored in JSON, build a pipeline to discover the **top 5 conversation topics**. Use:
- **Preprocessing/cleaning**
- **LLM-based conversation distillation** (to preserve context while reducing noise)
- **BERTopic** to cluster and label topics

Avoid dumping the entire dataset into an LLM directly.

---

## High-Level Approach
1. **Load JSON**
2. **Flatten each conversation** into a single text string (optionally include both user + agent turns)
3. **LLM distill** each conversation into **one sentence** capturing the core issue/request (preserves agent context without agent-script noise)
4. Feed the resulting list of distilled texts into **BERTopic**
5. Let BERTopic **find topics naturally**, then identify the **top 5 by size**
   - Ignore topic `-1` (outliers)

---

## Assumptions (Adjust as Needed)
Your JSON likely contains something like:
- `conversation['messages']`
- each `msg` has:
  - `msg['role']` (e.g., `"user"` / `"agent"`)
  - `msg['text']` (message content)

Update key names to match your schema.

---

## Preprocessing / Cleaning Requirements
### 1) Flatten Conversations (Do NOT filter roles blindly)
To avoid skew while reducing noise, include both roles, but distill later.

Example logic:
- Iterate messages in a conversation
- Create lines like: `User: ...` and `Agent: ...`
- Join into one transcript string per conversation

### 2) Optional lightweight cleaning
- Strip whitespace
- Remove obvious boilerplate/system messages if present
- Filter extremely short conversations (e.g., < 20 chars), if desired

**Do not split conversations**—splitting would lose context.

---

## LLM Distillation Step (Critical)
### Goal
Convert each conversation transcript into:
- a **single sentence** describing the **core issue/request**
- incorporating relevant troubleshooting context that the agent introduced

### Prompt Contract
System prompt: “Summarize the core issue or request in this conversation in one sentence.”
User prompt: include the flattened transcript.

Return: one concise sentence.

### Batch/Rate Limit Strategy
Batching should be about **number of API calls per time window**, not splitting conversations.
- Each conversation is summarized independently.
- “Batch” = summarize conversations N at a time, then sleep.

Also implement:
- progress bar (optional)
- retries/backoff (recommended)
- saving intermediate outputs (recommended)

---

## BERTopic Step
### What BERTopic does (Pipeline)
1. Embeds text into vectors
2. Dimensionality reduction (UMAP)
3. Clustering (HDBSCAN)
4. Topic keyword extraction (c-TF-IDF) and topic labeling

### Should you force `nr_topics=5`?
**Recommended:** no.
- Let BERTopic find clusters naturally first
- Then reduce after inspecting results, or just select the top 5 by count

### How to get “Top 5”
- Use `model.get_topic_info()`
- Filter out topic `-1`
- Sort by `Count` descending
- Take top 5

---

## Deliverables (What the code should produce)
1. A list of distilled one-sentence summaries (cached to disk)
2. BERTopic model results
3. A printed table of **Top 5 topics**:
   - Topic ID
   - Count
   - Representative name/label
   - Optional: representative keywords

---

## Tooling
- Use **`uv`** for all package management and running scripts (not pip/poetry)
  - Install deps: `uv add bertopic anthropic sentence-transformers umap-learn hdbscan`
  - Run pipeline: `uv run python pipeline.py --input transcripts.json`

## Coding Style
- Write **simple, straightforward, readable** code
- No clever tricks, no over-engineering
- Prefer obvious over concise when they conflict

---

## Minimal Implementation Plan (Code)
### 1) Install
```bash
uv add bertopic anthropic sentence-transformers umap-learn hdbscan

### 2) Implement JSON loading + flattening
Convert each conversation object into a single transcript string.

### 3) Implement LLM summarization
Summarize each transcript into one sentence.
Batch API calls with sleep to avoid rate limits.
Cache results to summaries.json.

#### 4) Run BERTopic
model = BERTopic()
topics, probs = model.fit_transform(summaries)
Print top 5 topics.

