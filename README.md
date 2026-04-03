# Topic Extraction Pipeline

Finds the top 5 topics from a year of JSON conversation transcripts using LLM distillation + BERTopic.

## How it works

1. Loads conversation transcripts from a JSON file
2. Flattens each conversation into a single transcript string
3. Calls `claude-sonnet-4-6` to summarize each conversation into one sentence
4. Runs BERTopic on the summaries to cluster them into topics
5. Prints the top 5 topics by size

## Setup

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Usage

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-...

# Run with your data
uv run python pipeline.py --input /path/to/transcripts.json

```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | none | Path to JSON transcript file |
| `--batch-size` | 10 | Conversations per API batch |
| `--batch-sleep` | 5 | Seconds to sleep between batches |

## Input format

The JSON file should be a list of conversations:

```json
[
  {
    "messages": [
      {"role": "user", "text": "I was charged twice this month."},
      {"role": "agent", "text": "I can see the duplicate charge, let me fix that."}
    ]
  }
]
```

If your keys differ (e.g. `content` instead of `text`), update `flatten_conversation()` in `pipeline.py`.

## Resuming interrupted runs

Summaries are cached to `summaries.json` after each batch. If the pipeline is interrupted, re-running it will skip already-summarized conversations and pick up where it left off.

## Output

```
======================================================================
Rank   Topic ID   Count    Label                        Keywords
----------------------------------------------------------------------
1      0          312      billing_charge_refund        charge, refund, billing, invoice, payment
2      1          287      login_password_access        password, login, account, reset, locked
...
======================================================================
```

The fitted BERTopic model is saved to `bertopic_model/` for later inspection.
