import json
import os
import re
import time
import random
import argparse
from dotenv import load_dotenv
from os import getenv
from pathlib import Path


load_dotenv()

BATCH_SIZE = 10
BATCH_SLEEP = 5
MAX_RETRIES = 5
MIN_CONV_CHARS = 20
SYSTEM_PROMPT = "Summarize the core issue or request in this conversation in one sentence."


def load_conversations(path):
    with open(path) as f:
        return json.load(f)


def flatten_conversation(conv):
    lines = []
    for msg in conv.get("messages", []):
        role = msg.get("role", "unknown").capitalize()
        text = msg.get("text", "").strip()
        if text:
            lines.append(f"{role}: {text}")
    transcript = "\n".join(lines)
    if len(transcript) < MIN_CONV_CHARS:
        return None
    return transcript



def clean_summary(text):
    """Strip markdown headers and bold prefixes from LLM summary responses."""
    # Drop everything after a horizontal rule (some responses add extra sections)
    text = text.split("\n---")[0]
    # Strip bold prefixes like **Core Issue:** or **Summary:**
    text = re.sub(r"^\*\*[^*]+\*\*:?\s*", "", text.strip())
    # Skip markdown header lines, return first non-empty non-header line
    for line in text.splitlines():
        if re.match(r"^#+\s+", line):
            continue
        line = line.strip()
        if line:
            return line
    return text.strip()


def summarize_one(client, transcript):
    from anthropic import RateLimitError, APIError, AuthenticationError
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=128,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": transcript}],
            )
            return clean_summary(response.content[0].text)
        except AuthenticationError:
            print("Error: Anthropic API key is missing or invalid.")
            print("Set it with: export ANTHROPIC_API_KEY=sk-...")
            raise SystemExit(1)
        except (RateLimitError, APIError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            sleep_time = (2**attempt) + random.uniform(0, 1)
            print(f"  Retry {attempt + 1}/{MAX_RETRIES} after {sleep_time:.1f}s ({e})")
            time.sleep(sleep_time)


def load_cached_summaries(summaries_path):
    if not summaries_path.exists():
        return {}
    with open(summaries_path) as f:
        return json.load(f)


def save_summaries(cache, summaries_path):
    tmp = summaries_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    os.replace(tmp, summaries_path)


def run_distillation(transcripts, summaries_path):
    from anthropic import Anthropic, AuthenticationError
    client = Anthropic()
    cache = load_cached_summaries(summaries_path)

    uncached = [i for i in range(len(transcripts)) if str(i) not in cache]
    total = len(transcripts)
    print(f"Distilling conversations: {len(cache)} cached, {len(uncached)} remaining")

    for batch_start in range(0, len(uncached), BATCH_SIZE):
        batch = uncached[batch_start : batch_start + BATCH_SIZE]
        for i in batch:
            summary = summarize_one(client, transcripts[i])
            cache[str(i)] = summary
            done = len(cache)
            print(f"  [{done}/{total}] {summary[:80]}")
        save_summaries(cache, summaries_path)
        if batch_start + BATCH_SIZE < len(uncached):
            print(f"  Sleeping {BATCH_SLEEP}s...")
            time.sleep(BATCH_SLEEP)

    # Return summaries in original order, cleaning any cached markdown artifacts
    return [clean_summary(cache[str(i)]) for i in range(len(transcripts))]


def summarize_topic(client, keywords, representative_docs):
    from anthropic import RateLimitError, APIError
    docs_text = "\n".join(f"- {d}" for d in representative_docs[:5])
    prompt = (
        f"Keywords: {', '.join(keywords)}\n\n"
        f"Example conversations in this cluster:\n{docs_text}\n\n"
        "In one short sentence, describe what this group of conversations is about."
    )
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                system="You are labeling conversation topic clusters. Be concise and user-friendly.",
                messages=[{"role": "user", "content": prompt}],
            )
            return clean_summary(response.content[0].text)
        except (RateLimitError, APIError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep((2 ** attempt) + random.uniform(0, 1))


def print_topic_summaries(top5, model):
    from anthropic import Anthropic
    client = Anthropic()
    print("Top 5 Topics:\n")
    for rank, row in enumerate(top5.itertuples(), 1):
        topic_words = model.get_topic(row.Topic)
        keywords = [word for word, _ in topic_words[:5]]
        rep_docs = model.get_representative_docs(row.Topic) or []
        label = summarize_topic(client, keywords, rep_docs)
        print(f"  {rank}. [{row.Count} conversations] {label}")
    print()


def print_topic_table(top5, model):
    print("\n" + "=" * 72)
    print(f"{'Rank':<6} {'Topic ID':<10} {'Count':<8} {'Label':<28} Keywords")
    print("-" * 72)
    for rank, row in enumerate(top5.itertuples(), 1):
        topic_words = model.get_topic(row.Topic)
        keywords = ", ".join(word for word, _ in topic_words[:5])
        label = str(getattr(row, "Name", row.Topic))
        print(f"{rank:<6} {row.Topic:<10} {row.Count:<8} {label:<28} {keywords}")
    print("=" * 72 + "\n")


def run_bertopic(summaries, model_path, min_cluster_size=None):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    print(f"\nRunning BERTopic on {len(summaries)} summaries...")
    vectorizer = CountVectorizer(stop_words="english")
    if min_cluster_size is not None:
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, prediction_data=True)
        model = BERTopic(hdbscan_model=hdbscan_model, vectorizer_model=vectorizer, verbose=True)
    else:
        model = BERTopic(vectorizer_model=vectorizer, verbose=True)
    topics, _ = model.fit_transform(summaries)
    model.save(model_path)
    print(f"Model saved to {model_path}/")

    info = model.get_topic_info()
    info = info[info["Topic"] != -1]
    info = info.sort_values("Count", ascending=False)
    top5 = info.head(5).reset_index(drop=True)

    print_topic_table(top5, model)
    print_topic_summaries(top5, model)


def main():
    global BATCH_SIZE, BATCH_SLEEP

    parser = argparse.ArgumentParser(description="Extract top 5 topics from conversation transcripts")
    parser.add_argument("--input", help="Path to JSON file with conversation transcripts")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Conversations per API batch")
    parser.add_argument("--batch-sleep", type=float, default=BATCH_SLEEP, help="Seconds to sleep between batches")
    parser.add_argument("--summaries-path", default="summaries.json", help="Path to cache LLM summaries")
    parser.add_argument("--model-path", default="bertopic_model", help="Path to save BERTopic model")
    parser.add_argument("--min-cluster-size", type=int, default=None,
        help="HDBSCAN min_cluster_size (default: BERTopic's default of 10; use 2 for small datasets like fixtures)")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    BATCH_SLEEP = args.batch_sleep

    summaries_path = Path(args.summaries_path)

    input_path = args.input
    if not input_path:
        parser.error("--input is required. Example: uv run python pipeline.py --input transcripts.json")
    if not Path(input_path).exists():
        parser.error(f"Input file not found: {input_path}")

    print(f"Loading conversations from {input_path}...")
    conversations = load_conversations(input_path)
    print(f"Loaded {len(conversations)} conversations")

    print("Flattening conversations...")
    transcripts = []
    skipped = 0
    for conv in conversations:
        text = flatten_conversation(conv)
        if text is None:
            skipped += 1
        else:
            transcripts.append(text)
    print(f"Kept {len(transcripts)} conversations ({skipped} skipped as too short)")

    summaries = run_distillation(transcripts, summaries_path)
    run_bertopic(summaries, args.model_path, min_cluster_size=args.min_cluster_size)


if __name__ == "__main__":
    main()
