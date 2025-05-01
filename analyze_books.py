#!/usr/bin/env python3
import re
import sys
from collections import Counter
import statistics

def parse_entries(text):
    """Return a list of raw ENTRY blocks."""
    return re.findall(r"<ENTRY\b[^>]*?>(.*?)</ENTRY>", text, flags=re.DOTALL)

def extract_blurb(entry):
    """Extract the text inside <BLURB>…</BLURB>, or '' if missing."""
    m = re.search(r"<BLURB>(.*?)</BLURB>", entry, flags=re.DOTALL)
    return m.group(1).strip() if m else ""

def extract_tags(entry):
    """Return a list of tags (split on commas) from <TAGS>…</TAGS>."""
    m = re.search(r"<TAGS>(.*?)</TAGS>", entry)
    return [tag.strip() for tag in m.group(1).split(",") if tag.strip()] if m else []

def analyze(file_path):
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    entries = parse_entries(content)
    if not entries:
        print("No <ENTRY> blocks found.")
        return

    # BLURB word counts: count only alphanumeric words, excluding symbol-only lines
    blurbs = [extract_blurb(e) for e in entries]
    word_counts = [len(re.findall(r"\b[0-9A-Za-z]+\b", b)) for b in blurbs]

    # Exclude outliers using IQR method
    if len(word_counts) >= 4:
        q1, _, q3 = statistics.quantiles(word_counts, n=4, method='inclusive')
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = [wc for wc in word_counts if lower <= wc <= upper]
    else:
        filtered = word_counts

    excluded = len(word_counts) - len(filtered)
    avg_wc = statistics.mean(filtered) if filtered else 0
    std_wc = statistics.stdev(filtered) if len(filtered) > 1 else 0

    # TAGS counting
    all_tags = []
    tags_per_entry = []
    for e in entries:
        tags = extract_tags(e)
        all_tags.extend(tags)
        tags_per_entry.append(len(tags))

    tag_counts = Counter(all_tags)
    num_unique = len(tag_counts)
    num_once   = sum(1 for c in tag_counts.values() if c == 1)

    # Tag count per book stats
    highest_tags = max(tags_per_entry)
    lowest_tags = min(tags_per_entry)
    avg_tags = round(sum(tags_per_entry) / len(tags_per_entry))

    # Output
    print(f"Entries processed:                     {len(entries)}")
    print(f"Avg. BLURB word-count (±σ, excl. outliers): {avg_wc:.2f} ± {std_wc:.2f} (excluded {excluded})")
    print(f"Unique tags total:                     {num_unique}")
    print(f"Tags appearing only once:              {num_once}")
    print(f"Highest tags per book:                 {highest_tags}")
    print(f"Average tags per book (rounded):       {avg_tags}")
    print(f"Lowest tags per book:                  {lowest_tags}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "scraped_output.txt"
    analyze(path)
