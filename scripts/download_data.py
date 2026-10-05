"""Скачать заданное число целых историй TinyStories, не загружая весь корпус."""

import argparse
import codecs
import json
from pathlib import Path
from urllib.request import Request, urlopen

from _common import ROOT, file_hash, write_stories

REPOSITORY = "roneneldan/TinyStories"
FILENAME = "TinyStoriesV2-GPT4-train.txt"
SEPARATOR = "<|endoftext|>"


def download_stories(url, count):
    stories = []
    buffer = ""
    decoder = codecs.getincrementaldecoder("utf-8")()
    request = Request(url, headers={"User-Agent": "smallGPT-learning-project"})
    # Закрываем HTTP-поток сразу после нужного числа полных историй.
    with urlopen(request, timeout=60) as response:
        while len(stories) < count:
            chunk = response.read(64 * 1024)
            if not chunk:
                raise RuntimeError(f"Source ended after {len(stories)} complete stories")
            buffer += decoder.decode(chunk)
            while SEPARATOR in buffer and len(stories) < count:
                story, buffer = buffer.split(SEPARATOR, 1)
                story = story.strip()
                if story:
                    stories.append(story)
    return stories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stories", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--revision", help="Optional Hugging Face commit SHA")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stories < 3:
        parser.error("--stories must be at least 3")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "stories.jsonl"
    if destination.exists() and not args.overwrite:
        parser.error(f"{destination} exists; use --overwrite or another output directory")
    revision = args.revision
    if not revision:
        with urlopen(f"https://huggingface.co/api/datasets/{REPOSITORY}", timeout=60) as response:
            revision = json.load(response)["sha"]
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{FILENAME}"
    print(f"Downloading {args.stories} stories from {url}", flush=True)
    stories = download_stories(url, args.stories)
    temporary = destination.with_suffix(".jsonl.tmp")
    write_stories(temporary, stories)
    temporary.replace(destination)
    metadata = {
        "dataset": REPOSITORY, "revision": revision, "source_url": url,
        "license": "CDLA-Sharing-1.0", "sampling": "first N complete source stories",
        "stories": len(stories), "sha256": file_hash(destination),
    }
    (args.output_dir / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    readme_url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/README.md"
    with urlopen(readme_url, timeout=60) as response:
        (args.output_dir / "DATASET_README.md").write_bytes(response.read())
    print(f"Saved {len(stories)} stories, {destination.stat().st_size / 2**20:.2f} MiB: {destination}")


if __name__ == "__main__":
    main()
