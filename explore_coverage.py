import csv
from datasets import load_dataset

TULU3_IT_NAME = "ItalianNarratives/tulu-3-sft-mixture-translated-IT"
OUTPUT_CSV = "coverage.csv"


def main():
    print("Loading T3-it (full, no filters)...")
    t3_it = load_dataset(TULU3_IT_NAME, split="train")

    # partition IDs
    translatable_ids = set()
    multilingual_ids = set()
    untranslatable_ids = set()  # english but no translation

    for ex in t3_it:
        if ex["original_language"] != "en":
            multilingual_ids.add(ex["id"])
        elif ex["translated_messages_it"] is not None:
            translatable_ids.add(ex["id"])
        else:
            untranslatable_ids.add(ex["id"])

    print(f"\nOverall split:")
    print(f"  Translatable (en + translated)  : {len(translatable_ids):>8,}")
    print(f"  Untranslatable (en, no trans)   : {len(untranslatable_ids):>8,}")
    print(f"  Multilingual (non-en)           : {len(multilingual_ids):>8,}")
    print(
        f"  Total                           : {len(translatable_ids) + len(untranslatable_ids) + len(multilingual_ids):>8,}"
    )

    # group by source
    by_source: dict[str, dict] = {}
    for ex in t3_it:
        s = ex["source"]
        if s not in by_source:
            by_source[s] = {"total": 0, "translatable": 0, "multilingual": 0}
        by_source[s]["total"] += 1
        if ex["id"] in translatable_ids:
            by_source[s]["translatable"] += 1
        elif ex["id"] in multilingual_ids:
            by_source[s]["multilingual"] += 1

    # per-source coverage — only sources with at least one translatable sample
    print(
        f"\n{'source':<50} {'total':>8} {'translatable':>13} {'coverage':>10} {'< 50%':>6}"
    )
    print("-" * 92)

    below_50 = []
    sources_with_translations = [
        s for s, c in by_source.items() if c["translatable"] > 0
    ]
    for source in sorted(sources_with_translations):
        counts = by_source[source]
        total = counts["total"]
        translatable = counts["translatable"]
        coverage = translatable / total
        flag = "X" if coverage < 0.5 else ""
        if flag:
            below_50.append(source)
        print(
            f"{source:<50} {total:>8,} {translatable:>13,} {coverage:>9.1%}  {flag:>6}"
        )

    print(
        f"\nSources with <50% translation coverage: {len(below_50)} / {len(sources_with_translations)}"
    )
    print("These sources will use all available IT samples in sub_50:")
    for s in below_50:
        print(f"  - {s}")

    # sources with 0 translatable (informational)
    zero_translatable = [s for s, c in by_source.items() if c["translatable"] == 0]
    print(
        f"\nSources with 0 translatable samples (will always be in T3-un): {len(zero_translatable)}"
    )
    for s in sorted(zero_translatable):
        counts = by_source[s]
        print(
            f"  - {s:<48} (total: {counts['total']:,}, multilingual: {counts['multilingual']:,})"
        )

    # export to csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "total",
                "translatable",
                "multilingual",
                "untranslatable",
                "coverage",
                "below_50",
            ],
        )
        writer.writeheader()
        for source in sorted(by_source):
            counts = by_source[source]
            total = counts["total"]
            translatable = counts["translatable"]
            multilingual = counts["multilingual"]
            untranslatable = total - translatable - multilingual
            coverage = translatable / total if total > 0 else 0.0
            writer.writerow(
                {
                    "source": source,
                    "total": total,
                    "translatable": translatable,
                    "multilingual": multilingual,
                    "untranslatable": untranslatable,
                    "coverage": round(coverage, 4),
                    "below_50": coverage < 0.5 and translatable > 0,
                }
            )

    print(f"\nCSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
