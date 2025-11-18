import json
from pathlib import Path


OUTPUT_PATH = "data/reverse_lookup.json"
def main() -> None:
    lookup = {}
    for f in Path("data/test_prompts").glob("*.jsonl"):
        for line in f.open():
            d = json.loads(line)
            long_desc = d["long_description"]
            short_desc = d["short_description"]
            t = d["type"]
            if long_desc in lookup:
                raise ValueError(f"Duplicate long description: {long_desc}")
            lookup[long_desc] = (t, short_desc)
            if short_desc != long_desc:
                lookup[short_desc] = (t, long_desc)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(lookup, f, indent=2)

if __name__ == "__main__":
    main()



