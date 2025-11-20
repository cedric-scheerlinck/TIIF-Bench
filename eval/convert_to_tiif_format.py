#!/usr/bin/env python3
"""
Convert images from sample_%05d/ directory structure to TIIF-Bench format.

Original structure:
    sample_00001/
        - image.png
        - prompt.txt

Target structure:
    output/
        <dimension>/
            <model_name>/
                short_description/
                    0.png
                long_description/
                    0.png
"""

import argh
import json
import shutil
from pathlib import Path
from tqdm import tqdm


def load_reverse_lookup(lookup_path: str = "data/reverse_lookup.json") -> dict[str, tuple[str, str]]:
    """Load reverse_lookup.json file."""
    lookup_file = Path(lookup_path)
    if not lookup_file.exists():
        raise FileNotFoundError(f"Reverse lookup file not found: {lookup_path}")
    
    with lookup_file.open("r") as f:
        lookup = json.load(f)

    lookup = {k: tuple(v) for k, v in lookup.items()}

    assert all(len(v) == 2 for v in lookup.values()), "Invalid lookup format"

    print(f"Loaded {len(lookup)} prompt mappings from {lookup_path}")
    return lookup


def find_prompt_line_number(
    prompt: str,
    dimension: str,
    jsonl_dir: Path,
    prompt_key: str
) -> int | None:
    """
    Find the line number (0-indexed) of a prompt in the corresponding JSONL file.
    
    Returns:
        Line number if found, None otherwise
    """
    # JSONL file name format: {dimension}_prompts.jsonl
    jsonl_file = jsonl_dir / f"{dimension}_prompts.jsonl"
    
    if not jsonl_file.exists():
        return None
    
    with jsonl_file.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # if data.get(prompt_key).strip() == prompt:
                if data.get(prompt_key).strip() in prompt:
                    return line_idx
            except json.JSONDecodeError:
                continue
    
    return None


def find_sample_directories(input_dir: Path) -> list[Path]:
    """Find all directories matching sample_%05d pattern."""
    sample_dirs: list[Path] = []
    
    for item in input_dir.iterdir():
        if item.is_dir() and item.name.startswith("sample_"):
            sample_dirs.append(item)
    
    sample_dirs.sort(key=lambda x: int(x.name.split("_")[-1]))
    return sample_dirs


def extract_sample_number(sample_dir: Path) -> int | None:
    """Extract sample number from directory name (e.g., sample_00001 -> 1)."""
    name = sample_dir.name
    if name.startswith("sample_"):
        num_str = name.split("_")[-1]
        if num_str.isdigit():
            return int(num_str)
    return None


def read_prompt_file(prompt_path: Path) -> str | None:
    """Read prompt text from prompt.txt file."""
    prompt_file = Path(prompt_path)
    if not prompt_file.exists():
        return None
    
    with prompt_file.open("r") as f:
        prompt = f.read().strip()
    
    return prompt


def get_from_lookup(lookup: dict[str, tuple[str, str]], prompt: str) -> tuple[str, str] | None:
    """get value of lookup[key] if key in prompt"""
    value = None
    for key in lookup:
        if key in prompt:
            if value is not None:
                raise ValueError(f"Multiple keys found in prompt: {key} and {value}")
            value = lookup[key]
    return value

def process_sample_directory(
    sample_dir: Path,
    lookup: dict[str, tuple[str, str]],
    output_dir: Path,
    model_name: str,
    prompt_key: str,
    jsonl_dir: Path,
) -> tuple[bool, str]:
    """
    Process a single sample directory.
    
    Returns:
        (success, message) tuple
    """
    # Read prompt
    prompt_path = sample_dir / "prompt.txt"
    prompt = read_prompt_file(prompt_path)
    if prompt is None:
        return False, f"prompt.txt not found in {sample_dir}"
    
    lookup_result = get_from_lookup(lookup, prompt)
    if lookup_result is None:
        return False, f"Prompt not found in reverse_lookup: {repr(prompt)}"

    dimension, _ = lookup_result
    
    # Find the line number in the JSONL file
    line_number = find_prompt_line_number(prompt, dimension, jsonl_dir, prompt_key)
    if line_number is None:
        return False, f"Prompt not found in JSONL file for dimension {dimension}: {sample_dir}"
    
    # Find image file
    src_image_path = sample_dir / "image.png"
    if not src_image_path.exists():
        return False, f"image.png not found in {sample_dir}"
    
    # Create output directory structure
    output_subdir = Path(output_dir) / dimension / model_name / prompt_key
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Copy image to destination (using line_number as index)
    dest_image_path = output_subdir / f"{line_number}.png"
    shutil.copy2(src_image_path, dest_image_path)
    
    return True, f"Processed {sample_dir} -> {dimension}/{model_name}/{prompt_key}/{line_number}.png"


@argh.arg("--desc-length", choices=["short", "long"])
def main(
    input_dir: str,
    output_name: str = "",
    output_dir: str = "output",
    desc_length: str = "long",
    jsonl_dir: str = "data/test_prompts",
) -> int:
    """
    Convert sample directories to TIIF-Bench format.
    """
    prompt_key = f"{desc_length}_description"
    input_path = Path(input_dir)
    if not output_name:
        output_name = input_path.name
    if not input_path.is_dir():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    # Validate JSONL directory
    jsonl_path = Path(jsonl_dir)
    if not jsonl_path.is_dir():
        print(f"Error: JSONL directory does not exist: {jsonl_dir}")
        return 1
    
    lookup = load_reverse_lookup()
    
    sample_dirs = find_sample_directories(input_path)
    if not sample_dirs:
        print(f"Warning: No sample directories found in {input_path}")
        return 1
    
    print(f"Found {len(sample_dirs)} sample directories")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Will save to {output_path}")
    
    # Process each sample directory
    success_count = 0
    error_count = 0
    errors = []
    
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        success, message = process_sample_directory(
            sample_dir,
            lookup,
            output_path,
            output_name,
            prompt_key,
            jsonl_path
        )
        
        if success:
            success_count += 1
        else:
            error_count += 1
            errors.append(message)
            print(f"Warning: {message}")
    
    # Print summary
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    print(f"\nOutput directory: {output_dir}")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    argh.dispatch_command(main)

