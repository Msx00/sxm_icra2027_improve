import argparse

def inference_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, help="pair directory, dataset root, or JSONL manifest")
    parser.add_argument("--output", default="results")
    parser.add_argument("--overwrite", action="store_true")
    return parser
