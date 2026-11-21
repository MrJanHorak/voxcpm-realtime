import argparse
import json
from .validator import inspect_voice

def main():
    parser = argparse.ArgumentParser(description="Inspect a VoxCPM2 voice adapter folder")
    parser.add_argument("folder", help="Folder containing lora_config.json and lora_weights.safetensors")
    args = parser.parse_args()
    result = inspect_voice(args.folder)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["compatible"] else 1)

if __name__ == "__main__":
    main()
