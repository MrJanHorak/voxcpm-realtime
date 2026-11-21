"""CLI for Phase 2 one-shot synthesis."""
import argparse
import json
from .inference import synthesize


def main():
    parser = argparse.ArgumentParser(description="Generate a WAV using a local VoxCPM2 base model and LoRA")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--voice-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--reference-wav")
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()
    print(json.dumps(synthesize(model_dir=args.model_dir, voice_dir=args.voice_dir,
                                text=args.text, output=args.output,
                                reference_wav=args.reference_wav, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
