"""Measure cold load and repeated generation; output contains no audio or text."""
import argparse
import json
import os
import subprocess
import threading
import time
from .manager import ModelManager


class GPUSampler:
    def __init__(self):
        self.samples = []
        self.shared_samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.shared_thread = threading.Thread(target=self._sample_shared, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
                used, total = map(int, result.stdout.strip().split(","))
                self.samples.append((used, total))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(0.5)

    def _sample_shared(self):
        if os.name != "nt":
            return
        # WDDM process counters; sum matching adapter instances for this process.
        command = ("$ErrorActionPreference='Stop'; $v=Get-Counter "
                   "'\\GPU Process Memory(*)\\Shared Usage'; "
                   f"$s=@($v.CounterSamples | Where-Object {{$_.InstanceName -like 'pid_{os.getpid()}_*'}}); "
                   "if ($s.Count -gt 0) {($s | Measure-Object -Property CookedValue -Sum).Sum}")
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                        capture_output=True, text=True, timeout=10, check=True)
                if result.stdout.strip():
                    self.shared_samples.append(int(float(result.stdout.strip())))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(2)

    def __enter__(self):
        self.thread.start()
        self.shared_thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self.shared_thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--voice-dir", required=True)
    parser.add_argument("--reference-wav")
    parser.add_argument("--text", default="Hello, this is a test of the voice engine.")
    parser.add_argument("--output", default="outputs/benchmark.wav")
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    import torch
    manager = ModelManager(args.model_dir, args.voice_dir, args.reference_wav)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with GPUSampler() as gpu:
        start = time.perf_counter()
        manager.warmup()
        warmup_seconds = time.perf_counter() - start
        runs = []
        for _ in range(args.runs):
            start = time.perf_counter()
            result = manager.write(args.text, args.output)
            elapsed = time.perf_counter() - start
            duration = result["duration_seconds"]
            runs.append({"seconds_including_wav_write": elapsed, "audio_seconds": duration,
                         "real_time_factor": elapsed / duration if duration else None})
    report = {"cold_model_load_seconds": manager.load_seconds, "first_warmup_seconds": warmup_seconds,
              "runs": runs, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
              "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
              "gpu_device_peak_used_mib_sampled": max((s[0] for s in gpu.samples), default=None),
              "gpu_device_total_mib": gpu.samples[0][1] if gpu.samples else None,
              "process_peak_shared_gpu_bytes_sampled": max(gpu.shared_samples, default=None),
              "shared_gpu_memory_note": "Windows WDDM process counters, summed across adapters; sampled, may miss peaks; null if unavailable.",
              "gpu_sampling_note": "Device-wide nvidia-smi samples include other processes and can miss peaks."}
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
