"""Client for the local offline audio.cpp server with required Perth watermarking."""
from __future__ import annotations
import argparse
import io
import json
import threading
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import numpy as np


class VoiceClient:
    """One configured offline voice. Returned WAVs are watermarked and checked locally."""
    def __init__(self, url='http://127.0.0.1:8766', model='step150', timeout=300, watermarker=None):
        parsed = urlsplit(url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or parsed.username or parsed.password:
            raise ValueError('VoiceClient only supports local HTTP servers')
        self.url = url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._watermarker = watermarker
        self._lock = threading.Lock()

    def synthesize(self, text, seed=35):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Text must not be empty')
        payload = json.dumps({'model': self.model, 'input': text, 'seed': seed}).encode('utf-8')
        request = Request(self.url + '/v1/audio/speech', data=payload,
                          headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urlopen(request, timeout=self.timeout) as response:
                if response.headers.get_content_type() not in ('audio/wav', 'audio/x-wav'):
                    raise RuntimeError('Server did not return WAV audio; use an offline session')
                raw = response.read()
        except HTTPError as exc:
            raise RuntimeError(f'Server returned HTTP {exc.code}: {exc.read(4096).decode("utf-8", errors="replace")}') from exc
        return self.watermark_wav(raw)

    def watermark_wav(self, raw):
        import soundfile as sf
        with self._lock:
            audio, rate = sf.read(io.BytesIO(raw), dtype='float32')
            if rate != 48000 or audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                raise ValueError('Expected nonempty, finite 48 kHz mono WAV')
            if self._watermarker is None:
                import perth
                if perth.PerthImplicitWatermarker is None:
                    raise RuntimeError('Perth implicit watermarker is unavailable')
                self._watermarker = perth.PerthImplicitWatermarker(device='cpu')
            result = np.asarray(self._watermarker.apply_watermark(audio, sample_rate=rate), dtype=np.float32)
            if result.shape != audio.shape or not np.isfinite(result).all():
                raise RuntimeError('Watermarker returned invalid audio')
            output = io.BytesIO()
            sf.write(output, result, rate, format='WAV', subtype='PCM_16')
            marked, _ = sf.read(io.BytesIO(output.getvalue()), dtype='float32')
            score = np.asarray(self._watermarker.get_watermark(marked, rate, round=False))
            if score.size == 0 or not np.isfinite(score).all() or float(score.mean()) < 0.5:
                raise RuntimeError('Perth watermark verification failed; no audio returned')
            return output.getvalue()


def main():
    p = argparse.ArgumentParser(description='Generate a locally watermarked WAV using an offline audio.cpp session')
    p.add_argument('--text', required=True)
    p.add_argument('--output', required=True, help='New WAV path; existing files are not overwritten')
    p.add_argument('--url', default='http://127.0.0.1:8766')
    p.add_argument('--model', default='step150')
    p.add_argument('--seed', type=int, default=35)
    args = p.parse_args()
    audio = VoiceClient(url=args.url, model=args.model).synthesize(args.text, args.seed)
    with open(args.output, 'xb') as f:
        f.write(audio)
    print(json.dumps({'output': args.output, 'watermark': 'perth-implicit', 'watermark_detected': True}))

if __name__ == '__main__':
    main()
