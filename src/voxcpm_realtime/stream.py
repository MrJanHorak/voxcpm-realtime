"""Receive audio.cpp VoxCPM2 SSE audio as 48 kHz mono PCM16 chunks."""
from __future__ import annotations

import argparse
import base64
import binascii
from contextlib import ExitStack
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import wave

SAMPLE_RATE = 48000


class StreamError(RuntimeError):
    """A server error, invalid stream, or premature disconnect."""


def read_events(lines):
    """Parse the audio.cpp SSE protocol; fail on errors and incomplete streams."""
    fields = []
    done = False
    finished = False
    audio_seen = False
    for raw in lines:
        line = raw.decode('utf-8').rstrip('\r\n')
        if line:
            if line.startswith('data:'):
                fields.append(line[5:].lstrip(' '))
            continue
        if not fields:
            continue
        data = '\n'.join(fields)
        fields.clear()
        if data == '[DONE]':
            if not done or not audio_seen:
                raise StreamError('Server ended before completing audio')
            finished = True
            break
        try:
            event = json.loads(data)
        except (ValueError, TypeError) as exc:
            raise StreamError('Invalid SSE event') from exc
        if not isinstance(event, dict):
            raise StreamError('Invalid SSE event')
        kind = event.get('type')
        if kind == 'error':
            raise StreamError(f"Streaming server error: {event.get('message', event)}")
        if kind == 'speech.audio.delta':
            if done:
                raise StreamError('Audio received after completion')
            try:
                chunk = base64.b64decode(event['audio'], validate=True)
            except (KeyError, ValueError, TypeError, binascii.Error) as exc:
                raise StreamError('Invalid audio chunk') from exc
            if not chunk or len(chunk) % 2:
                raise StreamError('Invalid PCM16 audio chunk length')
            audio_seen = True
            yield chunk
        elif kind == 'speech.audio.done':
            if done:
                raise StreamError('Duplicate completion event')
            done = True
    if not finished:
        raise StreamError('Streaming response ended before completion')


def stream_speech(text, *, url='http://127.0.0.1:8766', model='step150', seed=35, timeout=300):
    """Yield signed 16-bit little-endian mono PCM chunks. Close iterator to cancel."""
    if not text.strip():
        raise ValueError('Text must not be empty')
    payload = json.dumps({'model': model, 'input': text, 'seed': seed,
                          'response_format': 'pcm', 'stream_format': 'sse',
                          'options': {'retry_badcase': False}}).encode('utf-8')
    request = Request(url.rstrip('/') + '/v1/audio/speech', data=payload,
                      headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'}, method='POST')
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.headers.get_content_type() != 'text/event-stream':
                raise StreamError('Server did not return an event stream; check model streaming mode')
            yield from read_events(response)
    except HTTPError as exc:
        detail = exc.read(4096).decode('utf-8', errors='replace')
        raise StreamError(f'Server returned HTTP {exc.code}: {detail}') from exc


def stream_watermarked_speech(text, *, url='http://127.0.0.1:8766', model='step150', seed=35, timeout=300, watermarker=None):
    """Yield locally watermarked and individually verified PCM16 segments.

    Earlier chunks may already be played if a later chunk fails. Close this
    iterator to release the HTTP connection when cancelling.
    """
    from urllib.parse import urlsplit
    from .stream_watermark import PerthChunkWatermarker, watermark_segments
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or parsed.username or parsed.password:
        raise ValueError('Watermarked streaming only supports local HTTP servers')
    marker = watermarker if watermarker is not None else PerthChunkWatermarker()
    source = stream_speech(text, url=url, model=model, seed=seed, timeout=timeout)
    try:
        yield from watermark_segments(source, marker)
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description='Stream VoxCPM2 speech from the local audio.cpp server')
    parser.add_argument('--text', required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--model', default='step150')
    parser.add_argument('--seed', type=int, default=35)
    parser.add_argument('--output', help='Optional new WAV file; existing files are not overwritten')
    parser.add_argument('--no-play', action='store_true', help='Receive without live playback')
    args = parser.parse_args()
    if args.no_play and not args.output:
        parser.error('Use --output when --no-play is specified')
    start = time.perf_counter()
    first = None
    count = total = 0
    with ExitStack() as stack:
        player = None
        if not args.no_play:
            try:
                import sounddevice
            except ImportError as exc:
                raise SystemExit('Install playback support: pip install -e ".[playback]"') from exc
            player = stack.enter_context(sounddevice.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16'))
        writer = None
        if args.output:
            f = stack.enter_context(open(args.output, 'xb'))
            writer = stack.enter_context(wave.open(f, 'wb'))
            writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(SAMPLE_RATE)
        chunks = stream_watermarked_speech(args.text, url=args.url, model=args.model, seed=args.seed)
        stack.callback(chunks.close)
        for chunk in chunks:
            if first is None:
                first = time.perf_counter() - start
            count += 1
            total += len(chunk)
            if writer: writer.writeframesraw(chunk)
            if player: player.write(chunk)
        elapsed = time.perf_counter() - start
    print(json.dumps({'first_audio_seconds': first, 'receive_seconds': elapsed,
                      'audio_seconds': total / 2 / SAMPLE_RATE, 'chunks': count,
                      'playback_enabled': not args.no_play, 'watermark': 'perth-implicit', 'chunk_detection_passed': True, 'output': args.output}, indent=2))


if __name__ == '__main__':
    main()
