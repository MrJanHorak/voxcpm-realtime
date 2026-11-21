import base64
import json
import pytest
from voxcpm_realtime.stream import read_events, StreamError


def event(value):
    return ('data: ' + (value if isinstance(value, str) else json.dumps(value)) + '\n\n').encode().splitlines(keepends=True)


def test_stream_yields_distinct_chunks_and_requires_completion():
    pcm1, pcm2 = b'\x01\x00', b'\x02\x00\x03\x00'
    lines = event({'type': 'speech.audio.delta', 'audio': base64.b64encode(pcm1).decode()})
    lines += event({'type': 'speech.audio.delta', 'audio': base64.b64encode(pcm2).decode()})
    lines += event({'type': 'speech.audio.done'}) + event('[DONE]')
    chunks = read_events(lines)
    assert next(chunks) == pcm1
    assert list(chunks) == [pcm2]


def test_rejects_truncated_and_server_errors():
    delta = event({'type': 'speech.audio.delta', 'audio': 'AAA='})
    with pytest.raises(StreamError, match='before completion'):
        list(read_events(delta))
    with pytest.raises(StreamError, match='server error'):
        list(read_events(event({'type': 'error', 'message': 'out of memory'})))


@pytest.mark.parametrize('value', ['???', 'AA==', ''])
def test_rejects_invalid_pcm(value):
    with pytest.raises(StreamError, match='audio chunk|PCM16'):
        list(read_events(event({'type': 'speech.audio.delta', 'audio': value})))


def test_done_without_audio_is_error():
    with pytest.raises(StreamError, match='before completing'):
        list(read_events(event({'type': 'speech.audio.done'}) + event('[DONE]')))
