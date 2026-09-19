# whisper-tts

Transcribe an audio file with Deepgram, then correct the transcript with
OpenRouter.

```bash
uv run whisper-tts --deepgram-api-key="$(vault kv get -mount=secret -field=deepgram_api_key airflow)" --openrouter-api-key="$(vault kv get -mount=kv -field=openrouter_api_key puppet)" --input-file ~/Downloads/audio.mp3
```

The same command can be run through the new package module directly:

```bash
uv run python -m whisper_tts.cli --deepgram-api-key="$(vault kv get -mount=secret -field=deepgram_api_key airflow)" --openrouter-api-key="$(vault kv get -mount=kv -field=openrouter_api_key puppet)" --input-file ~/Downloads/audio.mp3
```

## Install

```bash
uv sync
```
