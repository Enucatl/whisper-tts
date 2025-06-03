# whisper-tts

transcribe an audio file with the whisper model and huggingface transformers

```bash
/opt/home/user/venv/whisper-tts/bin/python ~/src/whisper-tts/main.py --deepgram-api-key=$(vault kv get -mount=secret -field=deepgram_api_key airflow) --google-api-key $(vault kv get -mount=secret -field=google_aistudio_api_key airflow) --input-file  ~/Downloads/audio.mp3
```

## Install

```bash
uv pip install --python /opt/home/user/venv/whisper-tts/bin/python -e .
```
