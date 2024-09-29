## Description

OpenVoiceOS STT plugin for [Whisper](https://github.com/guillaumekln/faster-whisper), using transformers library


## Install

`pip install ovos-stt-plugin-whisper`

## Configuration

to use Large model with GPU

```json
  "stt": {
    "module": "ovos-stt-plugin-whisper",
    "ovos-stt-plugin-whisper": {
        "model": "openai/whisper-large-v3",
        "use_cuda": true
    }
  }
```

you can also pass a full path to a local model or any huggingface repo_id,
eg. `"projecte-aina/whisper-large-v3-ca-3catparla"`
