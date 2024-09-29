import numpy as np
import pytest
from speech_recognition import AudioFile, Recognizer

from ovos_stt_plugin_whisper import WhisperLangClassifier, WhisperSTT


@pytest.fixture
def audio_data():
    recognizer = Recognizer()
    with AudioFile("jfk.wav") as source:
        return recognizer.record(source)


def test_faster_whisper_stt_execute(audio_data):
    stt = WhisperSTT()
    transcription = stt.execute(audio_data, language="en")
    assert isinstance(transcription, str)
    assert len(transcription) > 0


def test_faster_whisper_stt_available_languages():
    stt = WhisperSTT()
    available_languages = stt.available_languages
    assert isinstance(available_languages, set)
    assert "en" in available_languages


def test_faster_whisper_lang_classifier_detect(audio_data):
    classifier = WhisperLangClassifier()
    language, probability = classifier.detect(audio_data.get_wav_data())
    assert isinstance(language, str)
    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0


def test_faster_whisper_lang_classifier_audiochunk2array():
    audio_data = b"\x00\x01\x02\x03"
    array = WhisperLangClassifier.audiochunk2array(audio_data)
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32


def test_faster_whisper_stt_audiodata2array(audio_data):
    array = WhisperSTT.audiodata2array(audio_data)
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32


def test_faster_whisper_stt_invalid_model():
    stt = WhisperSTT(config={"model": "invalid_model"})
    assert stt.config["model"] == "small"


def test_faster_whisper_lang_classifier_invalid_model():
    classifier = WhisperLangClassifier(config={"model": "invalid_model"})
    assert classifier.config["model"] == "small"

if __name__ == "__main__":
    pytest.main()
