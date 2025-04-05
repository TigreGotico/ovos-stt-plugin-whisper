import torch
from ovos_plugin_manager.templates.stt import STT
from ovos_utils.log import LOG
from transformers import WhisperFeatureExtractor
from transformers import WhisperForConditionalGeneration
from transformers import WhisperTokenizer
from transformers import pipeline


class WhisperSTT(STT):
    MODELS = ["openai/whisper-tiny",
              "openai/whisper-base",
              "openai/whisper-small",
              "openai/whisper-medium",
              "openai/whisper-tiny.en",
              "openai/whisper-base.en",
              "openai/whisper-small.en",
              "openai/whisper-medium.en",
              "openai/whisper-large",
              "openai/whisper-large-v2",
              "openai/whisper-large-v3",
              "openai/whisper-large-v3-turbo"]
    LANGUAGES = {
        "en": "english",
        "zh": "chinese",
        "de": "german",
        "es": "spanish",
        "ru": "russian",
        "ko": "korean",
        "fr": "french",
        "ja": "japanese",
        "pt": "portuguese",
        "tr": "turkish",
        "pl": "polish",
        "ca": "catalan",
        "nl": "dutch",
        "ar": "arabic",
        "sv": "swedish",
        "it": "italian",
        "id": "indonesian",
        "hi": "hindi",
        "fi": "finnish",
        "vi": "vietnamese",
        "iw": "hebrew",
        "uk": "ukrainian",
        "el": "greek",
        "ms": "malay",
        "cs": "czech",
        "ro": "romanian",
        "da": "danish",
        "hu": "hungarian",
        "ta": "tamil",
        "no": "norwegian",
        "th": "thai",
        "ur": "urdu",
        "hr": "croatian",
        "bg": "bulgarian",
        "lt": "lithuanian",
        "la": "latin",
        "mi": "maori",
        "ml": "malayalam",
        "cy": "welsh",
        "sk": "slovak",
        "te": "telugu",
        "fa": "persian",
        "lv": "latvian",
        "bn": "bengali",
        "sr": "serbian",
        "az": "azerbaijani",
        "sl": "slovenian",
        "kn": "kannada",
        "et": "estonian",
        "mk": "macedonian",
        "br": "breton",
        "eu": "basque",
        "is": "icelandic",
        "hy": "armenian",
        "ne": "nepali",
        "mn": "mongolian",
        "bs": "bosnian",
        "kk": "kazakh",
        "sq": "albanian",
        "sw": "swahili",
        "gl": "galician",
        "mr": "marathi",
        "pa": "punjabi",
        "si": "sinhala",
        "km": "khmer",
        "sn": "shona",
        "yo": "yoruba",
        "so": "somali",
        "af": "afrikaans",
        "oc": "occitan",
        "ka": "georgian",
        "be": "belarusian",
        "tg": "tajik",
        "sd": "sindhi",
        "gu": "gujarati",
        "am": "amharic",
        "yi": "yiddish",
        "lo": "lao",
        "uz": "uzbek",
        "fo": "faroese",
        "ht": "haitian creole",
        "ps": "pashto",
        "tk": "turkmen",
        "nn": "nynorsk",
        "mt": "maltese",
        "sa": "sanskrit",
        "lb": "luxembourgish",
        "my": "myanmar",
        "bo": "tibetan",
        "tl": "tagalog",
        "mg": "malagasy",
        "as": "assamese",
        "tt": "tatar",
        "haw": "hawaiian",
        "ln": "lingala",
        "ha": "hausa",
        "ba": "bashkir",
        "jw": "javanese",
        "su": "sundanese",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_id = self.config.get("model") or "openai/whisper-large-v3-turbo"
        if not self.config.get("ignore_warnings", False):
            valid_model = model_id in self.MODELS
            if not valid_model:
                LOG.info(f"{model_id} is not default model_id ({self.MODELS}), "
                         f"assuming huggingface repo_id or path to local model")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        device = "cpu"
        if self.config.get("use_cuda"):
            if not torch.cuda.is_available():
                LOG.error("CUDA is not available, running on CPU. inference will be SLOW!")
            else:
                model.to("cuda")
                device = "cuda"
        else:
            LOG.warning("running on CPU. inference will be SLOW! "
                        "consider passing '\"use_cuda\": True' to the plugin config")
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            device=device,
            feature_extractor=feature_extractor,
            tokenizer=self.tokenizer,
            chunk_length_s=30,
            stride_length_s=(4, 2)
        )

    def execute(self, audio, language=None):
        lang = language or self.lang
        lang = lang.split("-")[0]
        if lang != "auto" and lang not in self.LANGUAGES:
            return ValueError(f"Unsupported language: {lang}")
        if lang == "auto":
            LOG.debug("Auto-detecting language")
            result = self.pipe(audio.get_wav_data())
        else:
            result = self.pipe(audio.get_wav_data(),
                               generate_kwargs={"language":self.LANGUAGES[lang]})
        return result["text"]

    @property
    def available_languages(self) -> set:
        return set(WhisperSTT.LANGUAGES.keys())


if __name__ == "__main__":
    b = WhisperSTT({"use_cuda": True, "model": "openai/whisper-large-v3-turbo"})

    from speech_recognition import Recognizer, AudioFile

    jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-whisper/jfk.wav"
    with AudioFile(jfk) as source:
        audio = Recognizer().record(source)

    a = b.execute(audio, language="es")
    print(a)
    # And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
