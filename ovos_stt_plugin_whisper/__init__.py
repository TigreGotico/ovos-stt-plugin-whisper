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
              "openai/whisper-large-v3"]
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
        model_id = self.config.get("model") or "openai/whisper-tiny"
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
        if lang in self.LANGUAGES:
            lang = self.LANGUAGES[lang]
            forced_decoder_ids = self.tokenizer.get_decoder_prompt_ids(language=lang,
                                                                       task="transcribe")
            result = self.pipe(audio.get_wav_data(),
                               generate_kwargs={"forced_decoder_ids": forced_decoder_ids})
        else:
            result = self.pipe(audio.get_wav_data())
        return result["text"]

    @property
    def available_languages(self) -> set:
        return set(WhisperSTT.LANGUAGES.keys())


class MyNorthAISTT(STT):
    MODELS = ["my-north-ai/whisper-small-pt",
              "my-north-ai/whisper-medium-pt",
              "my-north-ai/whisper-large-v3-pt"]

    def __init__(self, config=None):
        super().__init__(config)
        model_id = self.config.get("model") or "my-north-ai/whisper-small-pt"
        if model_id == "small":
            model_id = "my-north-ai/whisper-small-pt"
        elif model_id == "medium":
            model_id = "my-north-ai/whisper-medium-pt"
        elif model_id == "large" or model_id == "large-v3":
            model_id = "my-north-ai/whisper-large-v3-pt"
        self.config["model"] = model_id
        self.config["lang"] = "pt"
        self.config["ignore_warnings"] = True
        valid_model = model_id in self.MODELS
        if not valid_model:
            LOG.info(f"{model_id} is not default model_id ({self.MODELS}), "
                     f"assuming huggingface repo_id or path to local model")
        self.stt = WhisperSTT(self.config)

    def execute(self, audio, language=None):
        return self.stt.execute(audio, language)

    @property
    def available_languages(self) -> set:
        return {"pt"}


class XabierZuazoSTT(STT):
    MODELS = ["zuazo/whisper-tiny-pt",
              "zuazo/whisper-tiny-pt-old",
              "zuazo/whisper-tiny-gl",
              "zuazo/whisper-tiny-eu",
              "zuazo/whisper-tiny-eu-from-es",
              "zuazo/whisper-tiny-eu-cv16_1",
              "zuazo/whisper-tiny-es",
              "zuazo/whisper-tiny-ca",
              "zuazo/whisper-small-pt",
              "zuazo/whisper-small-pt-old",
              "zuazo/whisper-small-gl",
              "zuazo/whisper-small-eu",
              "zuazo/whisper-small-eu-from-es",
              "zuazo/whisper-small-eu-cv16_1",
              "zuazo/whisper-small-es",
              "zuazo/whisper-small-ca",
              "zuazo/whisper-base-pt",
              "zuazo/whisper-base-pt-old",
              "zuazo/whisper-base-gl",
              "zuazo/whisper-base-eu",
              "zuazo/whisper-base-eu-from-es",
              "zuazo/whisper-base-eu-cv16_1",
              "zuazo/whisper-base-es",
              "zuazo/whisper-base-ca",
              "zuazo/whisper-medium-pt",
              "zuazo/whisper-medium-pt-old",
              "zuazo/whisper-medium-gl",
              "zuazo/whisper-medium-eu",
              "zuazo/whisper-medium-eu-from-es",
              "zuazo/whisper-medium-eu-cv16_1",
              "zuazo/whisper-medium-es",
              "zuazo/whisper-medium-ca",
              "zuazo/whisper-large-pt",
              "zuazo/whisper-large-pt-old",
              "zuazo/whisper-large-gl",
              "zuazo/whisper-large-eu",
              "zuazo/whisper-large-eu-from-es",
              "zuazo/whisper-large-eu-cv16_1",
              "zuazo/whisper-large-es",
              "zuazo/whisper-large-ca",
              "zuazo/whisper-large-v2-pt",
              "zuazo/whisper-large-v2-gl",
              "zuazo/whisper-large-v2-eu",
              "zuazo/whisper-large-v2-eu-from-es",
              "zuazo/whisper-large-v2-eu-cv16_1",
              "zuazo/whisper-large-v2-es",
              "zuazo/whisper-large-v2-ca",
              "zuazo/whisper-large-v2-pt-old",
              "zuazo/whisper-large-v3-pt",
              "zuazo/whisper-large-v3-gl",
              "zuazo/whisper-large-v3-eu",
              "zuazo/whisper-large-v3-eu-from-es",
              "zuazo/whisper-large-v3-eu-cv16_1",
              "zuazo/whisper-large-v3-es",
              "zuazo/whisper-large-v3-ca",
              "zuazo/whisper-large-v3-pt-old"]

    def __init__(self, config=None):
        super().__init__(config)
        model_id = self.config.get("model")
        l = self.lang.split("-")[0]
        if not model_id and l in ["pt", "es", "ca", "eu", "gl"]:
            model_id = f"zuazo/whisper-small-{l}"
        if not model_id:
            raise ValueError("invalid model")
        if model_id == "small":
            model_id = f"zuazo/whisper-small-{l}"
        elif model_id == "medium":
            model_id = f"zuazo/whisper-medium-{l}"
        elif model_id == "large" or model_id == "large-v3":
            model_id = f"zuazo/whisper-large-v3-{l}"
        elif model_id == "large-v2":
            model_id = f"zuazo/whisper-large-v2-{l}"
        elif model_id == "large-v1":
            model_id = f"zuazo/whisper-large-{l}"
        self.config["model"] = model_id
        self.config["lang"] = l
        self.config["ignore_warnings"] = True
        valid_model = model_id in self.MODELS
        if not valid_model:
            LOG.info(f"{model_id} is not default model_id ({self.MODELS}), "
                     f"assuming huggingface repo_id or path to local model")
        self.stt = WhisperSTT(self.config)

    def execute(self, audio, language=None):
        return self.stt.execute(audio, language)

    @property
    def available_languages(self) -> set:
        return {"pt"}


class ProjectAINAWhisperSTT(STT):
    def __init__(self, config=None):
        super().__init__(config)
        model_id = "projecte-aina/whisper-large-v3-ca-3catparla"
        self.config["model"] = model_id
        self.config["lang"] = "ca"
        self.config["ignore_warnings"] = True
        self.stt = WhisperSTT(self.config)

    def execute(self, audio, language=None):
        return self.stt.execute(audio, language)

    @property
    def available_languages(self) -> set:
        return {"ca"}


if __name__ == "__main__":
    b = ProjectAINAWhisperSTT()

    from speech_recognition import Recognizer, AudioFile

    jfk = "/home/miro/PycharmProjects/ovos-stt-plugin-whisper/jfk.wav"
    with AudioFile(jfk) as source:
        audio = Recognizer().record(source)

    a = b.execute(audio, language="en")
    print(a)
    # And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.
