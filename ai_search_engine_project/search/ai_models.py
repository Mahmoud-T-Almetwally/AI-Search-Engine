import logging
from typing import List
from abc import ABC, abstractmethod
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured, RequestAborted

logger = logging.getLogger(name=__name__)

try:

    from sentence_transformers import SentenceTransformer
    from transformers import CLIPProcessor, CLIPModel, ClapModel, ClapProcessor
    from PIL import Image
    import torch
    import numpy as np

except ImportError as e:
    raise ImproperlyConfigured(
        "Couldn't import a required AI/ML library. "
        "Please ensure your environment is set up correctly by running 'pip install -r requirements.txt'. "
        f"Original error: {e}"
    ) from e


class ModelLoadingError(Exception):
    """Custom exception for when an AI model fails to load."""

    pass


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(msg=f"Loading Models on: {"GPU" if DEVICE == "cuda" else "CPU"}")


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors."""

    _instance = None
    model = None
    processor = None
    model_name = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BaseFeatureExtractor, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.processor = None
        return cls._instance

    @abstractmethod
    def _load_model(self):
        """Each subclass must implement its own model loading logic."""
        pass

    def _ensure_model_loaded(self):
        """Public-facing method to trigger loading."""
        if self.model is None:
            try:
                self._load_model()
                logger.info(
                    f"Successfully loaded model and processor for {self.__class__.__name__}"
                )
            except Exception as e:
                # Raise our new custom exception
                raise ModelLoadingError(
                    f"Couldn't Load Model '{self.model_name}'. "
                    "Please ensure you have a working internet connection and enough VRAM/RAM. "
                    f"Original Error: {e}"
                ) from e


class TextFeatureExtractor(BaseFeatureExtractor):

    def _load_model(self):
        if self.model is None:
            model_name = settings.TEXT_MODEL_CONFIG["NAME"]
            logger.info(msg=f"Loading Text Model: {model_name}")
            self.model = SentenceTransformer(model_name, device=DEVICE)

    def extract_features(self, texts: List[str]):
        self._ensure_model_loaded()
        return self.model.encode(texts, batch_size=settings.TEXT_MODEL_CONFIG["BATCH_SIZE"]).tolist()  # type: ignore


class ImageFeatureExtractor(BaseFeatureExtractor):

    def _load_model(self):
        if self.model is None:
            model_name = settings.IMAGE_MODEL_CONFIG["NAME"]
            logger.info(msg=f"Loading Image Model: {model_name}")
            self.model = CLIPModel.from_pretrained(model_name).to(DEVICE)  # type: ignore
            self.processor = CLIPProcessor.from_pretrained(model_name)

    def extract_from_images(self, imgs: List[Image.Image]) -> List[List[float]]:
        self._ensure_model_loaded()
        inputs = self.processor(images=imgs, return_tensors="pt", padding=True).to(DEVICE)  # type: ignore

        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)  # type: ignore

        return image_embeddings.tolist()

    def extract_from_texts(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model_loaded()
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)  # type: ignore

        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)  # type: ignore

        return text_embeddings.cpu().tolist()


class AudioFeatureExtractor(BaseFeatureExtractor):

    def _load_model(self):
        if self.model is None:
            model_name = settings.AUDIO_MODEL_CONFIG["NAME"]
            logger.info(msg=f"Loading Audio Model: {model_name}")
            self.model = ClapModel.from_pretrained(model_name).to(DEVICE)  # type: ignore
            self.processor = ClapProcessor.from_pretrained(model_name)

    def extract_from_audios(self, audios: List[np.ndarray]) -> List[List[float]]:
        self._ensure_model_loaded()
        inputs = self.processor(audios=audios, return_tensors="pt", padding=True, sampling_rate=settings.AUDIO_MODEL_CONFIG["SAMPLING_RATE"]).to(DEVICE)  # type: ignore
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_features(**inputs)  # type: ignore

        return audio_embeddings.tolist()

    def extract_from_texts(self, texts: List[str]):
        self._ensure_model_loaded()
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)  # type: ignore
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)  # type: ignore

        return text_embeddings.cpu().tolist()


audio_extractor = AudioFeatureExtractor()
image_extractor = ImageFeatureExtractor()
text_extractor = TextFeatureExtractor()