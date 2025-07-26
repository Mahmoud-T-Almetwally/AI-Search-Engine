from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)

class SearchConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'search'

    def ready(self) -> None:

        from . import ai_models

        logger.info(msg="Search Config: Initializing AI models...")

        try:

            ai_models.text_extractor._ensure_model_loaded()
            ai_models.image_extractor._ensure_model_loaded()
            ai_models.audio_extractor._ensure_model_loaded()

        except ai_models.ModelLoadingError as e:

            logger.critical(msg=f"A critical error occurred while loading AI models: {e}")

            raise e

