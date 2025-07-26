from django.db import models
from pgvector.django import VectorField
from django.conf import settings


class BaseAsset(models.Model):
    id = models.BigIntegerField(primary_key=True, auto_created=True)
    asset_url = models.URLField(max_length=1024, blank=True, null=True, db_index=True)
    source_page_url = models.URLField(max_length=1024, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class ImageFeatures(BaseAsset):
    alt_text = models.CharField(max_length=512, blank=True)
    embedding = VectorField(dimensions=512)

    def save(self, *args, **kwargs):
        expected_dims = settings.IMAGE_MODEL_CONFIG.get("DIMENSIONS", 512)
        if not (self.embedding is None) and len(self.embedding) != expected_dims:
            raise ValueError(
                f"Embedding for ImageFeatures has incorrect dimensions. "
                f"Expected {expected_dims}, but got {len(self.embedding)}."
            )
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Image from: {self.source_page_url}"
    
    class Meta: # type: ignore
        constraints = [
            models.UniqueConstraint(fields=['asset_url'], name='unique_image_asset_url')
        ]
    
class TextFeatures(BaseAsset):
    content = models.TextField()
    embedding = VectorField(dimensions=384)

    def save(self, *args, **kwargs):
        expected_dims = settings.TEXT_MODEL_CONFIG.get("DIMENSIONS", 384)
        if not (self.embedding is None) and len(self.embedding) != expected_dims:
            raise ValueError(
                f"Embedding for TextFeatures has incorrect dimensions. "
                f"Expected {expected_dims}, but got {len(self.embedding)}."
            )
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Text Snippet from: {self.source_page_url}"


class AudioFeatures(BaseAsset):
    begin_stamp_seconds = models.PositiveIntegerField()
    end_stamp_seconds = models.PositiveIntegerField()
    
    embedding = VectorField(dimensions=512)

    def save(self, *args, **kwargs):
        expected_dims = settings.AUDIO_MODEL_CONFIG.get("DIMENSIONS", 512)
        if not (self.embedding is None) and len(self.embedding) != expected_dims:
            raise ValueError(
                f"Embedding for AudioFeatures has incorrect dimensions. "
                f"Expected {expected_dims}, but got {len(self.embedding)}."
            )
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Audio Chunk ({self.begin_stamp_seconds}s-{self.end_stamp_seconds}s) from: {self.source_page_url}"

    class Meta: # type: ignore
        constraints = [
            models.UniqueConstraint(fields=['asset_url', 'begin_stamp_seconds'], name='unique_audio_chunk')
        ]