from rest_framework import serializers
from .models import TextFeatures, ImageFeatures, AudioFeatures


class TextSearchQuerySerializer(serializers.Serializer):
    """Validates the query parameters for text-based searches."""

    q = serializers.CharField(
        max_length=200, required=True, help_text="The text search query."
    )
    type = serializers.ChoiceField(
        choices=["text", "image", "audio"],
        required=True,
        help_text="The type of content to search for.",
    )
    limit = serializers.IntegerField(
        default=10, min_value=1, max_value=50, help_text="Number of results to return."
    )


class FileSearchQuerySerializer(serializers.Serializer):
    """Validates the form data for file-based searches."""

    file = serializers.FileField(
        required=True, help_text="The image or audio file to search with."
    )
    type = serializers.ChoiceField(
        choices=["image", "audio"],
        required=True,
        help_text="The type of the uploaded file.",
    )
    limit = serializers.IntegerField(
        default=10, min_value=1, max_value=50, help_text="Number of results to return."
    )


class TextResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = TextFeatures
        fields = ["source_page_url", "content"]


class ImageResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageFeatures
        fields = ["source_page_url", "asset_url", "alt_text"]


class AudioResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = AudioFeatures
        fields = [
            "source_page_url",
            "asset_url",
            "begin_stamp_seconds",
            "end_stamp_seconds",
        ]
