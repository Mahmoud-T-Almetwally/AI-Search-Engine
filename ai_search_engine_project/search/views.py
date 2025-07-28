from rest_framework.decorators import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from pgvector.django import L2Distance

from PIL import Image
import librosa

from .serializers import (
    TextSearchQuerySerializer, FileSearchQuerySerializer,
    TextResultSerializer, ImageResultSerializer, AudioResultSerializer
)
from .models import TextFeatures, ImageFeatures, AudioFeatures
from .ai_models import text_extractor, image_extractor, audio_extractor


class MultiModalSearchView(APIView):
    """
    A unified search API for text, image, and audio content.
        - Use GET with 'q' and 'type' for text-to-X searches.
        - Use POST with a 'file' and 'type' for file-to-X searches.
    """

    async def get(self, request, *args, **kwargs):
        """Handles all text-to-X searches."""
        
        serializer = TextSearchQuerySerializer(data=request.query_params)
        if not serializer.is_valid():
            return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        validated_data = serializer.validated_data
        
        query_text = validated_data['q']
        search_type = validated_data['type']
        limit = validated_data['limit']

        if search_type == "text":
            query_embedding = text_extractor.extract_features([query_text])[0]
            
            results = TextFeatures.objects.order_by(L2Distance("embedding", query_embedding))[:limit]

            output_serializer = TextResultSerializer(results, many=True)

        elif search_type == "image":
            query_embedding = image_extractor.extract_from_texts([query_text])[0]

            results = ImageFeatures.objects.order_by(L2Distance("embedding", query_embedding))[:limit]

            output_serializer = ImageResultSerializer(results, many=True)

        elif search_type == "audio":
            query_embedding = audio_extractor.extract_from_texts([query_text])[0]

            results = AudioFeatures.objects.order_by(L2Distance("embedding", query_embedding))[:limit]

            output_serializer = AudioResultSerializer(results, many=True)

        else:
            return Response({"error":"Invalid search type"}, status=status.HTTP_400_BAD_REQUEST)
        
        return Response(output_serializer.data, status=status.HTTP_200_OK)
        

    async def post(self, request, *args, **kwargs):
        """Handles all file-to-X searches (image-to-image, audio-to-audio)."""
        
        serializer = FileSearchQuerySerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        validated_data = serializer.validated_data

        query_file = validated_data['file']
        search_type = validated_data['type']
        limit = validated_data['limit']

        query_embedding = None

        if search_type == "image":
            try:

                img = Image.open(query_file)
                img.verify()
                img = Image.open(query_file)

                query_embedding = image_extractor.extract_from_images([img])[0]

                results = ImageFeatures.objects.order_by(L2Distance("embedding", query_embedding))[:limit]

                output_serialzer = ImageResultSerializer(results, many=True)

            except Exception as e :
                return Response({"error":"Invalid or Corrupted Image file"}, status=status.HTTP_400_BAD_REQUEST)

        elif search_type == "audio":
            try:

                sampling_rate = settings.AUDIO_MODEL_CONFIG.get("SAMPLING_RATE", 48000)

                audio_waveform, _ = librosa.load(query_file, sr=sampling_rate)

                query_embedding = audio_extractor.extract_from_audios([audio_waveform])[0]

                results = AudioFeatures.objects.order_by(L2Distance("embedding", query_embedding))[:limit]

                output_serialzer = AudioResultSerializer(results, many=True)
            
            except Exception as e:
                return Response({"error":"Invalid or Corrupted Audio file"}, status=status.HTTP_400_BAD_REQUEST)

        else:
            return Response({"error":"Invalid Search Type for file upload"}, status=status.HTTP_400_BAD_REQUEST)
        
        query_file.close()

        return Response(output_serialzer.data, status=status.HTTP_200_OK)
        

        