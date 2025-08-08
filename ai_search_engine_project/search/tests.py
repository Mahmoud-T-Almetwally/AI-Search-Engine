import numpy as np
from PIL import Image
from scipy.io import wavfile
import tempfile
import requests
import io
from unittest.mock import patch

from rest_framework.test import APITestCase
from rest_framework import status

from django.test import TestCase
from django.db.utils import IntegrityError
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings

from .models import TextFeatures, ImageFeatures, AudioFeatures
from .serializers import FileSearchQuerySerializer
from .ai_models import TextFeatureExtractor, ImageFeatureExtractor, AudioFeatureExtractor, text_extractor, image_extractor, audio_extractor
from .utils import chunk_audio

class TextFeaturesModelTest(TestCase):
    """
        Test suite for the TextFeatures model.
    """

    def test_string_representation(self):
        text_content = "this is a test snippet for our search engine"
        vector_dimension = settings.TEXT_MODEL_CONFIG.get("DIMENSIONS", 384)

        test_vector = np.random.rand(vector_dimension).tolist()

        TextFeatures.objects.create(
            id=1,
            source_page_url="http://example.com/page1",
            asset_url="http://example.com/page1#snippet1",
            content=text_content,
            embedding=test_vector,
        )

        retrieved_features = TextFeatures.objects.first()

        self.assertEqual(str(retrieved_features), f"Text Snippet from: {"http://example.com/page1"}")

    def test_create_and_retrieve(self):

        text_content = "this is a test snippet for our search engine"
        vector_dimension = settings.TEXT_MODEL_CONFIG.get("DIMENSIONS", 384)

        test_vector = np.random.rand(vector_dimension).tolist()

        TextFeatures.objects.create(
            id=1,
            source_page_url="http://example.com/page1",
            asset_url="http://example.com/page1#snippet1",
            content=text_content,
            embedding=test_vector,
        )

        self.assertEqual(TextFeatures.objects.count(), 1)

        retrieved_features = TextFeatures.objects.first()

        self.assertEqual(retrieved_features.content, text_content)
        self.assertEqual(retrieved_features.source_page_url, "http://example.com/page1")

        retrieved_vector = np.array(retrieved_features.embedding)
        
        np.testing.assert_allclose(retrieved_vector, test_vector, rtol=1e-6)

    def test_invalid_embedding_dimensions(self):

        text_content = "this is a test snippet for our search engine"
        invalid_vector_dimension = 10

        invalid_vector = np.random.rand(invalid_vector_dimension).tolist()

        with self.assertRaises(ValueError):
            TextFeatures.objects.create(
                id=1,
                source_page_url="http://example.com/page1",
                asset_url="http://example.com/page1#snippet1",
                content=text_content,
                embedding=invalid_vector
            )


class ImageFeaturesModelTest(TestCase):
    """
        Test Suite for the ImageFeatures model.
    """

    def test_string_representation(self):
        alt_text = "this is a test snippet for our search engine"
        vector_dimensions = settings.IMAGE_MODEL_CONFIG.get("DIMENSIONS", 512)

        test_vector = np.random.rand(vector_dimensions)

        ImageFeatures.objects.create(
            id=1,
            source_page_url="http://example.com/page1",
            asset_url="http://example.com/page1/snippet1.png",
            alt_text=alt_text,
            embedding=test_vector,
        )

        retrieved_features = ImageFeatures.objects.first()

        self.assertEqual(str(retrieved_features), f"Image from: {"http://example.com/page1"}")

    def test_create_and_retrieve(self):

        alt_text = "this is a test snippet for our search engine"
        vector_dimensions = settings.IMAGE_MODEL_CONFIG.get("DIMENSIONS", 512)

        test_vector = np.random.rand(vector_dimensions)

        ImageFeatures.objects.create(
            id=1,
            source_page_url="http://example.com/page1",
            asset_url="http://example.com/page1/snippet1.png",
            alt_text=alt_text,
            embedding=test_vector,
        )

        self.assertEqual(ImageFeatures.objects.count(), 1)

        retrieved_features = ImageFeatures.objects.first()

        self.assertEqual(retrieved_features.alt_text, alt_text)
        self.assertEqual(retrieved_features.asset_url, "http://example.com/page1/snippet1.png")

        retrieved_vector = np.array(retrieved_features.embedding)

        np.testing.assert_allclose(retrieved_vector, test_vector, rtol=1e-6)

    def test_unique_asset_url_constraint(self):
        alt_text = "this is a test snippet for our search engine"
        vector_dimensions = settings.IMAGE_MODEL_CONFIG.get("DIMENSIONS", 512)

        test_vector = np.random.rand(vector_dimensions)

        with self.assertRaises(IntegrityError):
            ImageFeatures.objects.create(
                id=1,
                source_page_url="http://example.com/page1",
                asset_url="http://example.com/page1/snippet1.png",
                alt_text=alt_text,
                embedding=test_vector,
            )

            ImageFeatures.objects.create(
                id=2,
                source_page_url="http://example.com/page1",
                asset_url="http://example.com/page1/snippet1.png",
                alt_text=alt_text,
                embedding=test_vector,
            )

    def test_invalid_embedding_dimensions(self):
        
        alt_text = "this is a test snippet for our search engine"
        invalid_vector_dimensions = 10

        invalid_vector = np.random.rand(invalid_vector_dimensions).tolist()

        with self.assertRaises(ValueError):
            ImageFeatures.objects.create(
                id=1,
                source_page_url="http://example.com/page1",
                asset_url="http://example.com/page1/snippet1.png",
                alt_text=alt_text,
                embedding=invalid_vector
            )


class AudioFeaturesModelTest(TestCase):
    """
        Test Suite for the AudioFeatures model.
    """

    def test_string_representation(self):

        begin_stamp = 0
        end_stamp = begin_stamp + settings.AUDIO_MODEL_CONFIG.get("INPUT_LEN_SECONDS", 20)

        vector_dimensions = settings.AUDIO_MODEL_CONFIG.get("DIMENSIONS", 512)

        test_vector = np.random.rand(vector_dimensions)

        AudioFeatures.objects.create(
            id=1,
            source_page_url="http://example.com/page1",
            asset_url="http://example.com/page1/snippet1.mp3",
            embedding=test_vector,
            begin_stamp_seconds=begin_stamp,
            end_stamp_seconds=end_stamp
        )

        retrieved_features = AudioFeatures.objects.first()

        self.assertEqual(str(retrieved_features), f"Audio Chunk ({begin_stamp}s-{end_stamp}s) from: {"http://example.com/page1"}")

    def test_create_and_retrieve(self):

        begin_stamp = 0
        end_stamp = begin_stamp + settings.AUDIO_MODEL_CONFIG.get("INPUT_LEN_SECONDS", 20)

        vector_dimensions = settings.AUDIO_MODEL_CONFIG.get("DIMENSIONS", 512)

        test_vector = np.random.rand(vector_dimensions)

        AudioFeatures.objects.create(
            id=1,
            source_page_url="http://example.com/page1",
            asset_url="http://example.com/page1/snippet1.mp3",
            embedding=test_vector,
            begin_stamp_seconds=begin_stamp,
            end_stamp_seconds=end_stamp
        )

        self.assertEqual(AudioFeatures.objects.count(), 1)

        retrieved_features = AudioFeatures.objects.first()

        self.assertEqual(retrieved_features.begin_stamp_seconds, retrieved_features.end_stamp_seconds - settings.AUDIO_MODEL_CONFIG.get("INPUT_LEN_SECONDS", 20))
        self.assertEqual(retrieved_features.asset_url, "http://example.com/page1/snippet1.mp3")

        retrieved_vector = np.array(retrieved_features.embedding)

        np.testing.assert_allclose(retrieved_vector, test_vector, rtol=1e-6)

    def test_duplicate_asset_url_and_begin_stamp(self):
        begin_stamp = 0
        end_stamp = begin_stamp + settings.AUDIO_MODEL_CONFIG.get("INPUT_LEN_SECONDS", 20)

        vector_dimensions = settings.AUDIO_MODEL_CONFIG.get("DIMENSIONS", 512)

        test_vector = np.random.rand(vector_dimensions)

        with self.assertRaises(IntegrityError):
            AudioFeatures.objects.create(
                id=1,
                source_page_url="http://example.com/page1",
                asset_url="http://example.com/page1/snippet1.mp3",
                embedding=test_vector,
                begin_stamp_seconds=begin_stamp,
                end_stamp_seconds=end_stamp
            )
            AudioFeatures.objects.create(
                id=2,
                source_page_url="http://example.com/page1",
                asset_url="http://example.com/page1/snippet1.mp3",
                embedding=test_vector,
                begin_stamp_seconds=begin_stamp,
                end_stamp_seconds=end_stamp
            )

    def test_invalid_embedding_dimensions(self):

        begin_stamp = 0
        end_stamp = begin_stamp + settings.AUDIO_MODEL_CONFIG.get("INPUT_LEN_SECONDS", 20)

        invalid_vector_dimensions = 10

        invalid_vector = np.random.rand(invalid_vector_dimensions).tolist()

        with self.assertRaises(ValueError):
            AudioFeatures.objects.create(
                id=1,
                source_page_url="http://example.com/page1",
                asset_url="http://example.com/page1/snippet1.mp3",
                begin_stamp_seconds=begin_stamp,
                end_stamp_seconds=end_stamp,
                embedding=invalid_vector
            )


class TextFeatureExtractorTest(TestCase):
    
    def test_singleton(self):
        self.assertIs(text_extractor, TextFeatureExtractor())

    def test_get_and_store_output(self):
        
        example = "This is an Example Text Input"

        output = text_extractor.extract_features([example])

        TextFeatures.objects.create(
            id=1,
            source_page_url="http://example.com/page1",
            asset_url="http://example.com/page1#snippet1",
            content=example,
            embedding=output[0],
        )

        self.assertEqual(TextFeatures.objects.count(), 1)

        retrieved_features = TextFeatures.objects.first()

        self.assertEqual(retrieved_features.content, example)
        np.testing.assert_allclose(retrieved_features.embedding, output[0], rtol=1e-6)


class ImageFeatureExtractorTest(TestCase):
    
    def test_singleton(self):
        self.assertIs(image_extractor, ImageFeatureExtractor())

    def test_get_and_store_output(self):
        
        alt_text = "Two Cats Laying down"
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        output = image_extractor.extract_from_images([image])

        ImageFeatures.objects.create(
            id=1,
            source_page_url="http://images.cocodataset.org/val2017",
            asset_url="http://images.cocodataset.org/val2017/000000039769.jpg",
            alt_text=alt_text,
            embedding=output[0],
        )

        self.assertEqual(ImageFeatures.objects.count(), 1)

        retrieved_features = ImageFeatures.objects.first()

        self.assertEqual(retrieved_features.alt_text, alt_text)
        np.testing.assert_allclose(retrieved_features.embedding, output[0], rtol=1e-6)


class AudioFeatureExtractorTest(TestCase):
    
    def test_singleton(self):
        self.assertIs(audio_extractor, AudioFeatureExtractor())

    def test_get_and_store_output(self):

        url = "https://getsamplefiles.com/download/wav/sample-5.wav"

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
            response = requests.get(url)
            self.assertEqual(response.status_code, 200)
            tmp_file.write(response.content)
            tmp_file.flush()

            sampling_rate = settings.AUDIO_MODEL_CONFIG.get("SAMPLING_RATE", 48000)

            chuck_duration_s = settings.AUDIO_MODEL_CONFIG.get("INPUT_LEN_SECONDS", 20)

            chunk_count = 0

            for audio_chunk, start_s, end_s in chunk_audio(tmp_file.name, chuck_duration_s, sampling_rate):

                output = audio_extractor.extract_from_audios([audio_chunk])

                AudioFeatures.objects.create(
                    id=chunk_count+1,
                    begin_stamp_seconds=start_s,
                    end_stamp_seconds=end_s,
                    source_page_url="http://example.com/audio-page",
                    asset_url=url,
                    embedding=output[0]
                )

                chunk_count += 1

        self.assertEqual(AudioFeatures.objects.count(), chunk_count)

        first_chunk = AudioFeatures.objects.order_by("begin_stamp_seconds").first()
        self.assertIsNotNone(first_chunk)
        self.assertEqual(first_chunk.begin_stamp_seconds, 0)
        self.assertEqual(first_chunk.end_stamp_seconds, first_chunk.begin_stamp_seconds + chuck_duration_s)

        last_chunk = AudioFeatures.objects.order_by("begin_stamp_seconds").last()

        np.testing.assert_allclose(last_chunk.embedding, output[0], rtol=1e-6)


class AISearchViewTest(APITestCase):
    
    def setUp(self):

        self.url = reverse("multi_modal_search")

        image_vector_dimensions = settings.IMAGE_MODEL_CONFIG.get("DIMENSIONS", 512)

        audio_vector_dimensions = settings.AUDIO_MODEL_CONFIG.get("DIMENSIONS", 512)

        self.image_vector = np.random.rand(image_vector_dimensions).tolist()
        self.audio_vector = np.random.rand(audio_vector_dimensions).tolist()

        self.image_to_find = ImageFeatures.objects.create(
            id=1,
            source_page_url="http://example.com/page1",
            asset_url="http://example.com/image_to_find.jpg",
            alt_text="Cat in a box",
            embedding=self.image_vector
        )

        self.audio_to_find = AudioFeatures.objects.create(
            id=1,
            source_page_url="http://example.com/page1",
            asset_url="http://example.com/audio_to_find.wav",
            begin_stamp_seconds=0,
            end_stamp_seconds=20,
            embedding=self.audio_vector
        )

    def create_fake_image(self):
        image_buffer = io.BytesIO()

        img = Image.new("RGB", (256, 256), "white")

        img.save(image_buffer, "PNG")

        image_content = image_buffer.getvalue()
        
        return SimpleUploadedFile(
            name="image.png",
            content=image_content,
            content_type="image/png"
        )

    def create_fake_audio(self):

        audio_buffer = io.BytesIO()

        sample_rate = settings.AUDIO_MODEL_CONFIG.get("SAMPLE_RATE", 48000)

        wav = np.sin(np.linspace(0., 1., sample_rate * 20))

        wavfile.write(audio_buffer, rate=sample_rate, data=wav)

        audio_content = audio_buffer.getvalue()

        return SimpleUploadedFile(
            name="audio.wav",
            content=audio_content,
            content_type="audio/wav"
        )
        
    def create_corrupted_image(self):
        invalid_image_buffer = io.BytesIO(b"this is an invalid image")
        invalid_image_buffer.seek(0)
        return invalid_image_buffer

    def create_invalid_file_format(self):
        return SimpleUploadedFile("text.txt", b"this is an invalid file")
    
    @patch("search.ai_models.image_extractor.extract_from_images")
    def test_image_to_image_search_sucess(self, mock_extract_from_images):
        
        mock_extract_from_images.return_value = [self.image_vector]

        fake_image_file = self.create_fake_image()
        
        data = {
            "file":fake_image_file,
            "type":"image"
        }

        response = self.client.post(self.url, data, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        mock_extract_from_images.assert_called_once()

        self.assertEqual(len(response.data), 1)
        
        self.assertEqual(response.data[0]['asset_url'], self.image_to_find.asset_url)

    @patch("search.ai_models.audio_extractor.extract_from_audios")
    def test_audio_to_audio_search_sucess(self, mock_extract_from_audios):
        
        mock_extract_from_audios.return_value = [self.audio_vector]

        fake_audio_file = self.create_fake_audio()

        data = {
            "file":fake_audio_file,
            "type":"audio"
        }

        response = self.client.post(self.url, data=data, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        mock_extract_from_audios.assert_called_once()

        self.assertEqual(len(response.data), 1)
        
        self.assertEqual(response.data[0]['asset_url'], self.audio_to_find.asset_url)

    def test_search_with_corrupted_image(self):
        
        corrupted_image = self.create_corrupted_image()

        data = {
            "file": corrupted_image,
            "type": "image",
        }

        response = self.client.post(self.url, data=data, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        # self.assertIn("Invalid or Corrupted Image file", response.data['error'])

    def test_search_with_invalid_file_extension(self):
        
        invalid_file = self.create_invalid_file_format()

        data = {
            "file": invalid_file,
            "type": "audio"
        }

        response = self.client.post(self.url, data=data, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        # self.assertIsNotNone(response.data['error'])

    def test_search_missing_file_parameter(self):
        data = {
            "type": "audio"
        }

        response = self.client.post(self.url, data=data, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        # self.assertIsNotNone(response.data['error'])

    @patch("django.core.files.uploadedfile.SimpleUploadedFile")
    def test_search_with_file_too_large(self, mock_file):
        
        mock_file.size = 1024*1024*30

        data = {
            "file":mock_file,
            "type":"image"
        }

        serializer = FileSearchQuerySerializer(data=data)

        self.assertFalse(serializer.is_valid())


class KeywordSearch(APITestCase):
    
    def setUp(self):

        self.url = reverse("keyword_search")

        image_vector_dimensions = settings.IMAGE_MODEL_CONFIG.get("DIMENSIONS", 512)
        text_vector_dimensions = settings.TEXT_MODEL_CONFIG.get("DIMENSIONS", 384)

        self.image_vector = np.random.rand(image_vector_dimensions).tolist()
        self.text_vector = np.random.rand(text_vector_dimensions).tolist()

        self.image_to_find = ImageFeatures.objects.create(
            id=1,
            source_page_url="http://example.com/page1",
            asset_url="http://example.com/image_to_find.jpg",
            alt_text="Cat in a box",
            embedding=self.image_vector
        )

        self.text_to_find = TextFeatures.objects.create(
            id=1,
            source_page_url="http://example.com/page1",
            asset_url="http://example.com/image_to_find.jpg",
            content="lorem ipsum",
            embedding=self.text_vector
        )

    def test_text_to_image_success(self):
        
        data = {
            "q":"cat",
            "type":"image",
        }

        response = self.client.get(self.url, data)

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(len(response.data), 1)
        
        self.assertEqual(response.data[0]['asset_url'], self.image_to_find.asset_url)

    def test_text_to_text_success(self):
        data = {
            "q":"lorem",
            "type":"text",
        }

        response = self.client.get(self.url, data)

        self.assertEqual(response.status_code, status.HTTP_200_OK)

        self.assertEqual(len(response.data), 1)
        
        self.assertEqual(response.data[0]['content'], self.text_to_find.content)