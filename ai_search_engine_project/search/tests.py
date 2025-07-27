import numpy as np
from PIL import Image
import tempfile
import requests

from django.test import TestCase
from django.db.utils import IntegrityError
from django.conf import settings

from .models import TextFeatures, ImageFeatures, AudioFeatures
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

            dimensions = settings.AUDIO_MODEL_CONFIG.get("DIMENSIONS", 512)

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


