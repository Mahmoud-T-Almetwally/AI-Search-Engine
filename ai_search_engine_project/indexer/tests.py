from unittest.mock import patch
import tempfile

import numpy as np
from PIL import Image
import requests
import scipy.io.wavfile
import celery

from django.core.management import call_command, CommandError
from django.test import TestCase
from django.conf import settings

from .tasks import index_text_snippet, index_media_asset

from search.models import ImageFeatures, TextFeatures, AudioFeatures


class IndexerTasksTest(TestCase):
    def setUp(self):
        self.fake_img_embed = np.random.rand(settings.IMAGE_MODEL_CONFIG.get('DIMENSIONS', 512))
        self.fake_audio_embed = np.random.rand(settings.AUDIO_MODEL_CONFIG.get('DIMENSIONS', 512))
        self.fake_text_embed = np.random.rand(settings.TEXT_MODEL_CONFIG.get('DIMENSIONS', 384))

    @patch("search.ai_models.text_extractor.extract_features")
    def test_index_text_snippet_task(self, mock_extract_features):

        mock_extract_features.return_value = [self.fake_text_embed]

        test_content = "This is a test snippet."
        source_url = "http://example.com/page1"

        result = index_text_snippet.delay(content=test_content, source_url=source_url)
        
        self.assertTrue(result.successful())
        
        mock_extract_features.assert_called_once_with([test_content])
        
        self.assertEqual(TextFeatures.objects.count(), 1)

        created_obj = TextFeatures.objects.first()
        self.assertEqual(created_obj.content, test_content)
        np.testing.assert_allclose(created_obj.embedding, self.fake_text_embed, rtol=1e-6)

    @patch("requests.get")
    @patch("search.ai_models.image_extractor.extract_from_images")
    def test_index_media_snippet_image_task_success(self, mock_extract_from_images, mock_requests_get):
        
        mock_extract_from_images.return_value = [self.fake_img_embed]

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
            img = Image.new('RGB', (1, 1))
            img.save(tmp_file, format='PNG')
            tmp_file.seek(0)
            
            mock_requests_get.return_value.raise_for_status.return_value = None
            mock_requests_get.return_value.content = tmp_file.read()

        media_info = {
            'url': 'http://example.com/image.png',
            'alt_text': 'A test image'
        }
        source_url = "http://example.com/page1"

        result = index_media_asset.delay(
            media_info=media_info,
            source_url=source_url,
            content_type='image'
        )

        self.assertTrue(result.successful())

        mock_requests_get.assert_called_once_with(media_info['url'], timeout=30, stream=True)
        
        mock_extract_from_images.assert_called_once()

        self.assertEqual(ImageFeatures.objects.count(), 1)

        created_obj = ImageFeatures.objects.first()
        self.assertEqual(created_obj.asset_url, media_info['url'])
        self.assertEqual(created_obj.alt_text, media_info['alt_text'])
        np.testing.assert_allclose(created_obj.embedding, self.fake_img_embed)

    @patch("requests.get")
    @patch("search.ai_models.image_extractor.extract_from_images")
    def test_index_media_snippet_task_network_fail(self, mock_extract_from_images, mock_requests_get):

        mock_requests_get.side_effect = requests.exceptions.HTTPError("404 Not Found")

        media_info = {
            'url': 'http://example.com/image.png',
            'alt_text': 'A test image'
        }
        source_url = "http://example.com/page1"

        with self.assertRaises(celery.exceptions.Retry):
            index_media_asset.delay(
                media_info=media_info,
                source_url=source_url,
                content_type='image'
            )

        mock_requests_get.assert_called_once()
        
        mock_extract_from_images.assert_not_called()
        
        self.assertEqual(ImageFeatures.objects.count(), 0)

    @patch("requests.get")
    @patch("search.ai_models.audio_extractor.extract_from_audios")
    def test_index_media_snippet_task_audio_success(self, mock_extract_from_audios, mock_requests_get):
        
        mock_extract_from_audios.return_value = [self.fake_audio_embed]

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_file:
            duration_s = settings.AUDIO_MODEL_CONFIG.get("INPUT_LEN_SECONDS", 20)
            sample_rate = settings.AUDIO_MODEL_CONFIG.get("SAMPLE_RATE", 48000)

            frequency = 440 
            t = np.linspace(0., duration_s, int(sample_rate * duration_s), endpoint=False)
            amplitude = np.iinfo(np.int16).max * 0.5 
            data = amplitude * np.sin(2. * np.pi * frequency * t)

            scipy.io.wavfile.write(filename=tmp_file.name, rate=sample_rate, data=data)
            tmp_file.seek(0)
            
            mock_requests_get.return_value.raise_for_status.return_value = None
            mock_requests_get.return_value.content = tmp_file.read()

        media_info = {
            'url': 'http://example.com/audio.wav',
        }

        source_url = "http://example.com/page1"

        result = index_media_asset.delay(
            media_info=media_info,
            source_url=source_url,
            content_type='audio'
        )

        self.assertTrue(result.successful())

        mock_requests_get.assert_called_once_with(media_info['url'], timeout=30, stream=True)
        
        mock_extract_from_audios.assert_called_once()

        self.assertEqual(AudioFeatures.objects.count(), 1)

        created_obj = AudioFeatures.objects.first()
        self.assertEqual(created_obj.asset_url, media_info['url'])
        np.testing.assert_allclose(created_obj.embedding, self.fake_audio_embed)

    @patch('requests.get')
    @patch('search.ai_models.image_extractor.extract_from_images')
    def test_index_media_asset_model_failure(self, mock_extract_images, mock_requests_get):
        """
        Tests that the task handles a failure within the AI model,
        and does not create a database entry.
        """

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
            img = Image.new('RGB', (1, 1))
            img.save(tmp_file, format='PNG')
            tmp_file.seek(0)
            
            mock_requests_get.return_value.raise_for_status.return_value = None
            mock_requests_get.return_value.content = tmp_file.read()


        mock_extract_images.side_effect = Exception("CUDA out of memory")

        media_info = {'url': 'http://example.com/good_url_bad_model.png', 'alt_text': 'An image'}
        source_url = "http://example.com/page1"

        with self.assertRaises(Exception) as cm:
            index_media_asset.delay(
                media_info=media_info,
                source_url=source_url,
                content_type='image'
            )

        self.assertIn("CUDA out of memory", str(cm.exception))

        mock_requests_get.assert_called_once()
        
        mock_extract_images.assert_called_once()

        self.assertEqual(ImageFeatures.objects.count(), 0)


class IndexerCommandTest(TestCase):
    def setUp(self):
        self.fake_img_embed = np.random.rand(settings.IMAGE_MODEL_CONFIG.get('DIMENSIONS', 512))
        self.fake_audio_embed = np.random.rand(settings.AUDIO_MODEL_CONFIG.get('DIMENSIONS', 512))
        self.fake_text_embed = np.random.rand(settings.TEXT_MODEL_CONFIG.get('DIMENSIONS', 384))

    @patch("search.ai_models.image_extractor.extract_from_images")
    def test_index_image_success(self, mock_extract_from_images):
        
        mock_extract_from_images.return_value = [self.fake_img_embed]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:

            img = Image.new("RGB", (1, 1))
            img.save(tmp_file, format="PNG")

            call_command(
                    'run_indexer',
                    '--type=image',
                    '--path', tmp_file.name, # Use the path of our temporary file
                    '--asset-url', 'http://example.com/image.png',
                    '--source-url', 'http://example.com/page1',
                    '--alt-text', 'A test image'
                )
        
        mock_extract_from_images.assert_called_once()

        self.assertEqual(ImageFeatures.objects.count(), 1)

        indexed_image = ImageFeatures.objects.first()
        self.assertEqual(indexed_image.asset_url, 'http://example.com/image.png')
        self.assertEqual(indexed_image.alt_text, 'A test image')
        np.testing.assert_allclose(indexed_image.embedding, self.fake_img_embed, rtol=1e-6)

    def test_index_image_corrupted(self):

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
            tmp_file.write(b"sample text. this is not an image")
            tmp_file.flush()

            with self.assertRaises(CommandError) as cm:
                call_command(
                    'run_indexer',
                    '--type=image',
                    '--path', tmp_file.name, 
                    '--asset-url', 'http://example.com/image.png',
                    '--source-url', 'http://example.com/page1',
                    '--alt-text', 'A test image'
                )
            
            self.assertIn("Invalid or corrupted image file", str(cm.exception))


        self.assertEqual(ImageFeatures.objects.count(), 0)

    @patch("search.ai_models.audio_extractor.extract_from_audios")
    def test_index_audio_success(self, mock_extract_from_audios):
        
        mock_extract_from_audios.return_value = [self.fake_audio_embed]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:

            duration_s = settings.AUDIO_MODEL_CONFIG.get("INPUT_LEN_SECONDS", 20)
            sample_rate = settings.AUDIO_MODEL_CONFIG.get("SAMPLE_RATE", 48000)

            frequency = 440 
            t = np.linspace(0., duration_s, int(sample_rate * duration_s), endpoint=False)
            amplitude = np.iinfo(np.int16).max * 0.5 
            data = amplitude * np.sin(2. * np.pi * frequency * t)

            scipy.io.wavfile.write(filename=tmp_file.name, rate=sample_rate, data=data)

            call_command(
                    'run_indexer',
                    '--type=audio',
                    '--path', tmp_file.name,
                    '--asset-url', 'http://example.com/audio.wav',
                    '--source-url', 'http://example.com/page1',
                )
        
        mock_extract_from_audios.assert_called_once()

        self.assertEqual(AudioFeatures.objects.count(), 1)

        indexed_audio = AudioFeatures.objects.first()
        self.assertEqual(indexed_audio.asset_url, 'http://example.com/audio.wav')
        np.testing.assert_allclose(indexed_audio.embedding, self.fake_audio_embed, rtol=1e-6)

    def test_index_audio_corrupted(self):

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_file:
            tmp_file.write(b"sample text. this is not an audio file")
            tmp_file.flush()
            
            with self.assertRaises(CommandError) as cm:
                call_command(
                        'run_indexer',
                        '--type=audio',
                        '--path', tmp_file.name,
                        '--asset-url', 'http://example.com/audio.wav',
                        '--source-url', 'http://example.com/page1',
                    )
                
            self.assertIn("Could not load audio file", str(cm.exception))
        

        self.assertEqual(AudioFeatures.objects.count(), 0)

    @patch("search.ai_models.text_extractor.extract_features")
    def test_index_text_success(self, mock_extract_features):
        
        mock_extract_features.return_value = [self.fake_text_embed]

        call_command(
            'run_indexer',
            '--type=text',
            '--content', 'Sample text',
            '--source-url', 'http://example.com/page1',
        )

        mock_extract_features.assert_called_once()

        self.assertEqual(TextFeatures.objects.count(), 1)

        indexed_text = TextFeatures.objects.first()
        self.assertEqual(indexed_text.source_page_url, 'http://example.com/page1')
        np.testing.assert_allclose(indexed_text.embedding, self.fake_text_embed, rtol=1e-6)