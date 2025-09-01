from unittest.mock import patch
from io import StringIO
import tempfile

import numpy as np
from PIL import Image
import scipy.io.wavfile

from django.core.management import call_command, CommandError
from django.test import TestCase
from django.conf import settings

from search.models import ImageFeatures, TextFeatures, AudioFeatures


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
            tmp_file.write(b"sample text. this is not an audio")
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