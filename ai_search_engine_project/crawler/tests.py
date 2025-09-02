from unittest.mock import patch, Mock
import io
from io import StringIO

import requests
import numpy as np
from PIL import Image

from django.core.management import call_command
from django.test import TestCase

from search.models import TextFeatures, ImageFeatures


class CrawlerIndexerIntegrationTest(TestCase):
    def setUp(self):
        """Set up our fake mini-internet."""

        image_buffer = io.BytesIO()
        fake_image = Image.new('RGB', (1, 1), 'black')
        fake_image.save(image_buffer, format='PNG')
        image_bytes = image_buffer.getvalue()

        self.fake_site_content = {
            "http://example.com/testpage": """
                <html>
                    <body>
                        <h1>Main Title</h1>
                        <p>Some interesting text content.</p>
                        <img src="/images/photo.png" alt="A test photo">
                    </body>
                </html>
            """,
            "http://example.com/images/photo.png": image_bytes,
        }

    def mock_requests_get(self, url, timeout=None, stream=False):
        """
        A more advanced mock that can return HTML text or image bytes.
        """
        mock_response = Mock()
        if url in self.fake_site_content:
            content = self.fake_site_content[url]
            if isinstance(content, str):
                mock_response.text = content
                mock_response.content = content.encode("utf-8")
            else:
                mock_response.content = content

            mock_response.raise_for_status.return_value = None
            return mock_response
        else:
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "404 Not Found"
            )
            return mock_response

    @patch("requests.get")
    @patch("search.ai_models.text_extractor.extract_features")
    @patch("search.ai_models.image_extractor.extract_from_images")
    def test_crawler_successfully_triggers_indexer_tasks(
        self, mock_extract_images, mock_extract_text, mock_requests_get
    ):
        """
        Verifies that running the crawler command results in the correct
        objects being created in the database by the indexer tasks.
        """

        mock_requests_get.side_effect = self.mock_requests_get

        fake_text_vector = [0.1] * 384
        fake_image_vector = [0.2] * 512
        mock_extract_text.return_value = [fake_text_vector]
        mock_extract_images.return_value = [fake_image_vector]

        call_command(
            "run_crawler", "http://example.com/testpage", "--limit=1", "--delay=0"
        )

        self.assertEqual(TextFeatures.objects.count(), 2)
        first_text = TextFeatures.objects.get(content="Main Title")
        self.assertIsNotNone(first_text)
        np.testing.assert_allclose(first_text.embedding, fake_text_vector, rtol=1e-6)
        self.assertEqual(first_text.source_page_url, "http://example.com/testpage")

        self.assertEqual(ImageFeatures.objects.count(), 1)
        indexed_image = ImageFeatures.objects.first()
        self.assertIsNotNone(indexed_image)
        self.assertEqual(indexed_image.asset_url, "http://example.com/images/photo.png")
        self.assertEqual(indexed_image.alt_text, "A test photo")
        np.testing.assert_allclose(
            indexed_image.embedding, fake_image_vector, rtol=1e-6
        )

        self.assertEqual(mock_requests_get.call_count, 2)
        mock_extract_text.assert_called()
        mock_extract_images.assert_called_once()


class CrawlerCommandTest(TestCase):
    def setUp(self):
        """
        Create a fake mini-internet for our tests.
        This runs before each test method.
        """

        self.fake_site_content = {
            "http://example.com/page1": """
                <html>
                    <body>
                        <h1>Welcome</h1>
                        <p>This is some text content.</p>
                        <img src="/images/photo.jpg" alt="A nice photo">
                        <audio src="http://example.com/assets/sound.wav"></audio>
                        <a href="/page2">Go to Page 2</a>
                        <a href="http://external.com">External Link (ignore)</a>
                    </body>
                </html>
            """,
            "http://example.com/page2": """
                <html>
                    <body>
                        <p>This is the second page.</p>
                        <a href="http://example.com/page1">Go back to Page 1</a>
                    </body>
                </html>
            """,
        }

    def mock_requests_get(self, url, timeout=None, stream=None):
        """
        This function will replace requests.get.
        It returns a mock Response object based on our fake_site_content.
        """
        if url in self.fake_site_content:

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = self.fake_site_content[url]

            mock_response.raise_for_status.return_value = None
            return mock_response
        else:

            mock_response = Mock()
            mock_response.status_code = 404

            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "Not Found"
            )
            return mock_response

    @patch("indexer.tasks.index_text_snippet.delay")
    @patch("indexer.tasks.index_media_asset.delay")
    @patch("crawler.management.commands.run_crawler.requests.get")
    def test_crawler_happy_path(
        self, mock_crawler_get, mock_index_media_task, mock_index_text_task
    ):
        """
        Tests that the crawler successfully calls the celery tasks
        without actually running them.
        """

        mock_crawler_get.side_effect = self.mock_requests_get
        out = StringIO()

        call_command(
            "run_crawler",
            "http://example.com/page1",
            "--limit=2",
            "--delay=0",
            stdout=out,
        )

        self.assertEqual(mock_crawler_get.call_count, 2)

        self.assertEqual(mock_index_text_task.call_count, 6)

        mock_index_media_task.assert_any_call(
            {"url": "http://example.com/images/photo.jpg", "alt_text": "A nice photo"},
            "http://example.com/page1",
            "image",
        )
        mock_index_media_task.assert_any_call(
            {"url": "http://example.com/assets/sound.wav"},
            "http://example.com/page1",
            "audio",
        )
        self.assertEqual(mock_index_media_task.call_count, 2)

    @patch("requests.get")
    def test_crawler_handles_broken_link(self, mock_requests_get):
        """
        Tests that the crawler logs an error but doesn't crash if a page fails to load.
        """

        mock_requests_get.side_effect = self.mock_requests_get

        err = StringIO()

        call_command(
            "run_crawler", "http://example.com/broken_link", "--limit=1", stderr=err
        )

        self.assertIn("Could not fetch http://example.com/broken_link", err.getvalue())

    @patch("indexer.tasks.index_media_asset.delay")
    @patch("crawler.management.commands.run_crawler.requests.get")
    def test_crawler_respects_page_limit(self, mock_crawler_get, mock_index_media_task):
        """
        Tests that the crawler stops once it reaches the specified limit.
        """
        mock_crawler_get.side_effect = self.mock_requests_get

        call_command("run_crawler", "http://example.com/page1", "--limit=1")

        self.assertEqual(mock_crawler_get.call_count, 1)

        self.assertEqual(mock_index_media_task.call_count, 2)

    @patch("indexer.tasks.index_media_asset.delay")
    @patch("crawler.management.commands.run_crawler.requests.get")
    def test_crawler_avoids_loops(self, mock_crawler_get, mock_index_media_task):
        """
        Tests that the 'visited' set prevents the crawler from re-visiting pages.
        """
        mock_crawler_get.side_effect = self.mock_requests_get

        call_command("run_crawler", "http://example.com/page1", "--limit=5")

        self.assertEqual(mock_crawler_get.call_count, 2)

        self.assertEqual(mock_index_media_task.call_count, 2)
