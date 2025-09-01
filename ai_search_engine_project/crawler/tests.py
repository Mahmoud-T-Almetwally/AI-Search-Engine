from unittest.mock import patch, Mock
from io import StringIO

import requests

from django.core.management import call_command
from django.test import TestCase

from crawler.management.commands.run_crawler import Command

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
            """
        }

    def mock_requests_get(self, url, timeout=None):
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
            
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Not Found")
            return mock_response
        
    @patch('crawler.management.commands.run_crawler.process_page_content')
    @patch('requests.get')
    def test_crawler_happy_path(self, mock_requests_get, mock_process_page):
        """
        Tests that the crawler successfully crawls a site, extracts content,
        finds links, and calls the indexer hand-off function correctly.
        """

        mock_requests_get.side_effect = self.mock_requests_get

        out = StringIO()

        call_command(
            'run_crawler',
            'http://example.com/page1',
            '--limit=2',
            '--delay=0',
            stdout=out
        )

        mock_process_page.assert_any_call(
            page_url='http://example.com/page1',
            texts=['Welcome', 'This is some text content.', 'Go to Page 2', 'External Link (ignore)'],
            image_urls=[{'url': 'http://example.com/images/photo.jpg', 'alt_text': 'A nice photo'}],
            audio_urls=[{'url': 'http://example.com/assets/sound.wav'}]
        )

        mock_process_page.assert_any_call(
            page_url='http://example.com/page2',
            texts=['This is the second page.', 'Go back to Page 1'],
            image_urls=[],
            audio_urls=[]
        )

        self.assertEqual(mock_process_page.call_count, 2)
        
        self.assertEqual(mock_requests_get.call_count, 2)
    
    @patch('requests.get')
    def test_crawler_handles_broken_link(self, mock_requests_get):
        """
        Tests that the crawler logs an error but doesn't crash if a page fails to load.
        """
        
        mock_requests_get.side_effect = self.mock_requests_get
        
        err = StringIO()

        call_command('run_crawler', 'http://example.com/broken_link', '--limit=1', stderr=err)
        
        self.assertIn("Could not fetch http://example.com/broken_link", err.getvalue())

    @patch('requests.get')
    def test_crawler_respects_page_limit(self, mock_requests_get):
        """
        Tests that the crawler stops once it reaches the specified limit.
        """
        mock_requests_get.side_effect = self.mock_requests_get
        
        call_command('run_crawler', 'http://example.com/page1', '--limit=1')
        
        self.assertEqual(mock_requests_get.call_count, 1)
    
    @patch('requests.get')
    def test_crawler_avoids_loops(self, mock_requests_get):
        """
        Tests that the 'visited' set prevents the crawler from re-visiting pages.
        """
        mock_requests_get.side_effect = self.mock_requests_get
        
        call_command('run_crawler', 'http://example.com/page1', '--limit=5')
        
        self.assertEqual(mock_requests_get.call_count, 2)