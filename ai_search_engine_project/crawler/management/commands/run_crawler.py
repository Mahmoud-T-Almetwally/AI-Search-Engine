import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from django.core.management.base import BaseCommand
from django.core.management import call_command

from indexer.tasks import index_media_asset, index_text_snippet


def print_page_content(page_url, texts: list[str], image_urls: list[str], audio_urls: list[str]):
    """
    This function will eventually trigger the real indexing process.
    For now, it just prints what it found.
    """
    print(f"\n--- Found Content on: {page_url} ---")
    if texts:
        print(f"  [+] Found {len(texts)} text snippets.")
    if image_urls:
        print(f"  [+] Found {len(image_urls)} images.")
    if audio_urls:
        print(f"  [+] Found {len(audio_urls)} audio files.")


class Command(BaseCommand):
    help = 'A simple web crawler to discover and index content.'

    def add_arguments(self, parser):
        parser.add_argument('seed_url', type=str, help='The initial URL to start crawling from.')
        parser.add_argument(
            '--limit',
            type=int,
            default=10,
            help='The maximum number of pages to crawl.'
        )
        parser.add_argument(
            '--delay',
            type=float,
            default=1.0,
            help='Seconds to wait between requests to be polite.'
        )

    def handle(self, *args, **options):
        seed_url = options['seed_url']
        page_limit = options['limit']
        crawl_delay = options['delay']
        
        frontier = deque([seed_url])
        
        visited = set()

        self.stdout.write(self.style.SUCCESS(f"Starting crawl from {seed_url} with a limit of {page_limit} pages."))

        while frontier and len(visited) < page_limit:
            
            current_url = frontier.popleft()

            if current_url in visited:
                continue

            visited.add(current_url)
            self.stdout.write(f"Crawling: {current_url} ({len(visited)}/{page_limit})")

            try:
                response = requests.get(current_url, timeout=5)
                response.raise_for_status() 
            except requests.RequestException as e:
                self.stderr.write(self.style.ERROR(f"Could not fetch {current_url}: {e}"))
                continue

            soup = BeautifulSoup(response.text, 'html.parser')

            texts = self.extract_text(soup)
            image_urls = self.extract_media_urls(soup, current_url, tag='img', extensions=['.jpg', '.jpeg', '.png'])
            audio_urls = self.extract_media_urls(soup, current_url, tag='audio', extensions=['.wav', '.mp3'])
            
            print_page_content(
                page_url=current_url,
                texts=texts,
                image_urls=image_urls,
                audio_urls=audio_urls
            )

            for text in texts:
                index_text_snippet.delay(text, current_url)
            
            for image_info in image_urls:
                index_media_asset.delay(image_info, current_url, 'image')
            
            for audio_info in audio_urls:
                index_media_asset.delay(audio_info, current_url, 'audio')
            
            self.stdout.write(f"Dispatched {len(texts)} text tasks and {len(image_urls) + len(audio_urls)} media tasks to the queue.")

            new_links = self.extract_links(soup, current_url)
            for link in new_links:
                if link not in visited:
                    frontier.append(link)
            
            time.sleep(crawl_delay)

        self.stdout.write(self.style.SUCCESS(f"Crawl finished. Visited {len(visited)} pages."))

    def extract_text(self, soup: BeautifulSoup) -> list[str]:
        """
        Extracts all meaningful, visible text snippets from a page,
        ignoring content from script, style, and head tags.
        """
        
        ignored_tags = {'style', 'script', 'head', 'title', 'meta', '[document]'}
        
        meaningful_texts = []
        for text_element in soup.body.find_all(string=True):
            if text_element.parent.name not in ignored_tags:
                stripped_text = text_element.strip()
                if stripped_text:
                    meaningful_texts.append(stripped_text)
                    
        return meaningful_texts

    def extract_media_urls(self, soup: BeautifulSoup, base_url: str, tag: str, extensions: list) -> list[dict]:
        """Extracts URLs for images or audio files."""
        media_list = []
        for element in soup.find_all(tag):
            src = element.get('src')
            if src and any(src.lower().endswith(ext) for ext in extensions):
                 
                absolute_url = urljoin(base_url, src)
                media_info = {'url': absolute_url}
                if tag == 'img':
                    media_info['alt_text'] = element.get('alt', '')
                media_list.append(media_info)
        return media_list

    def extract_links(self, soup: BeautifulSoup, base_url: str) -> set[str]:
        """Finds all valid, crawlable links on a page."""
        links = set()
        base_netloc = urlparse(base_url).netloc 
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
             
            absolute_link = urljoin(base_url, href)
            
            parsed_link = urlparse(absolute_link)
            if parsed_link.scheme in ['http', 'https'] and parsed_link.netloc == base_netloc:
                links.add(absolute_link)
        return links