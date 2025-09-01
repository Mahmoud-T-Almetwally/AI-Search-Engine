# indexer/management/commands/run_indexer.py

import logging
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from PIL import Image
import librosa
import numpy as np

from search.ai_models import text_extractor, image_extractor, audio_extractor
from search.models import TextFeatures, ImageFeatures, AudioFeatures

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Processes and indexes a single piece of content (text, image, or audio).'

    def add_arguments(self, parser):
        parser.add_argument('--type', type=str, required=True, choices=['text', 'image', 'audio'], help='The type of content to index.')
        parser.add_argument('--source-url', type=str, required=True, help='The source page URL where the content was found.')

        parser.add_argument('--content', type=str, help='The text content to index (for type=text).')
        parser.add_argument('--path', type=str, help='The local file path to the asset (for type=image or type=audio).')
        parser.add_argument('--asset-url', type=str, help='The original public URL of the asset (for type=image or type=audio).')
        parser.add_argument('--alt-text', type=str, default='', help='The alt text for an image (for type=image).')

    def handle(self, *args, **options):
        content_type = options['type']
        source_url = options['source_url']
        
        self.stdout.write(f"Starting indexing for type: {content_type}")

        try:
            if content_type == 'text':
                if not options['content']:
                    raise CommandError("Argument --content is required for type=text.")
                self._index_text(options['content'], source_url)
            
            elif content_type == 'image':
                if not options['path'] or not options['asset_url']:
                    raise CommandError("Arguments --path and --asset-url are required for type=image.")
                self._index_image(options['path'], options['asset_url'], source_url, options['alt_text'])

            elif content_type == 'audio':
                if not options['path'] or not options['asset_url']:
                    raise CommandError("Arguments --path and --asset-url are required for type=audio.")
                self._index_audio(options['path'], options['asset_url'], source_url)

        except Exception as e:
            raise CommandError(f"An error occurred during indexing: {e}")

        self.stdout.write(self.style.SUCCESS("Indexing completed successfully."))


    def _index_text(self, content: str, source_url: str):
        """Processes and saves a text snippet."""
        self.stdout.write("Generating text embedding...")
        embedding = text_extractor.extract_features([content])[0]

        TextFeatures.objects.create(
            source_page_url=source_url,
            content=content,
            embedding=embedding
        )
        self.stdout.write("Saved TextFeatures to database.")

    def _index_image(self, file_path: str, asset_url: str, source_url: str, alt_text: str):
        """Processes and saves an image file."""
        self.stdout.write(f"Opening image: {file_path}")
        try:
            img = Image.open(file_path)
            img.verify() 
            img = Image.open(file_path)
        except Exception as e:
            raise IOError(f"Invalid or corrupted image file: {file_path}. Error: {e}")

        self.stdout.write("Generating image embedding...")
        embedding = image_extractor.extract_from_images([img])[0]
        
        ImageFeatures.objects.update_or_create(
            asset_url=asset_url,
            defaults={
                'source_page_url': source_url,
                'alt_text': alt_text,
                'embedding': embedding
            }
        )
        self.stdout.write(f"Saved/Updated ImageFeatures for {asset_url}.")

    def _index_audio(self, file_path: str, asset_url: str, source_url: str):
        """Processes, chunks, transcribes, and saves an audio file."""
        self.stdout.write(f"Processing audio: {file_path}")
        
        target_sr = settings.AUDIO_MODEL_CONFIG.get('SAMPLING_RATE', 48000)
        chunk_duration_s = settings.AUDIO_MODEL_CONFIG.get('INPUT_LEN_SECONDS', 20)
        
        try:
            waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
        except Exception as e:
            raise IOError(f"Could not load audio file: {file_path}. Error: {e}")

        chunk_length_samples = chunk_duration_s * sr
        num_chunks = int(np.ceil(len(waveform) / chunk_length_samples))

        for i in range(num_chunks):
            start_sample = i * chunk_length_samples
            end_sample = start_sample + chunk_length_samples
            chunk_waveform = waveform[start_sample:end_sample]

            if not chunk_waveform.any(): continue

            start_time_s = i * chunk_duration_s
            end_time_s = start_time_s + chunk_duration_s
            self.stdout.write(f"  - Processing chunk: {start_time_s}s - {end_time_s}s")

            self.stdout.write("    - Generating audio embedding...")
            embedding = audio_extractor.extract_from_audios([chunk_waveform])[0]

            AudioFeatures.objects.update_or_create(
                asset_url=asset_url,
                begin_stamp_seconds=start_time_s,
                defaults={
                    'source_page_url': source_url,
                    'end_stamp_seconds': end_time_s,
                    'embedding': embedding
                }
            )
        self.stdout.write(f"Saved/Updated {num_chunks} AudioFeatures chunks for {asset_url}.")