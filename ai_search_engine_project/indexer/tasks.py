from celery import shared_task
import requests
import tempfile
import os
from django.core.management import call_command

# The @shared_task decorator turns this function into a Celery task.
@shared_task(bind=True, autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def index_media_asset(self, media_info, source_url, content_type):
    """
    A robust Celery task to download, process, and index a media asset.
    `bind=True`: gives us access to `self` for retrying.
    `autoretry_for`: automatically retry on any exception.
    `retry_kwargs`: wait 60 seconds between retries, try a max of 3 times.
    """
    asset_url = media_info['url']
    print(f"[Celery Worker] Starting task for: {asset_url}")

    # The download and call_command logic is identical to what we planned before
    response = requests.get(asset_url, timeout=30, stream=True)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(asset_url)[1]) as tmp_file:
        tmp_file.write(response.content)
        tmp_file.flush()
        
        args = [
            '--type', content_type, '--path', tmp_file.name,
            '--asset-url', asset_url, '--source-url', source_url
        ]
        if content_type == 'image':
            args.extend(['--alt-text', media_info.get('alt_text', '')])
        
        call_command('run_indexer', *args)
    
    print(f"[Celery Worker] Task success for: {asset_url}")
    return f"Successfully indexed {asset_url}"

@shared_task
def index_text_snippet(content, source_url):
    """A Celery task to index a text snippet."""
    print(f"[Celery Worker] Indexing text snippet from {source_url}")
    call_command('run_indexer', '--type=text', '--content', content, '--source-url', source_url)
    return f"Successfully indexed text from {source_url}"