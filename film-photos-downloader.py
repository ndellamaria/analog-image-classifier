import json
import requests
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

class AnalogImageDownloader:
    def __init__(self, output_dir='raw_images', page_size=200):
        self.base_url = 'https://api.analogdb.com'
        self.output_dir = output_dir
        self.page_size = page_size
        os.makedirs(output_dir, exist_ok=True)

    def fetch_posts(self):
        """
        Fetch posts from the API.
        """
        params = {
            'page_size': self.page_size
        }
            
        try:
            response = requests.get(
                f'{self.base_url}/posts',
                params=params,
                verify=False  # Disable SSL verification
            )
            return response.json()
        except Exception as e:
            print(f"Error fetching posts: {str(e)}")
            return {'posts': [], 'meta': {}}  # Return empty data instead of None

    def download_raw_image(self, post):
        """
        Download only the raw resolution image from a post.
        """
        try:
            # Get the raw image URL
            raw_image = next(img for img in post['images'] if img['resolution'] == 'raw')
            
            # Create filename from post ID
            filename = f"{post['id']}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(filepath):
                return
            
            # Download image
            response = requests.get(raw_image['url'], stream=True)
            response.raise_for_status()
            
            # Save image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        except Exception as e:
            print(f"Error downloading post {post['id']}: {str(e)}")

    def download_batch(self, posts):
        """
        Download a batch of posts in parallel.
        """
        with ThreadPoolExecutor(max_workers=5) as executor:
            list(tqdm(
                executor.map(self.download_raw_image, posts),
                total=len(posts),
                desc="Downloading raw images"
            ))

    def run(self, delay=1):
        """
        Run the downloader to fetch and download images.
        
        Args:
            delay: Delay between API requests in seconds
        """
        images_downloaded = 0
        
        while True:
            # Fetch posts
            data = self.fetch_posts()
            if not data:
                break
                
            # Download images from this batch
            self.download_batch(data['posts'])
            
            # Update counter
            images_downloaded += len(data['posts'])
            
            # Check if there are more pages
            if 'next_page_id' in data['meta']:
                next_page_id = data['meta']['next_page_id']
                print(f"Downloaded {images_downloaded} images. Moving to next page...")
                time.sleep(delay)  # Be nice to the API
            else:
                break

if __name__ == "__main__":
    # Create and run downloader
    downloader = AnalogImageDownloader(
        output_dir='raw_images',
        page_size=200  # Number of posts per API request
    )
    
    # Download images (optionally specify a limit)
    downloader.run(
        delay=1  # Seconds between API requests
    )