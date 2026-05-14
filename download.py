import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelGBIFDownloader:
    def __init__(self, max_workers=10):
        self.base_url = "https://api.gbif.org/v1"
        self.max_workers = max_workers
        self.session = requests.Session()
        
    def get_taxon_key(self, species_name):
        url = f"{self.base_url}/species/match"
        params = {'name': species_name}
        try:
            resp = self.session.get(url, params=params, timeout=10).json()
            return resp.get('usageKey'), species_name
        except Exception:
            return None, species_name

    def download_single_image(self, img_url, folder, filename):
        try:
            # Note: We use a simple get here as external image hosts 
            # don't share GBIF's OAuth token
            resp = requests.get(img_url, timeout=15)
            if resp.status_code == 200:
                with open(os.path.join(folder, filename), 'wb') as f:
                    f.write(resp.content)
                return True
        except Exception:
            pass
        return False

    def process_species(self, species_name, offset,total_needed=1000):
        taxon_key, name = self.get_taxon_key(species_name)
        if not taxon_key:
            print(f"Skipping: {species_name} (Not found)")
            return

        folder = f"downloads/{name.replace(' ', '_')}"
        os.makedirs(folder, exist_ok=True)
        
        print(f"Starting downloads for {name} (Target: {total_needed})...")

        images_processed = 0
        limit_per_page = 300  # GBIF max is 300
        
        # Outer loop for pagination
        while images_processed < total_needed:
            search_url = f"{self.base_url}/occurrence/search"
            params = {
                'taxonKey': taxon_key, 
                'mediaType': 'StillImage', 
                'limit': min(limit_per_page, total_needed - images_processed),
                'offset': offset+images_processed
            }
            
            response = self.session.get(search_url, params=params).json()
            results = response.get('results', [])
            
            if not results:
                break  # Exit if no more images are available

            # Nested parallel execution for the current page
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for i, rec in enumerate(results):
                    media = rec.get('media', [])
                    if media:
                        img_url = media[0].get('identifier')
                        # Sanitize extension
                        ext = img_url.split('.')[-1].split('?')[0] or 'jpg'
                        if len(ext) > 4: ext = 'jpg'
                        
                        fname = f"img_{images_processed + i+offset}.{ext}"
                        futures.append(executor.submit(self.download_single_image, img_url, folder, fname))
                
                success_count = sum(1 for f in as_completed(futures) if f.result())
                images_processed += len(results)
                print(f"Progress for {name}: {images_processed} records checked...")

        print(f"Finished {name}: Total records processed: {images_processed}")

    def run(self, species_list, offset, images_per_species=600):
        # Top-level parallel execution for species
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Using partial or a helper to pass the target count
            executor.map(lambda s: self.process_species(s, offset,images_per_species), species_list)

if __name__ == "__main__":
    # Example species list
    my_species = ["Daucus carota L.","Conium maculatum L."]
    
    bot = ParallelGBIFDownloader(max_workers=4)
    bot.run(my_species, offset=300, images_per_species=10000)