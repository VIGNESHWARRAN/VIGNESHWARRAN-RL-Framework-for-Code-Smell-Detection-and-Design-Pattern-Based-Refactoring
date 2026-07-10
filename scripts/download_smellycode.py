"""
Downloads SmellyCode++ from Figshare (DOI: 10.6084/m9.figshare.28519385.v1).
Raises RuntimeError if download fails or file is corrupt (<10MB).
"""
import os
import urllib.request
import logging
import sys

# Configure standard logger to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
log = logging.getLogger("download")

URL = "https://figshare.com/ndownloader/files/52744561"
OUT = "data/smellycode/SmellyCode++.csv"
MIN_BYTES = 10 * 1024 * 1024  # 10 MB sanity check

def main():
    os.makedirs("data/smellycode", exist_ok=True)
    log.info(f"Downloading SmellyCode++ from {URL} to {OUT}...")
    try:
        urllib.request.urlretrieve(URL, OUT)
    except Exception as e:
        log.error(f"Download failed: {e}")
        sys.exit(1)
        
    size = os.path.getsize(OUT)
    log.info(f"Downloaded file size: {size / 1e6:.2f} MB")
    if size < MIN_BYTES:
        raise RuntimeError(f"Downloaded file is too small ({size} bytes). Expected >= {MIN_BYTES}. Check URL or connectivity.")
    log.info("Download complete. Run: python main.py --stage preprocess")

if __name__ == "__main__":
    main()
