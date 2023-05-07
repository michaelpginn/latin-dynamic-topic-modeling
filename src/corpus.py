import os
import glob
from cltk.data.fetch import FetchCorpus
from cltk.lemmatize.lat import LatinBackoffLemmatizer
from cltk.tokenizers.lat.lat import LatinWordTokenizer


def download_corpus():
    """Downloads the Latin Library corpus to your disk"""
    corpus_downloader = FetchCorpus(language='lat')
    corpus_downloader.import_corpus('lat_text_latin_library')
    

def load_corpus():
    """Loads the entire corpus from the disk"""
    print("Loading corpus...")
    all_texts = []

    for filename in glob.glob('./data/lat/text/lat_text_latin_library/**/*.txt', recursive=True):
        text = open(filename, 'r').read()
        all_texts.append((filename, text))

    print(f"Loaded {len(all_texts)} texts.")
    return all_texts