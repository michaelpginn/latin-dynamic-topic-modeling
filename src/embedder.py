import os
from IPython.display import clear_output
from tqdm import tqdm
import numpy as np
from cltk import NLP
from bertopic.backend import BaseEmbedder
import multiprocessing as mp
import math
# from corpus import load_corpus


def average_sentence_embeddings(doc):
    all_sentences = []
    for sentence in doc.sentence_embeddings.values():
        if np.any(sentence):
            # Skip sentences that are just zeroes
            all_sentences.append(sentence)

    return np.average(np.array(all_sentences), axis=0)


def compute_document_embeddings(docs, bar_queue=None) -> list:
    """Creates a list of document embeddings, using the average sentence embedding.
    """

    # Create the NLP pipeline for Latin
    nlp = NLP(language='lat')
    nlp.pipeline.processes.pop(-1) # Removing ``LatinLexiconProcess``, it provides definitions which we don't need right now

    # Compute embeddings for every document
    all_doc_embeddings = []
    for index, text in enumerate(docs):
        if '.' not in text[1]:
            # This text isn't divided into sentences, we'll skip it
            all_doc_embeddings.append(np.zeros((300,)))
        else:
            doc = nlp.analyze(text[1])
            all_doc_embeddings.append(average_sentence_embeddings(doc))
        if bar_queue is not None:
            bar_queue.put_nowait(1)

    return all_doc_embeddings


def load_document_embeddings(documents, filename) -> dict:
    """Loads document embeddings from a file"""
    embeddings = np.load(filename, allow_pickle=True)

    embeddings_dict = dict()
    for i in range(len(documents)):
        embeddings_dict[documents[i][1]] = embeddings[i]

    return embeddings_dict


class LatinEmbedder(BaseEmbedder):
    def __init__(self, document_embeddings: dict):
        super().__init__()
        self.document_embeddings = document_embeddings
        self.nlp = NLP(language='lat')
        self.nlp.pipeline.processes.pop(-1)
    
    def embed(self, documents, verbose=False):
        all_embeddings = []
        for document in documents:
            if document in self.document_embeddings:
                """We've already computed our embeddings, so we don't need to do so again"""
                all_embeddings.append(self.document_embeddings[document])
            else:
                clear_output(wait=True)
                print(f"Embedding: {document}")
                all_embeddings.append(average_sentence_embeddings(self.nlp.analyze(document)))

        return np.array(all_embeddings)


# Helper funcs for multithreading
def _compute_embeddings_chunk(index_range, position, documents, all_embeddings, bar_queue):
    embeddings = compute_document_embeddings(documents[index_range.start:index_range.stop], bar_queue) 
    all_embeddings[position] = embeddings

    
def _update_bar(q, total_documents):
    pbar = tqdm(total=total_documents)
    while True:
        x = q.get()
        pbar.update(x)
    

from numpy import dot
from numpy.linalg import norm

nlp = NLP(language='lat')
nlp.pipeline.processes.pop(-1)
cos_sim = lambda a, b: dot(a, b)/(norm(a)*norm(b))

def topic_coherence(top_terms):
    embeddings = []
    for term in top_terms:
        word_embed = nlp.analyze(term).words[0].embedding
        if np.any(word_embed):
            embeddings.append(word_embed)
    
    total = 0
    num = 0
    for j in range(1,len(embeddings)):
        for i in range(0, j):
            num += 1
            total += cos_sim(embeddings[j], embeddings[i])
    
    return total / num

def mean_pairwise_jaccard(topics):
    total = 0
    num = 0
    for j in range(1, len(topics)):
        for i in range(0, j):
            num += 1
            intersect = len([v for v in topics[j] if v in topics[i]])
            union = len(set(topics[j] + topics[i]))
            total += intersect / union
    return total / num

def main():
    documents = load_corpus()
    
    # Create embeddings in parallel
    bar_queue = mp.Queue()
    all_embeddings = mp.Manager().dict()
    
    num_threads = 8
    chunk_size = math.ceil(len(documents) / num_threads)
    processes = []
    for i in range(num_threads):
        index_range = range(chunk_size * i, chunk_size * (i+1))
        print(index_range)
        process = mp.Process(target=_compute_embeddings_chunk,
                             args=(index_range, i, documents, all_embeddings, bar_queue))
        processes.append(process)

    bar_process = mp.Process(target=_update_bar, args=(bar_queue, len(documents)), daemon=True)
    bar_process.start()
    
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()

    bar_process.kill()

    # print(all_embeddings)
    embeddings_list = []
    for i in range(num_threads):
        embeddings_list += all_embeddings[i]
    np.save('doc_embeddings.npy', np.array(embeddings_list, dtype=object), allow_pickle=True)

    
if __name__ == "__main__":
    main()