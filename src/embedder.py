import os
from tqdm import tqdm
import numpy as np
from cltk import NLP
from bertopic.backend import BaseEmbedder
import multiprocessing as mp
import math
from corpus import load_corpus

os.environ["CLTK_DATA"] = "../data"

def compute_document_embeddings(docs, bar_queue) -> list:
    """Creates a list of document embeddings, using the average sentence embedding.
    """

    # Create the NLP pipeline for Latin
    nlp = NLP(language='lat')
    nlp.pipeline.processes.pop(-1) # Removing ``LatinLexiconProcess``, it provides definitions which we don't need right now
    # print("Created NLP pipeline.")

    # Run the NLP pipeline on all docs, then get average sentence embeddings for each doc
    def average_sentence_embeddings(doc):
        all_sentences = []
        for sentence in doc.sentence_embeddings.values():
            if np.any(sentence):
                # Skip sentences that are just zeroes
                all_sentences.append(sentence)

        return np.average(np.array(all_sentences), axis=0)
   
    # Compute embeddings for every document
    # print("Computing document embeddings...")
    all_doc_embeddings = []
    for index, text in enumerate(docs):
        # print(f"Task {task_num}: {index}/{len(docs)}")
            
        doc = nlp.analyze(text[1])
        all_doc_embeddings.append(average_sentence_embeddings(doc))
        bar_queue.put_nowait(1)


    # np.save('doc_embeddings.npy', np.array(all_doc_embeddings.values(), dtype=object), allow_pickle=True)
    return all_doc_embeddings


class LatinEmbedder(BaseEmbedder):
    def __init__(self, document_embeddings: dict):
        super().__init__()
        self.document_embeddings = document_embeddings
    
    def embed(self, documents, verbose=False):
        """We've already computed our embeddings, so we don't need to do so again"""
        return np.array([self.document_embeddings[document] for document in documents])


# Helper funcs for multithreading
def _compute_embeddings_chunk(index_range, position, documents, all_embeddings, bar_queue):
    embeddings = compute_document_embeddings(documents[index_range.start:index_range.stop], bar_queue) 
    all_embeddings[position] = embeddings

    
def _update_bar(q, total_documents):
    pbar = tqdm(total=total_documents)
    while True:
        x = q.get()
        pbar.update(x)
    

def main():
    documents = load_corpus()[:500]
    
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

    bar_process = mp.Process(target=_update_bar, args=(bar_queue,len(documents)), daemon=True)
    bar_process.start()
    
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()

    bar_process.kill()

    embeddings_list = []
    for i in range(num_threads):
        embeddings_list += all_embeddings[i]
    np.save('doc_embeddings.npy', np.array(embeddings_list, dtype=object), allow_pickle=True)

    
if __name__ == "__main__":
    main()