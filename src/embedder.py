import os
from tqdm import tqdm
import numpy as np
from cltk import NLP
from bertopic.backend import BaseEmbedder
from corpus import load_corpus
import multiprocessing as mp

os.environ["CLTK_DATA"] = "../data"

def compute_document_embeddings(docs, bar_queue) -> dict:
    """Creates a dict of document embeddings, using the average sentence embedding.
    :returns: A dict where the key is the document text and the value is the embedding vector
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
    all_doc_embeddings = dict()
    for index, text in enumerate(docs):
        # print(f"Task {task_num}: {index}/{len(docs)}")
            
        doc = nlp.analyze(text[1])
        all_doc_embeddings[text[1]] = average_sentence_embeddings(doc)
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
def _compute_embeddings_chunk(index_range, documents, all_embeddings, bar_queue):
    embeddings = compute_document_embeddings(documents[index_range.start:index_range.stop], bar_queue) 
    all_embeddings[position] = embeddings

    
def _update_bar(q, total_documents):
    pbar = tqdm(total=total_documents)
    while True:
        x = q.get()
        pbar.update(x)
    

def main():
    documents = load_corpus()
    
    # Create embeddings in parallel
    bar_queue = mp.Queue()
    all_embeddings = dict()
    
    num_threads = 8
    chunk_size = int(len(documents) / num_threads)
    processes = [ 
        mp.Process(target=_compute_embeddings_chunk, 
                   args=(range(chunk_size * i, chunk_size * (i+1)),
                         documents,
                         all_embeddings, 
                         bar_queue)
                  ) for i in range(num_threads) 
    ]
    bar_process = mp.Process(target=_update_bar, args=(bar_queue,len(documents)), daemon=True)
    bar_process.start()
    
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()
        
    embeddings_list = []
    for i in range(num_threads):
        embeddings_list.append(all_embeddings[i])
    np.save('doc_embeddings.npy', np.array(embeddings_list.values(), dtype=object), allow_pickle=True)

    
if __name__ == "__main__":
    main()