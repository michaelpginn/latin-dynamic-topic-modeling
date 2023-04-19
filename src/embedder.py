from tqdm.notebook import tqdm
import numpy as np
from cltk import NLP
from bertopic.backend import BaseEmbedder


def compute_document_embeddings(docs, task_num, lock) -> dict:
    """Creates a dict of document embeddings, using the average sentence embedding.
    :returns: A dict where the key is the document text and the value is the embedding vector
    """

    with lock:
        # Create the NLP pipeline for Latin
        nlp = NLP(language='lat')
        nlp.pipeline.processes.pop(-1) # Removing ``LatinLexiconProcess``, it provides definitions which we don't need right now
        print("Created NLP pipeline.")

    # Run the NLP pipeline on all docs, then get average sentence embeddings for each doc
    def average_sentence_embeddings(doc):
        all_sentences = []
        for sentence in doc.sentence_embeddings.values():
            if np.any(sentence):
                # Skip sentences that are just zeroes
                all_sentences.append(sentence)

        return np.average(np.array(all_sentences), axis=0)
   
    # Compute embeddings for every document
    with lock:
        print("Computing document embeddings...")
    all_doc_embeddings = dict()
    for index, text in enumerate(docs):
        with lock:
            print(f"Task {task_num}: {index}/{len(docs)}")
            
        doc = nlp.analyze(text[1])
        all_doc_embeddings[text[1]] = average_sentence_embeddings(doc)


    # np.save('doc_embeddings.npy', np.array(all_doc_embeddings.values(), dtype=object), allow_pickle=True)
    return all_doc_embeddings


class LatinEmbedder(BaseEmbedder):
    def __init__(self, document_embeddings: dict):
        super().__init__()
        self.document_embeddings = document_embeddings
    
    def embed(self, documents, verbose=False):
        """We've already computed our embeddings, so we don't need to do so again"""
        return np.array([self.document_embeddings[document] for document in documents])