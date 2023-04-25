from bertopic import BERTopic
from corpus import load_corpus
from embedder import load_document_embeddings, LatinEmbedder


def create_topic_model():
    documents = load_corpus()
    embeddings = load_document_embeddings(documents, './doc_embeddings_all.npy')
    
    print("Creating topic model...")
    topic_model = BERTopic(
        embedding_model=LatinEmbedder(document_embeddings=embeddings)
    )

    # Filter out any documents that didn't have a valid embedding
    documents = [doc[1] for doc in documents if not not embeddings[doc[1]].any()]

    topics, probs = topic_model.fit_transform(documents)
    return topics, probs


def main():
    topics, probs = create_topic_model()


if __name__ == "__main__":
    main()