from bertopic import BERTopic
from corpus import load_corpus
from embedder import compute_document_embeddings, LatinEmbedder


def create_topic_model():
    documents = load_corpus()
    embeddings = compute_document_embeddings(documents)
    
    print("Creating topic model...")
    topic_model = BERTopic(
        embedding_model=LatinEmbedder(document_embeddings=embeddings)
    )

    topics, probs = topic_model.fit_transform([doc[1] for doc in documents])
    return topics, probs


def main():
    topics, probs = create_topic_model()


if __name__ == "__main__":
    main()