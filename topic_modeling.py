import numpy as np
import preprocessing
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation



def retrieve_top_words(model, feature_names, num_top_words):
    for idx, topic in enumerate(model.components_):
        print("Topic #{}:".format(idx), end='\n')
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]), end='\n\n')
    print()


def NMF_tf_idf(tfidf, tfidf_feature_names, num_top_words):
    # Initialize NMF
    nmf = NMF(n_components=10, random_state=1,
              alpha=.1, l1_ratio=.5)

    nmf_tfidf = nmf.fit(tfidf)
    nmf_W = nmf_tfidf.transform(tfidf)
    Counter([np.argmax(i) for i in nmf_W])
    print(retrieve_top_words(nmf_tfidf, tfidf_feature_names, num_top_words))


def LDA_tf_idf(tfidf, tfidf_feature_names, num_top_words):
    # Initialize Ida
    lda = LatentDirichletAllocation(n_topics=10, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)

    lda_tfidf = lda.fit(tfidf)

    lda_W = lda_tfidf.transform(tfidf)
    Counter([np.argmax(i) for i in lda_W])

    print(retrieve_top_words(lda_tfidf, tfidf_feature_names, num_top_words))


def topic_extraction():

    print("*************ASSIGNMENT 2 - Topic Modelling using NMF & LDA*************")

    cell_phone_accessories = preprocessing.convert_to_DF('reviews_Cell_Phones_and_Accessories_5.json.gz')
    stops = stopwords.words('english')
    review_text = cell_phone_accessories["reviewText"]
    num_top_words = 15

    # Use tf-idf features
    tfidf_vectorizer = TfidfVectorizer(stop_words=stops)
    tfidf = tfidf_vectorizer.fit_transform(review_text)

    # Use tf features
    tf_vectorizer = CountVectorizer(stop_words=stops)
    tf = tf_vectorizer.fit_transform(review_text)

    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print("\nNumber of total features: {}\n".format(len(tfidf_feature_names)))

    #NMF TF IDF
    print("Using NMF - TF IDF\n")
    NMF_tf_idf(tfidf, tfidf_feature_names, num_top_words)

    print("Using LDA - TF IDF\n")
    #LDA TF IDF
    LDA_tf_idf(tfidf, tfidf_feature_names, num_top_words)



if __name__ == '__main__':
    topic_extraction()







