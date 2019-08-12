
import itertools
import wordcloud
import numpy as np
import matplotlib.pyplot as plt
import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from nltk.corpus import stopwords



def sentiment_check():
    print("*************ASSIGNMENT 1 - Sentiment Analysis using Logistic Regression*************\n")

    cell_phone_accessories = preprocessing.convert_to_DF('reviews_Cell_Phones_and_Accessories_5.json.gz')
    print('Dataset size: {:,} words'.format(len(cell_phone_accessories)))


    #Number of unique products

    products = cell_phone_accessories['overall'].groupby(cell_phone_accessories['asin']).count()
    print("Number of Unique Products in the Cell Phones and Accesories Category = {}".format(products.count()))


    #Top 20 Reviewed Products

    sorted_products = products.sort_values(ascending=False)

    print("\n\nTop 20 Reviewed Products:\n")
    print(sorted_products[:20], end='\n\n')
    print('Most Reviewed Product, has {} reviews.'.format(products.max()))


    #Bottom 20 Reviewed Products

    print("\n\nBottom 20 Reviewed Products:\n")
    print(sorted_products[-20:], end='\n\n')
    print('Least Reviewed Product (Sorted), has {} reviews.'.format(products.min()))


    reviews = cell_phone_accessories['reviewText']
    print("\n\nReviews Count")
    print(reviews.count())


    try:
        stops = stopwords.words('english')
    #     reviews = reviews.apply(lambda x: tokenize(x))
    except Exception as e:
        print(e)



    cloud = wordcloud.WordCloud(background_color='gray', max_font_size=60,
                                    relative_scaling=1).generate(' '.join(cell_phone_accessories.reviewText))

    # fig = plt.figure(figsize=(20, 10))
    # plt.axis('off')
    # plt.show(cloud)


    # Insert pos_neg column for Sentiment modeling
    # Negative reviews:      1-3 Stars  = 0
    # Positive reviews:      4-5 Stars  = 1

    cell_phone_accessories['pos_neg'] = [1 if x > 3 else 0 for x in cell_phone_accessories.overall]


    review_text = cell_phone_accessories["reviewText"]


    print("\n\n")

    #Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(cell_phone_accessories.reviewText, cell_phone_accessories.pos_neg, random_state=0)
    print("x_train shape: {}".format(x_train.shape), end='\n')
    print("y_train shape: {}".format(y_train.shape), end='\n\n')
    print("x_test shape: {}".format(x_test.shape), end='\n')
    print("y_test shape: {}".format(y_test.shape), end='\n\n')


    #Logistic Regression

    # Vectorize X_train
    vectorizer = CountVectorizer(min_df=5).fit(x_train)
    X_train = vectorizer.transform(x_train)
    print("X_train:\n{}".format(repr(X_train)))

    feature_names = vectorizer.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))


    # on Training data
    scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
    print("Mean cross-validation accuracy: {:.3f}".format(np.mean(scores)))

    logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
    X_test = vectorizer.transform(x_test)
    log_y_pred = logreg.predict(X_test)

    logreg_score = accuracy_score(y_test, log_y_pred)
    print("Accuracy:   {:.3f}".format(logreg_score))

    print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))


    log_cfm = confusion_matrix(y_test, log_y_pred)
    print("Confusion matrix:")
    print(log_cfm, end='\n\n')
    print('-'*15)
    print(np.array([['TN', 'FP'],[ 'FN' , 'TP']]))


    plt.imshow(log_cfm, interpolation='nearest')

    for i, j in itertools.product(range(log_cfm.shape[0]), range(log_cfm.shape[1])):
        plt.text(j, i, log_cfm[i, j],
                 horizontalalignment="center",
                 color="white")

    plt.ylabel('True label (Recall)')
    plt.xlabel('Predicted label (Precision)')
    plt.title('Logistic Reg | Confusion Matrix')
    plt.colorbar()

    log_f1 = f1_score(y_test, log_y_pred)
    print("Logistic Reg - F1 score: {:.3f}".format(log_f1))



    #To increase performance - GridSearchCV
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train, y_train)

    print("Best cross-validation score: {:.3f}".format(grid.best_score_))
    print("Best parameters: ", grid.best_params_)


    # Using Best Parameters on Testing data from GridSearchCV
    print("{:.3f}".format(grid.score(X_test, y_test)))
    grid_log_f1 = f1_score(y_test, log_y_pred)
    print("Grid Logistic Reg - F1 score: {:.3f}".format(grid_log_f1))
    cnf_matrix = confusion_matrix(y_test, log_y_pred)

    print(cnf_matrix)


if __name__ == '__main__':
    sentiment_check()















