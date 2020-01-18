from flask import Flask, request, render_template
from pymed import PubMed
import pandas as pd
import sklearn
import pickle
import numpy as np

import nltk


app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template('url.html')

@app.route('/page1')
def page1():
    return render_template('req1.html')

@app.route('/page1', methods=['POST'])
def my_form_post():
    text = request.form['text']
    #count_vect.fit(text)
    text = [text]

    import pandas as pd

    pubmed = PubMed(tool="MyTool", email="p.karabowicz@gmail.com")
    results_pred = pubmed.query(text, max_results=50)

    from keras.models import load_model
    model1 = load_model('/home/piotr/projekt/drug/static/model3.h5')
    #from tensorflow import keras
    #model = keras.models.load_model('/static/model2.h5')


    lista_abstract_pred=[]

    for i in results_pred:
        lista_abstract_pred.append(i.abstract)

    import numpy as np
    df_base_pred = pd.DataFrame(lista_abstract_pred, columns = ['abstracts'])

    df_base_pred['abstracts_lower'] = df_base_pred['abstracts'].str.lower()
    df_base_pred1 = df_base_pred.dropna()
    import string
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords

    review_lines = list()

    lines = df_base_pred1['abstracts_lower'].values.tolist()

    for line in lines:
        tokens = word_tokenize(line)
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha() ]
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        review_lines.append(words)

    len(review_lines)

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical

    MAX_SEQUENCE_LENGTH = 200

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(review_lines)
    sequences = tokenizer.texts_to_sequences(review_lines)

    word_index = tokenizer.word_index
    #print('Found %s unique tokens.' % len(word_index))

    review_pad = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    #df_base_pred1['class1'] = 0
    #sentiment =  df_base_pred1['class1'].values
    #print(sentiment)
    #print('Shape of data tensor:', review_pad.shape)
    #print('Shape of label tensor:', sentiment.shape)

    indices = np.arange(review_pad.shape[0])
    np.random.shuffle(indices)
    review_pad = review_pad[indices]
    #sentiment = sentiment[indices]
    x_test = review_pad[:]

    ynew1 = model1.predict_classes(x_test)
    df_base_pred1['class'] = ynew1

    df = df_base_pred1[['abstracts','class']]
    df_good = df[df['class'] ==1]
    #df_good = df_good1['abstracts']
    len_df= len(df_good)

#unsupervised learning
    from gensim.models import Word2Vec

    model_ted1 = Word2Vec.load("/home/piotr/projekt/drug/static/word2vec.model")

    embedding_clusters = []
    word_clusters = []

    embeddings = []
    words = []
    for similar_word, _ in model_ted1.wv.most_similar(positive = ['head', 'gene'], topn=30):
        words.append(similar_word)
        embeddings.append(model_ted1[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)

    from sklearn.manifold import TSNE
    import numpy as np

    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

    import matplotlib.pyplot as plt

    import matplotlib.cm as cm
    keys = ['gene', 'protein']
    title = "title"
    labels = keys
    a=0.7
    filename = '/home/piotr/projekt/drug/static/output.png'


    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    #plt.show()



    #result3 = str(result2)
    return render_template('index22.html', len_df=len_df, lista_abstract=[df_good.to_html(classes='data')], text=text, titles=df_good.columns.values)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False)
