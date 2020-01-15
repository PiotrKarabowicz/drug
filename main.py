from flask import Flask, request, render_template
from pymed import PubMed
import pandas as pd
import sklearn
import pickle

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
    pubmed = PubMed(tool="MyTool", email="p.karabowicz@gmail.com")
    results1 = pubmed.query(text, max_results=50)

    lista_abstract_3=[]

    for i in results1:
        lista_abstract_3.append(i.abstract)

    df_abstract = pd.DataFrame(lista_abstract_3, columns = ['abstracts'])
    df_abstract['abstracts_lower'] = df_abstract['abstracts'].str.lower()

    df_abstract_1 = df_abstract.dropna()

    rnd = pickle.load(open('/static/finalized_model.sav', 'rb'))
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100)
    result1=count_vect.transform(df_abstract_1['abstracts_lower'])
    result2 = rnd.predict(result1)

    df_abstract_1['class'] = result2

#unsupervised learning
    from gensim.models import Word2Vec

    model_ted1 = Word2Vec.load("/static/word2vec.model")

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
    keys = ['craniosynostosis', 'receptor']
    title = "title"
    labels = keys
    a=0.7
    filename = '/static/output.png'


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
    return render_template('index22.html', lista_abstract=[df_abstract_1.to_html(classes='data')], text=text, titles=df_abstract_1.columns.values)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
