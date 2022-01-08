vec = TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5), max_features = 46000)
vec.fit(df['text'])

from gensim.models import KeyedVectors, FastText
from scipy.sparse import hstack
fmodel = FastText.load('../input/jigsaw-regression-based-data/FastText-jigsaw-256D/Jigsaw-Fasttext-Word-Embeddings-256D.bin')

from scipy.sparse import hstack


def splitter(text):
    tokens = []

    for word in text.split(' '):
        tokens.append(word)

    return tokens


def vectorizer(text):
    tokens = splitter(text)

    x1 = vec.transform([text]).toarray()
    x2 = np.mean(fmodel.wv[tokens], axis=0).reshape(1, -1)
    x = np.concatenate([x1, x2], axis=-1).astype(np.float16)
    del x1
    del x2

    return x


X_list = []

for text in df.text:
    X_list.append(vectorizer(text))

EMB_DIM = len(vec.vocabulary_) + 256

X_np = np.array(X_list).reshape(-1, EMB_DIM)

from scipy import sparse

X = sparse.csr_matrix(X_np)
del X_np