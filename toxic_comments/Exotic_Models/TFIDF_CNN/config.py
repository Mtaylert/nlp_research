from sklearn.feature_extraction.text import TfidfVectorizer
BATCH_SIZE = 32
TFIDF = TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))
HIDDEN1 = 512
HIDDEN2 = 128
LABELS = 1
LEARNING_RATE = 4e-2