# Topic Modeling and Perplexity Analysis
import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Read data
df = pd.read_excel('DataForPy.xlsx', sheet_name='DataForPhy')
df = df[['Content']].dropna()

# Data cleaning
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Manual Stopword-List
basic_stopwords = {
    'the', 'and', 'is', 'in', 'it', 'of', 'to', 'a', 'for', 'on', 'that',
    'with', 'as', 'this', 'was', 'at', 'by', 'an', 'be', 'are', 'from',
    'or', 'has', 'have', 'but', 'not', 'they', 'their', 'which', 'will',
    'can', 'its', 'about', 'more', 'one', 'we', 'also', 'all', 'who',
    'you', 'been', 'he', 'she', 'his', 'her', 'them', 'so', 'if', 'may', 'BBC',
    'data', 'news', 'article', 'report', 'said', 'say', 'like', 'just',
    'there', 'when', 'what', 'where', 'why', 'how', 'do', 'does', 'did',
    'such', 'than', 'then', 'now', 'out', 'up', 'down',
    'over', 'under', 'after', 'before', 'between', 'while',
    'during', 'since', 'until', 'within', 'without', 'through', 'across',
    'around', 'along', 'against', 'among', 'beyond', 'despite', 'except',
    'inside', 'outside', 'toward', 'towards', 'upon',
    'because', 'although', 'though', 'even', 'unless',
    'whereas','new', 'would', 'year','bbc','mr','uk'
}

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I | re.A)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in basic_stopwords]
    return ' '.join(tokens)

df['Cleaned_Content'] = df['Content'].apply(clean_text)

# Vectorization and Topic Modeling
vectorizer = CountVectorizer(max_df=0.9, min_df=2)
doc_term_matrix = vectorizer.fit_transform(df['Cleaned_Content'])

# Test different numbers of topics
perplexities = []
k_values = list(range(2, 10))

for k in k_values:
    lda_model = LatentDirichletAllocation(n_components=k, random_state=42)
    lda_model.fit(doc_term_matrix)
    perplexity = lda_model.perplexity(doc_term_matrix)
    perplexities.append(perplexity)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(k_values, perplexities, marker='o')
plt.title('Perplexity vs. Count of Topics')
plt.xlabel('Count of Topics')
plt.ylabel('Perplexity')
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()
