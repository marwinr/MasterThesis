# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Load data
df = pd.read_excel('DataForPy.xlsx', sheet_name='DataForPhy')
df = df[['Source', 'Author', 'Headline', 'Year', 'Content', 'Length']]

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

# Topic Modeling
vectorizer = CountVectorizer(max_df=0.9, min_df=2)
doc_term_matrix = vectorizer.fit_transform(df['Cleaned_Content'])

lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(doc_term_matrix)

# Manual Topic Labels
topic_labels = {
    0: 'Power and Environment',
    1: 'Social and Regulations',
    2: 'Economy and Technology',
    3: 'Technology and Innovation'
}

# Display Topics
for index, topic in enumerate(lda.components_):
    print(f'Topic #{index+1}:')
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print(f'Label: {topic_labels[index]}')

# Sentiment Analysis
sia = SentimentIntensityAnalyzer() 
df['Sentiment'] = df['Cleaned_Content'].apply(lambda x: sia.polarity_scores(x)['compound'])

topic_values = lda.transform(doc_term_matrix)
df['Dominant_Topic'] = topic_values.argmax(axis=1)
df['Topic_Label'] = df['Dominant_Topic'].map(topic_labels)

# Sentiment Categories
conditions = [
    (df['Sentiment'] > 0.05),
    (df['Sentiment'] < -0.05),
    (df['Sentiment'].between(-0.05, 0.05))
]
choices = ['positiv', 'negativ', 'neutral']
df['Sentiment_Label'] = np.select(conditions, choices, default='neutral')

# Average Sentiment by Topic
sentiment_by_topic_named = df.groupby('Topic_Label')['Sentiment'].mean()


# Visualizations
# Top words per Topic
fig, axes = plt.subplots(1, 4, figsize=(15, 7), sharex=True)
for idx, topic in enumerate(lda.components_):
    top_features_indices = topic.argsort()[:-11:-1]
    top_features = [vectorizer.get_feature_names_out()[i] for i in top_features_indices]
    weights = topic[top_features_indices]

    axes[idx].barh(top_features, weights)
    axes[idx].set_title(f'Topic {idx+1}: {topic_labels[idx]}')
    axes[idx].invert_yaxis()

fig.suptitle('Top words per Topic', fontsize=16)
plt.tight_layout()
plt.show()

# Distribution of Articles by Year
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Year', order=sorted(df['Year'].unique()))
plt.title('Count of Articles per Year')
plt.xlabel('Year')
plt.ylabel('Count of Articles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Sentiment by Topic
colors = ['red' if val < -0.05 else 'green' if val > 0.05 else 'gray' for val in sentiment_by_topic_named]
sentiment_by_topic_named.plot(kind='bar', color=colors, title='Average Sentiment by Topic', figsize=(10, 6))
plt.xlabel('Topic')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Distribution of Sentiment Classes
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Sentiment_Label', order=['positiv', 'neutral', 'negativ'])
plt.title('Distribution of Sentiment Classes')
plt.xlabel('Sentiment Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Boxplot of Sentiment by Topic
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='Topic_Label', y='Sentiment')
plt.title('Distribution of Sentiment by Topic')
plt.xlabel('Topic')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Article count by Source
source_counts = df['Source'].value_counts()
source_grouped = df['Source'].apply(lambda x: x if source_counts[x] > 10 else 'Other')
df['Source_Grouped'] = source_grouped

# Define order for countplot
order = df['Source_Grouped'].value_counts().index.tolist()
if 'Other' in order:
    order.remove('Other')
    order.append('Other')

# Define color palette
palette = ['grey' if source == 'Other' else 'C0' for source in order]

plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Source_Grouped', order=order, palette=palette)

plt.title('Article count by source (> 10)')
plt.xlabel('Source')
plt.ylabel('Count of Articles')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()