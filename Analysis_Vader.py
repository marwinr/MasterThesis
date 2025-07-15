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
from collections import Counter



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
    'because', 'although', 'though', 'even', 'unless','nh',
    'whereas','new', 'would', 'year','bbc','mr','uk','u', 'week', 
    'musk', 'china', 'trump', 'president', 'government', 'people', 'country', 'biden'
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

lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(doc_term_matrix)

# Manual Topic Labels
topic_labels = {
    0: 'Economy and Human Integration',
    1: 'Innovation and Economy',
    2: 'Energy and Environment'
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

# Average Sentiment by Year
sentiment_by_year = df.groupby('Year')['Sentiment'].mean().reset_index()


# Visualizations
# Top words per Topic
fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharex=True)
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

# Sentiment by Year
plt.figure(figsize=(10, 5))
sns.lineplot(data=sentiment_by_year, x='Year', y='Sentiment', marker='o')
plt.title('Average Sentiment by Year')
plt.xlabel('Year')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=45)
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

## advanced visualization with pyLDAvis
# Assign dominant topic per document
df['Dominant_Topic'] = np.argmax(lda.transform(doc_term_matrix), axis=1)
df['Topic_Label'] = df['Dominant_Topic'].map(topic_labels)

# Display Topics
for index, topic in enumerate(lda.components_):
    print(f'Topic #{index+1}:')
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print(f'Label: {topic_labels[index]}')


# Sentiment by Topic over Time
plt.figure(figsize=(12, 6))
sentiment_trend = df.groupby(['Year', 'Topic_Label'])['Sentiment'].mean().reset_index()
sns.lineplot(data=sentiment_trend, x='Year', y='Sentiment', hue='Topic_Label', marker='o')
plt.title('Sentiment by Topic over Time')
plt.xlabel('Year')
plt.ylabel('Average Sentiment')
plt.grid(True)
plt.tight_layout()
plt.show()

# Topics over Time
topic_distribution = df.groupby(['Year', 'Topic_Label']).size().reset_index(name='Count')
plt.figure(figsize=(12, 6))
sns.lineplot(data=topic_distribution, x='Year', y='Count', hue='Topic_Label', marker='o')
plt.title('Topics over Time')
plt.xlabel('Year')
plt.ylabel('Count of Articles')
plt.grid(True)
plt.tight_layout()
plt.show()

# Top words per Topic and Sentiment + export to Excel
top_words_rows = []

for topic_num, label in topic_labels.items():
    pos_docs = df[(df['Dominant_Topic'] == topic_num) & (df['Sentiment'] > 0.2)]['Cleaned_Content']
    pos_words = ' '.join(pos_docs).split()
    top_pos_words = Counter(pos_words).most_common(10)

    for word, freq in top_pos_words:
        top_words_rows.append({'Topic': label, 'Sentiment': 'Positive', 'Word': word, 'Frequency': freq})

    neg_docs = df[(df['Dominant_Topic'] == topic_num) & (df['Sentiment'] < -0.2)]['Cleaned_Content']
    neg_words = ' '.join(neg_docs).split()
    top_neg_words = Counter(neg_words).most_common(10)

    for word, freq in top_neg_words:
        top_words_rows.append({'Topic': label, 'Sentiment': 'Negative', 'Word': word, 'Frequency': freq})

# Convert to DataFrame and export to Excel
top_words_df = pd.DataFrame(top_words_rows)
top_words_df.to_excel('top_words_by_topic_and_sentiment.xlsx', index=False)

# Top documents by Topic and Sentiment
rows = []
for topic_num, label in topic_labels.items():
    top_pos = df[df['Dominant_Topic'] == topic_num].sort_values(by='Sentiment', ascending=False).head(3)
    for i, row in top_pos.iterrows():
        rows.append({
            'Topic': label,
            'Sentiment': row['Sentiment'],
            'Type': 'Positive',
            'Headline': row['Headline'],
            'Content': row['Cleaned_Content']
        })

    top_neg = df[df['Dominant_Topic'] == topic_num].sort_values(by='Sentiment').head(3)
    for i, row in top_neg.iterrows():
        rows.append({
            'Topic': label,
            'Sentiment': row['Sentiment'],
            'Type': 'Negative',
            'Headline': row['Headline'],
            'Content': row['Cleaned_Content']
        })

# Convert to DataFrame and export to Excel
top_docs_df = pd.DataFrame(rows)
top_docs_df.to_excel('top_documents_by_topic_and_sentiment.xlsx', index=False)
