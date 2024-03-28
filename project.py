## Feb 10th

# from transformers import BertTokenizer, BertForSequenceClassification, pipeline
# finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels = 3)
# tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

# nlp = pipeline('sentiment-analysis', model = finbert, tokenizer = tokenizer)
# sentences = ['there is a shortage of capital, and we need extra financing', 
#              'growth is strong and we have plenty of liquidity',
#              'there are doubts about our finances',
#              'profits are increasing']
# results = nlp(sentences)
# print(results) #LABEL_2: NEUTRAL; LABEL_1: POSITIVE; LABEL_2: NEGATIVE

##########################################################################################


## Feb 17th 
# from transformers import pipeline
# summarizer = pipeline("summarization", model = "facebook/bart-large-cnn", truncation = True)

# import requests 
# from bs4 import BeautifulSoup
# ''
# url = 'https://en.wikipedia.org/wiki/One_Hundred_and_One_Dalmatians'
# res = requests.get(url)
# html_page = res.content
# soup = BeautifulSoup(html_page, 'html.parser')
# text = ''
# for paragraph in soup.find_all('p'):
#     text += paragraph.text
# # Import package
# import re
# # Clean text
# text = re.sub(r'\[.*?\]+', '', text)
# text = text.replace('\n', '')
# text
# print(text)
# summary = summarizer(text,max_length=250,min_length=30,do_sample=False)
# print(summary)

from bertopic import BERTopic
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()

# Topic model
from bertopic import BERTopic

# Dimension reduction
from umap import UMAP


# Mount Google Drive
from pydrive.drive import GoogleDrive
from google.colab import drive
drive.mount('/content/drive')

# Change directory
import os
os.chdir("drive/My Drive/contents/nlp")

# Print out the current directory
pwd

# Read in data
amz_review = pd.read_csv('sentiment labelled sentences/amazon_cells_labelled.txt', sep='\t', names=['review', 'label'])

# Drop the label 
amz_review = amz_review.drop('label', axis=1);

# Take a look at the data
amz_review.head()

# Get the dataset information
amz_review.info()

# Remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
print(f'There are {len(stopwords)} default stopwords. They are {stopwords}')

# Remove stopwords
amz_review['review_without_stopwords'] = amz_review['review'].apply(lambda x: ' '.join([w for w in x.split() if w.lower() not in stopwords]))

# Lemmatization
amz_review['review_lemmatized'] = amz_review['review_without_stopwords'].apply(lambda x: ' '.join([wn.lemmatize(w) for w in x.split() if w not in stopwords]))

# Take a look at the data
amz_review.head()

# Initiate UMAP
umap_model = UMAP(n_neighbors=15, 
                  n_components=5, 
                  min_dist=0.0, 
                  metric='cosine', 
                  random_state=100)

# Initiate BERTopic
topic_model = BERTopic(umap_model=umap_model, language="english", calculate_probabilities=True)

# Run BERTopic model
topics, probabilities = topic_model.fit_transform(amz_review['review_lemmatized'])

# Get the list of topics
topic_model.get_topic_info()

# Get top 10 terms for a topic
topic_model.get_topic(0)
# Visualize top topic keywords
topic_model.visualize_barchart(top_n_topics=12)
# Visualize term rank decrease
topic_model.visualize_term_rank()
# Visualize intertopic distance
topic_model.visualize_topics()
# Visualize connections between topics using hierachical clustering
topic_model.visualize_hierarchy(top_n_topics=10)
# Visualize similarity using heatmap
topic_model.visualize_heatmap()
# Visualize probability distribution
topic_model.visualize_distribution(topic_model.probabilities_[0], min_probability=0.015)
# Save the chart to a variable
chart = topic_model.visualize_distribution(topic_model.probabilities_[0]) 

# Write the chart as a html file
chart.write_html("amz_review_topic_probability_distribution.html")
# Check the content for the first review
amz_review['review'][0]
# Get probabilities for all topics
topic_model.probabilities_[0]
# Get the topic predictions
topic_prediction = topic_model.topics_[:]

# Save the predictions in the dataframe
amz_review['topic_prediction'] = topic_prediction

# Take a look at the data
amz_review.head()
# New data for the review
new_review = "I like the new headphone. Its sound quality is great."

# Find topics
num_of_topics = 3
similar_topics, similarity = topic_model.find_topics(new_review, top_n=num_of_topics); 

# Print results
print(f'The top {num_of_topics} similar topics are {similar_topics}, and the similarities are {np.round(similarity,2)}')
for i in range(num_of_topics):
  print(f'The top keywords for topic {similar_topics[i]} are:')
  print(topic_model.get_topic(similar_topics[i]))
  # Save the topic model
topic_model.save("amz_review_topic_model")	

# Load the topic model
my_model = BERTopic.load("amz_review_topic_model")	
#####################################################################
