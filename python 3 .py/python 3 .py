import nltk
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

nltk.download('wordnet')
nltk.download('omw-1.4')

headlines = []

n = int(input("Enter number of headlines: "))
for i in range(n):
    headlines.append(input("Enter headline: "))

# Expt.No:3 EXPLORE VARIOUS TEXT SIMILARITY METRICS, INCLUDING WORDNET-BASED SIMILARITY, FOR
# CLUSTERING NEWS HEADLINES INTO TOPICS. Date:

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(headlines)

print("\nCosine Similarity Matrix:")
print(cosine_similarity(X))

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
kmeans.fit(X)

print("\nHeadline Clusters:")
for i in range(len(headlines)):
    print(headlines[i], "-> Cluster", kmeans.labels_[i])

w1 = input("\nEnter first word: ")
w2 = input("Enter second word: ")

s1 = wn.synsets(w1)
s2 = wn.synsets(w2)

if s1 and s2:
    sim = s1[0].path_similarity(s2[0])
    print("WordNet Similarity:", sim)
else:
    print("Similarity not found")