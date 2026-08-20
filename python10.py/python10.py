from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

posts = []

n = int(input("Enter number of posts: "))

for i in range(n):
    post = input("Enter post: ")
    posts.append(post)

k = int(input("Enter number of clusters: "))

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(posts)

model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

model.fit(X)

labels = model.labels_

print("\nCluster Results:\n")

for i in range(len(posts)):
    print("Post:", posts[i])
    print("Cluster:", labels[i])
    print()

terms = vectorizer.get_feature_names_out()

print("Important Keywords:\n")

for i in range(k):

    center = model.cluster_centers_[i]
    top = center.argsort()[-5:]

    print("Cluster", i)

    for j in top:
        print(terms[j])

    print()

print("Marketing Insight:")
print("Similar customer opinions are grouped together.")
print("Clusters help identify product trends and issues.")