import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE


class stop_words:
	"""Utility class for managing stop words."""
	def __init__(self, words=None):
		# initialize with sklearn english stop words by default if available
		if words is None:
			try:
				from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
				self._words = set(ENGLISH_STOP_WORDS)
			except Exception:
				self._words = set()
		else:
			self._words = set(words)

	def get_stop_words(self):
		"""Return the set of stop words."""
		return set(self._words)

	def add_stop_words(self, words):
		"""Add one or multiple stop words. Accepts a string or an iterable of strings."""
		if isinstance(words, str):
			self._words.add(words)
		else:
			for w in words:
				self._words.add(w)

	def remove_stop_word(self, word):
		"""Remove a stop word if it exists."""
		self._words.discard(word)

	def is_stop_word(self, word):
		"""Check if a word is a stop word."""
		return word in self._words


class Reviews:
	"""Container for customer reviews with simple helper methods."""

	def __init__(self, items=None):
		self._items = list(items) if items is not None else []

	def add_review(self, review):
		"""Add one review."""
		self._items.append(str(review))

	def extend(self, reviews):
		"""Add multiple reviews."""
		for review in reviews:
			self.add_review(review)

	def get_reviews(self):
		"""Return reviews as a list."""
		return list(self._items)

	def as_corpus(self):
		"""Return reviews for vectorization."""
		return self.get_reviews()

	def __len__(self):
		return len(self._items)

	def __iter__(self):
		return iter(self._items)

reviews = Reviews()
n = int(input("Enter number of reviews: "))

for i in range(n):
	reviews.add_review(input("Enter review: "))

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews.as_corpus())

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

words = vectorizer.get_feature_names_out()

print("\nTopics:")
for i, topic in enumerate(lda.components_):
	print("\nTopic", i + 1)
	top_words = topic.argsort()[-5:]
	for j in top_words:
		print(words[j])

X_dense = X.toarray()

tsne = TSNE(n_components=2, random_state=42, perplexity=2)
X_tsne = tsne.fit_transform(X_dense)

print("\nt-SNE Coordinates:")
for i, point in enumerate(X_tsne):
	print("Review", i + 1, ":", point)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

for i in range(len(reviews)):
	plt.text(X_tsne[i, 0], X_tsne[i, 1], "R" + str(i + 1))

plt.title("t-SNE Visualization of Customer Reviews")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()