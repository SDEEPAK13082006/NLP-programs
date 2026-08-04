import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
from nltk.tag import hmm
from nltk.corpus import treebank

# Download required datasets
nltk.download('punkt')
nltk.download('treebank')

# -----------------------------
# Input Tweet
# -----------------------------
tweet = input("Enter a tweet: ")

# Tokenization
tokens = nltk.word_tokenize(tweet.lower())

print("\nTokens:")
print(tokens)

# -----------------------------
# N-Gram Language Model
# -----------------------------
print("\n========== N-GRAM MODEL ==========")

# Unigrams
unigrams = list(ngrams(tokens, 1))
print("\nUnigrams:")
print(unigrams)

# Bigrams
bigrams = list(ngrams(tokens, 2))
print("\nBigrams:")
print(bigrams)

# Trigrams
trigrams = list(ngrams(tokens, 3))
print("\nTrigrams:")
print(trigrams)

# Word Frequency
fd = FreqDist(tokens)

print("\nWord Frequencies:")

for word, freq in fd.items():
	print(word, ":", freq)

# -----------------------------
# Hidden Markov Model (HMM)
# -----------------------------
print("\n========== HMM MODEL ==========")

# Train HMM using Treebank corpus
train_data = treebank.tagged_sents()[:3000]

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)

# Predict POS tags
tagged_sentence = hmm_tagger.tag(tokens)

print("\nHMM POS Tagging:")
for word, tag in tagged_sentence:
	print(word, "->", tag)

# -----------------------------
# Comparison
# -----------------------------
print("\n========== COMPARISON ==========")

print("N-Gram Model")
print("- Learns word sequences.")
print("- Predicts the next word based on previous words.")
print("- Used for text generation and language modeling.")

print("\nHMM Model")
print("- Predicts Part-of-Speech (POS) tags.")
print("- Uses transition and emission probabilities.")
print("- Used for sequence labeling tasks.")