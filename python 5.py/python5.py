import nltk
from nltk import word_tokenize, pos_tag

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

text = input("Enter legal text: ")

tokens = word_tokenize(text)

tags = pos_tag(tokens)

print("\nDetected Named Entities:")
count = 0

for word, tag in tags:
    if tag == "NNP":
        print(word, "-> ENTITY")
        count += 1

actual = int(input("\nEnter actual number of entities: "))

accuracy = (min(count, actual) /
max(count, actual)) * 100

print("\nPredicted Entities:", count)
print("NER Accuracy:", round(accuracy, 2), "%")