import nltk

nltk.download('all')
text = "Hello World"
k = 3

# Generate character-wise k-grams
kgrams = ngrams(text, k)

# Print the k-grams
for gram in kgrams:
    print(gram)
