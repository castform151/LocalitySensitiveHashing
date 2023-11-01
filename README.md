# Locality Sensitive Hashing

## Dataset
[20newsgroup](https://www.kaggle.com/datasets/crawford/20-newsgroups) dataset was used. It has around 18k new articles divided into topics. We used subset of this dataset for our experimentation

## Pre-processing
1. Special characters were removed and text was converted to lowercase.
2. Stopwords were not removed as aim was to detect similar or almost same documnets. Stemming was also not done for the same reason.

## Choice of k-shingles
1. Character-wise shingles ($k = 9$) generated high number of false positives in the end.
2. So, token-wise trigram shingles ($k = 3$) was used.

## Choice of Locality Sensitive Function
Jaccard distance and corresponding LSH family of MinHash signatures was used for this. Where jaccard distnace $d(x, y)$ is
$$ d(x, y) = 1 - \frac{x \cap y}{x \cup y} $$
where $x$ represents set of shingles from a document.

MinHash signatures were computed using minHash algorithm. It was ensured that hash functions used in MinHash algorithm generated unique permutations. Hash functions were randomly generaetd and varied accross each run of the programm.

## Experiment with banding techniques
We experimented with for 3 different values of number of hash functions and 2 different values of bands per rows. Scores were averages accross 5 runs.

Probablity of documents becoming candidate pairs is
$$ p(colision) = 1 - (1 - s^r)^b $$
where
- $s$ is probablity of documents having same row in one band of signature matrix. It effectively corresponds to jaccard similarity of documents.
- $r$ is number of rows per band.
- $b$ is number of bands. 
- $n = b \times r$ where n is number of hash functions.

## Running the code
Run `python3 hash.py` to generate subset of dataset

Run `python3 main.py` to get Documnets similar to documents in `Query_Doc/` folder.

To change values of number of Hash Fucntions and number of hash functions per band refer the end of the code.

Please check `main.html` for detailed documentation of all classes and functions
