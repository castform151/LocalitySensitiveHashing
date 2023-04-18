import numpy as np
# import pandas as pd
import os
import random
# import string
import re
from nltk import word_tokenize
# import unicodedata
from math import gcd
from time import time
from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
flag = 0
# class Document:

#     def similarity(self,doc1,doc2):
#         ctr=0
#         for i in range(len(doc1)):
#             ctr+= doc1[i]==doc2[i]
#         return ctr/len(doc1)


class Reader:

    @classmethod
    def normalise(cls, query):
        stop_words = set((stopwords.words("english")))
        query = cls.clean_line(query)
        tokens = word_tokenize(query.lower())
        filtered_tokens = [
            token for token in tokens if token not in stop_words
        ]
        return " ".join(filtered_tokens)

    @classmethod
    def clean_line(cls, text):
        text = re.sub(r"[^a-zA-Z0-9 \n\r\t]", "", text)
        return text

    # def lowercasify(self, data):
    #     return data.lower()

    # def clean_text(self,text):
    #     text = re.sub(r"[^a-zA-Z0-9 \n\r\t]", "", text)

    #     # Tokenize the cleaned text
    #     words = word_tokenize(text)

    #     return words

    # def preprocess_data(self, data):
    #     clean_data = ''
    #     words = self.clean_text(self.lowercasify(data))
    #     # print(words)
    #     for word in words:
    #         clean_data += word + ' '
    #     return clean_data


class Shingle:

    def __init__(self) -> None:
        """Intialize the shingle class
        """
        self.shingle_dict = dict()

    def make_kgrams(self, text,  k=9):
        """Make chracterwise k-grams of length k from the text

        Args:
            text (str): k-grams are made from this text
            k (int): length of characterwise k-grams

        Returns:
            set(): set of chracaterwise k-grams of length k made from text
        """
        kgrams = set()
        for i in range(len(text) - k + 1):
            kgrams.add(text[i:i+k])
        return kgrams

    def createShingleMapping(self):
        """Cretaes a dictionary of k-shingles and the list of documents they are present in
        """
        t0 = time()
        # for root,dir,files in os.walk('Originals'):
        #     self.numDocs = len(files)
        #     for i, file in enumerate(files):
        #         # if (file[0] == '7'):
        #         txt_file = open(os.path.join(root,file),encoding='utf-8')
        #         text = txt_file.read()
        #         reader_obj = Reader()
        #         preprocessed_text = reader_obj.preprocess_data(text)
        #         k_grams = self.make_kgrams(preprocessed_text, 9)
        #         for k_gram in k_grams:
        #             if k_gram in self.shingle_dict.keys():
        #                 self.shingle_dict[k_gram].append(i)
        #             else:
        #                 self.shingle_dict[k_gram] = [i]
        file = open('dataset/originals.txt', encoding='utf-8', errors="ignore")
        str = file.read()
        articles = str.split('END OF DOCUMENT')
        self.numDocs = len(articles)

        for i, text in enumerate(articles):
            preprocessed_text = Reader.normalise(text)
            k_grams = self.make_kgrams(preprocessed_text, 2)
            for k_gram in k_grams:
                if k_gram in self.shingle_dict.keys():
                    self.shingle_dict[k_gram].append(i)
                else:
                    self.shingle_dict[k_gram] = [i]
        print("Time taken to create shingle matrix: ", time() - t0)

    def getJaccrdSimilarity(self, threshold, queryShingle):
        t0 = time()
        global flag
        flag = 0
        query_shin_set = set(queryShingle)
        print("Using Jaccard Similarity on Shingle Matrix")
        for i, j in self.shingle_dict.items():
            doc_shin_set = set(j)
            aib = len(query_shin_set.intersection(doc_shin_set))
            aub = len(query_shin_set.union(doc_shin_set))
            sim = aib/aub
            if sim >= threshold:
                flag = 1
                print(f"Similarity of {sim*100}% with documnet {i}")
        if flag == 0:
            print("No similar document found using Jaccard")
        print("Time taken to find similarity using Jaccard: ", time() - t0)

    # def createMatrix(self):
    #     self.shingleDocMatrix = np.zeros((len(self.shingle_dict), self.numDocs), dtype = 'bool')
    #     for i, shingle in enumerate(self.shingle_dict.keys()):
    #         for docID in self.shingle_dict[shingle]:
    #             self.shingleDocMatrix[i][docID] = True


class MinHash:
    """Class to make minhash signatures for the documents
    """

    def __init__(self, shingle_dict, numofDocs, numofHashFuncs):
        """Intialize the minhash class

        Args:
            shingle_dict (dict{}): Dictionary of k-shingles and the list of documents they are present in   
            numofDocs (int): total number of dcouments in the dataset
            numofHashFuncs (int): number of hash functions to be used to make the minhash signatures
        """
        self.shingle_dict = shingle_dict
        self.numofDocs = numofDocs
        self.numofHashFuncs = numofHashFuncs
        self.maxShingles = len(shingle_dict)
        self.nextPrime = self.getSmallestPrime(self.maxShingles)
        self.signatureMatrix = np.full(
            (numofHashFuncs, numofDocs), self.nextPrime)
        # self.coeffA = self.pickRandomCoeffs(numofHashFuncs)
        # self.coeffB = self.pickRandomCoeffs(numofHashFuncs)
        self.coeffA = self.true_permutation(numofHashFuncs)
        self.coeffB = self.true_permutation(numofHashFuncs)

    def getSmallestPrime(self, n):
        """Get the smallest prime number greater than n

        Args:
            n (int): number greater than which the smallest prime number is to be found

        Returns:
            int: smallest prime number greater than n
        """
        def miller_rabin(n, k):
            """Implements Probabilistic Primality Test using Miller-Rabin Test

            Args:
                n (int): primality of this number is to be tested
                k (int): Miller-Rabin test is run for k iterations

            Returns:
                bool: False if n is composite, True if n is probably prime
            """
            # Implementation uses the Miller-Rabin Primality Test
            # The optimal number of rounds for this test is 40
            # See http://stackoverflow.com/questions/6325576/how-many-iterations-of-rabin-miller-should-i-use-for-cryptographic-safe-primes
            # for justification

            if n == 2:
                return True

            elif n % 2 == 0:
                return False

            r, s = 0, n - 1
            while s % 2 == 0:
                r += 1
                s //= 2
            for _ in range(k):
                a = random.randrange(2, n - 1)
                x = pow(a, s, n)
                if x == 1 or x == n - 1:
                    continue
                for _ in range(r - 1):
                    x = pow(x, 2, n)
                    if x == n - 1:
                        break
                else:
                    return False
            return True
        t0 = time()
        while True:
            if miller_rabin(n, 40):
                print(f"Time taken to find next prime {n}: ", time() - t0)
                return n
            n += 1

    def pickRandomCoeffs(self, k):
        """Pick k random coefficients for the random hash functions.

        Args:
            k (int): number of random coefficients to be picked

        Returns:
            list: list of k random coefficients
        """
        t0 = time()
        randList = []
        while k > 0:
            randIndex = random.randint(1, self.maxShingles)

            # Ensure that the same value is not picked twice and GCD of Modulus and Coefficient is 1
            while (randIndex in randList and gcd(randIndex, self.nextPrime) != 1):
                randIndex = random.randint(1, self.maxShingles)

            randList.append(randIndex)
            k = k - 1
        print("Time taken to pick random coefficients: ", time() - t0)
        print(randList)
        return randList

    def true_permutation(self, k):
        t0 = time()
        randList = []
        while k > 0:
            randIndex = random.randint(1, self.maxShingles)

            # Ensure that the same value is not picked twice and GCD of Modulus and Coefficient is 1
            while (randIndex in randList and gcd(randIndex, self.maxShingles) != 1):
                randIndex = random.randint(1, self.maxShingles)

            randList.append(randIndex)
            k = k - 1
        print("Time taken to pick random coefficients: ", time() - t0)
        print(randList)
        return randList

    def Hash(self, shingleIndex, HashFuncNum):
        """Hash the shingle to a bucket

        Args:
            shingleIndex (int): index of the shingle in the shingle dictionary
            HashFuncNum (int): number of the MinHash function to be used

        Returns:
            int: bucket number to which the shingle is hashed for given MinHash function
        """
        # return ((self.coeffA[HashFuncNum] * shingleIndex) + self.coeffB[HashFuncNum]) % self.nextPrime
        return ((self.coeffA[HashFuncNum] * shingleIndex) + self.coeffB[HashFuncNum]) % self.maxShingles

    def fillSignatureMatrix(self):
        """Fills the signature matrix with the minhash signatures
        """
        t0 = time()
        for i, shingle in enumerate((self.shingle_dict.keys())):
            for docID in self.shingle_dict[shingle]:
                for minHashNum in range(self.numofHashFuncs):
                    _temp = self.Hash(i, minHashNum)
                    if self.signatureMatrix[minHashNum][docID] > _temp:
                        self.signatureMatrix[minHashNum][docID] = _temp
        print("Time taken to fill signature matrix: ", time() - t0)


class LSH:
    """Class to implement the LSH algorithm to find the similar documents
    """

    def __init__(self, rowsperBand, numofBands, signetureMatrix) -> None:
        """Initialises LSH class

        Args:
            rowsperBand (int): number of rows per band in Signature Matrix
            numofBands (int): Total number of bands in Signature Matrix
            signatureMatrix (np.ndarray): Signature matrix of all the documents in the dataset
        """
        self.rowsperBand = rowsperBand
        self.numofBands = numofBands
        self.signatureMatrix = signetureMatrix

    def getSignatureSimilarity(self, threshold, querySignature):
        """Checks for plagiarism of query documnet in the dataset

        Args:
            threshold (float): threshold for similarity between query document and the documents in the dataset
            querySignature (np.ndarray): signature vector of the query document
        """
        t0 = time()
        print("Using LSH on Signature Matrix")
        simBand = 0
        global flag
        flag = 0
        for i in range(self.signatureMatrix.shape[1]):
            simBand = 0
            for j in range(self.numofBands):
                if np.array_equal(self.signatureMatrix[j:j+self.rowsperBand, i], querySignature[j:j+self.rowsperBand]):
                    simBand += 1
            # print(simBand/self.numofBands)
            sim = simBand/self.numofBands
            if sim >= threshold:
                flag = 1
                print(f"Similarity of {sim} with documnet {i}")
        if flag == 0:
            print("No similar document found using LSH")
        print("Time taken to find similar documents using LSH: ", time() - t0)


class Query:
    """Class to generate the query shingle vector and the query signature
    """

    def __init__(self, query_path, Shingle, MinHash, q) -> None:
        """Intialize the query class

        Args:
            query_path (str): relative path to the query document
            Shingle (Shingle): Instance of Shingle class
            MinHash (MinHash): Instance of MinHash class
        """
        query_path = query_path + os.listdir(query_path)[q]
        print(f"For {query_path}")
        query_file = open(query_path, encoding='utf-8', errors='ignore')
        # reader_obj = Reader()
        # preprocessed_text = reader_obj.preprocess_data(query_file.read())
        self.query = Reader.normalise(query_file.read())
        self.ShingleObj = Shingle
        self.MinHashObj = MinHash
        self.getQueryShingleVec()

    def getQueryShingleVec(self):
        """Generate the shingle vector for the query

        Returns:
            list[]: Returns the indices of the shingles in the shingle dictionary that are present in query document
        """
        queryShingles = self.ShingleObj.make_kgrams(self.query, 2)
        self.shingle_vec = []
        for i, shingle in enumerate((self.ShingleObj.shingle_dict.keys())):
            if shingle in queryShingles:
                self.shingle_vec.append(i)
        # print(shingle_vec)
        # self.shingle_vec

    def getQuerySignature(self):
        """Generates Signature for the query document

        Returns:
            np.ndarray: Returns the signature vector for the query document
        """
        # queryShingleVec = self.getQueryShingleVec()
        querySignature = np.full(
            (self.MinHashObj.numofHashFuncs), self.MinHashObj.nextPrime)
        for i in self.shingle_vec:
            for j in range(self.MinHashObj.numofHashFuncs):
                _temp = self.MinHashObj.Hash(i, j)
                if querySignature[j] > _temp:
                    querySignature[j] = _temp
        # print(querySignature)
        return querySignature


if __name__ == "__main__":
    shingle_obj = Shingle()
    shingle_obj.createShingleMapping()
    print("Shingle mapping created")
    shin_dict = shingle_obj.shingle_dict
    min_hash_obj = MinHash(shin_dict, shingle_obj.numDocs, 50)
    min_hash_obj.fillSignatureMatrix()
    print("fill signature matrix")
    # print(min_hash_obj.signatureMatrix)
    lsh_obj = LSH(5, 10, min_hash_obj.signatureMatrix)
    threshold = 0.5
    for q in range(5):
        query_obj = Query("Query_Doc/", shingle_obj, min_hash_obj, q)
        lsh_obj.getSignatureSimilarity(
            threshold, query_obj.getQuerySignature())
        shingle_obj.getJaccrdSimilarity(threshold, query_obj.shingle_vec)
    # print(shin_dict)
