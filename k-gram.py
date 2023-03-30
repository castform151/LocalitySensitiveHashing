import numpy as np
import os
import random
import string
import re
from nltk import word_tokenize
import unicodedata
from math import gcd

class Document:

    def similarity(self,doc1,doc2):
        ctr=0
        for i in range(len(doc1)):
            ctr+= doc1[i]==doc2[i]
        return ctr/len(doc1)

class Reader:
        
    def lowercasify(self, data):
        return data.lower()


    def clean_text(self,text):
        # Remove apostrophes along with other punctuations
        # translator = str.maketrans('', '', string.punctuation + "’" + '“' + '”')
        # text = text.translate(translator)
        # "".join([char for char in text if not self.is_pua(char)])
        text = re.sub(r"[^a-zA-Z0-9 \n\r\t]", "", text)

        # Tokenize the cleaned text
        words = word_tokenize(text)
        return words
    
    
    def preprocess_data(self, data):
        clean_data = ''
        words = self.clean_text(self.lowercasify(data))
        # print(words)
        for word in words:
            clean_data += word + ' '
        return clean_data


class Shingle:
    
    def __init__(self) -> None:
        self.shingle_dict = dict()
        # self.single_doc_mat = np.zeroes(())
        # self.shingleDocMatrix = None
        
    
    def make_kgrams(self, text,  k):
        kgrams = set()
        for i in range(len(text) - k + 1):
            kgrams.add(text[i:i+k])
        return kgrams


    def createShingleMapping(self):
        for root,dir,files in os.walk('Originals'):
            self.numDocs = len(files)
            for i, file in enumerate(files):
                # if (file[0] == '7'):
                txt_file = open(os.path.join(root,file),encoding='utf8')
                text = txt_file.read()
                reader_obj = Reader()
                preprocessed_text = reader_obj.preprocess_data(text)
                k_grams = self.make_kgrams(preprocessed_text, 9)
                for k_gram in k_grams:
                    if k_gram in self.shingle_dict.keys():
                        self.shingle_dict[k_gram].append(i)
                    else:
                        self.shingle_dict[k_gram] = [i]
                # print(self.shingle_dict)
                # print(k_grams)
            # print(files, len(files))


    def createMatrix(self):
        self.shingleDocMatrix = np.zeros((len(self.shingle_dict), self.numDocs), dtype = 'bool')
        for i, shingle in enumerate(self.shingle_dict.keys()):
            for docID in self.shingle_dict[shingle]:
                self.shingleDocMatrix[i][docID] = True
    
class MinHash:
    def __init__(self, shingleDocMatrix, numofHashFuncs):
        self.maxShingles = shingleDocMatrix.shape[0]
        self.nextPrime = self.getSmallestPrime(self.maxShingles)
        self.signatureMatrix = np.full((numofHashFuncs, shingleDocMatrix.shape[1]), self.nextPrime) 
        self.coeffA = self.pickRandomCoeffs(numofHashFuncs)
        self.coeffB = self.pickRandomCoeffs(numofHashFuncs)
        
    
    def miller_rabin(n, k):

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
    
    def getSmallestPrime(self, n):
        while True:
            if self.miller_rabin(n, 40):
                return n
            n += 1
            
    def pickRandomCoeffs(self, k):
        # Create a list of 'k' random values.
        randList = []
        while k > 0:
            # Get a random shingle ID.
            randIndex = random.randint(1, self.maxShingles) 
        
            # Ensure that each random number is unique.
            while (randIndex in randList  and gcd(randIndex, self.nextPrime) != 1):
                randIndex = random.randint(1, self.maxShingles) 
            
            # Add the random number to the list.
            randList.append(randIndex)
            k = k - 1
            
        return randList
    
    def Hash(self, shingleIndex, HashFuncNum):
        # Evaluate the hash function.
        return ((self.coeffA[HashFuncNum] * shingleIndex) + self.coeffB[HashFuncNum]) % self.nextPrime
    
    def fillSignatureMatrix(self, shingle_dict):
        for i, shingle in enumerate(shingle_dict.keys()):
            for docID in shingle_dict[shingle]:
                # _temp = self.Hash(i, 0)
                # if self.signatureMatrix[j][docID] == 0:
                for minHashNum in range(self.numofHashFuncs):
                    _temp = self.Hash(i, minHashNum)
                    if self.signatureMatrix[minHashNum][docID] > _temp:
                        self.signatureMatrix[minHashNum][docID] = _temp
        # print(self.signatureMatrix)
        # return self.signatureMatrix
        
        
    
    
# shingle_obj = Shingle()
# shingle_obj.createShingleMapping()
# shin_dict = shingle_obj.shingle_dict
# print(shin_dict)
# reader_obj = Reader()
