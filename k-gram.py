import numpy as np
import os
import string
import re
from nltk import word_tokenize
import unicodedata

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
    
    def make_kgrams(self, text,  k):
        kgrams = set()
        for i in range(len(text) - k + 1):
            kgrams.add(text[i:i+k])
        return kgrams


    def createShingleMapping(self):
        for root,dir,files in os.walk('Originals'):
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
        pass

shingle_obj = Shingle()
shingle_obj.createShingleMapping()
shin_dict = shingle_obj.shingle_dict
print(shin_dict)
# reader_obj = Reader()
