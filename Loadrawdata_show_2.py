import numpy as np
import matplotlib.pyplot as plt
class Loadrawdata_show_train():##  Load data
    #read raw data from files
    def load(self,filename):
        fread = open(filename, 'r', encoding='utf8')
        freport = open("logs/report.txt", "a", encoding="utf8")
        fword = open('Raw_Data/words.txt', 'w', encoding='utf8')
        ftags = open('Raw_Data/tags.txt', 'w', encoding='utf8')
        fsentence = open('Raw_Data/sentences.txt', 'w', encoding='utf8')
        creating_sent = []
        self.words = []
        self.tags = []
        self.sentence = []
        len_word=0
        len_sentence=0
        t=()
        end_of_line=[".","؟","؛","!"]
        for line in fread:
            if line=='\n':
                pass
            else:
                line = line.replace("\n", "")
                line = line.replace(u'\ufeff', "")
                line = line.replace(u'\u200c', "")
                line = line.replace(u'\u200e', "")
                line = line.replace(u'\u200f\\', "")
                line = line.replace(u'\u200f', "")
                data = line.split()
                self.words.append(data[0])
                len_word+=len(data[0])
                try:
                    self.tags.append(data[1])
                except Exception as err:
                    print(line)
                if data[0] not in end_of_line:
                    #print(data)
                    tuplex=(data[0],data[1])
                    creating_sent.append(tuplex)
                else:
                    tuplex = (data[0], data[1])
                    creating_sent.append(tuplex)
                    len_sentence+=len(creating_sent)
                    self.sentence.append(creating_sent)
                    creating_sent = []
        n_sentence=len(self.sentence)
        freport.write("all_word---------" + str(len_word) + "\n")
        freport.write("all_sentence---------" + str(len_sentence) + "\n")
        freport.write("n_sentence---------" + str(n_sentence) + "\n")
        self.unic_words = list(dict.fromkeys(self.words))
        self.uniq_tags=list(dict.fromkeys(self.tags))
        self.n_words=len(self.unic_words)
        fword.write(str(self.unic_words))
        self.n_tags=len(self.uniq_tags)
        ftags.write(str(self.uniq_tags))
        fsentence.write(str(self.sentence))
        print("len words are: "+str(self.n_words))
        print("len tags are: " + str(self.n_tags))
        #show histogram of sentences
        plt.hist([len(sen) for sen in self.sentence])
        plt.savefig('Raw_Data/hist.png')
        plt.show()


        return self.sentence,self.unic_words,self.uniq_tags,self.n_words,self.n_tags