class WordTagsToIdx_train():
   def word_tags_to_idx(self,myobj,MAX_LEN_vector):
        self.sentence,uniq_words, uniq_tags,self.n_words,self.n_tags = myobj
        self.max_len_vector=MAX_LEN_vector

####################################################################
        # Vocabulary Key:word -> Value:token_index
        # The first 2 entries are reserved for PAD and UNK
        self.word2idx = {w: i + 2 for i, w in enumerate(uniq_words)}
        self.word2idx["UNK"] = 1  # Unknown words
        self.word2idx["PAD"] = 0  # Padding
        # Vocabulary Key:token_index -> Value:word
        self.idx2word = {i: w for w, i in self.word2idx.items()}

###################################################################

        # Vocabulary Key:Label/Tag -> Value:tag_index
        # The first entry is reserved for PAD
        self.tag2idx = {t: i + 1 for i, t in enumerate(uniq_tags)}
        self.tag2idx["PAD"] = 0
        # Vocabulary Key:tag_index -> Value:Label/Tag
        self.idx2tag = {i: w for w, i in self.tag2idx.items()}
        freport=open("logs/report.txt","a",encoding="utf8")
        freport.write("word2idx---------"+str(self.word2idx)+"\n")
        freport.write("idx2word---------"+str(self.idx2word)+"\n")
        freport.write("tag2idx----------"+str(self.tag2idx) + "\n")
        freport.write("idx2tag----------"+str(self.idx2tag) + "\n")
        freport.write("sentence----------"+str(self.sentence) + "\n")
        freport.write("n_words-----------"+str(self.n_words) + "\n")
        freport.write("n_tags------------"+str(self.n_tags) + "\n")
        return self.word2idx,self.idx2word,self.tag2idx,self.idx2tag,self.sentence,self.n_words,self.n_tags,self.max_len_vector
   def __repr__(self):
        return f'{self.sentence}{self.word2idx}{self.tag2idx}{self.n_words}{self.n_tags}{self.max_len_vector}'




