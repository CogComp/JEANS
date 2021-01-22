import gensim


WIKI_DUMP_FILEPATH = "./data/wiki/fr.txt"
DICTIONARY_FILEPATH = "./data/wiki/fr_dict.txt"
# Construct corpus
wiki = gensim.corpora.textcorpus.TextCorpus(WIKI_DUMP_FILEPATH)
# Remove words occuring less than 20 times, and words occuring in more
# than 10% of the documents. (keep_n is the vocabulary size)
print("finish reading")
wiki.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=500000)
wiki.dictionary.save_as_text(DICTIONARY_FILEPATH)

# # Load dictionary from file
# dictionary = gensim.corpora.Dictionary.load_from_text(DICTIONARY_FILEPATH)
#
# # Construct corpus using dictionary
# wiki = gensim.corpora.WikiCorpus(WIKI_DUMP_FILEPATH, dictionary=dictionary)
#
