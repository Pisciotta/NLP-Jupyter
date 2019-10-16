
# coding: utf-8

# ### NLTK

# NLTK (Natural Language Toolkit) is a Python platform for human language data processing.
# <p>A complete list of NLTK modules can be found here: https://www.nltk.org/py-modindex.html</p>

# ### Open Multilingual Wordnet (OMW)

# <p>While a <b>dictionary</b> is used to define words, a <b>thesaurus</b> groups words with similar meaning. <b>WordNet</b> is a lexical database for the English combining dictionary and thesaurus functionalities.</p>
# <p>The goal of <b>Open Multilingual Wordnet</b> is to make it easy to use wordnets in multiple languages.</p>
# <p>To use Open Multilingual WordNet you need to download it (Corpora -> 'owm') interactively using a GUI interface of the <i>NLTK Downloader</i> from Python terminal:</p>

# In[1]:


import nltk
nltk.download()


# <i>Note: if you only download wordnet instead of owm, there will be no possibility to work with other languages</i>
# For the following examples you also need to download and import further corpora:
# - webtext (containing files in txt format)
# - stopwords (to filter out stopwords)

# In[2]:


from nltk.corpus import wordnet
from nltk.corpus import webtext
from nltk.corpus import stopwords


# ### Corpora

# Each corpus has different files containing some text. To get a list of such files of e.g. the above imported wordnet (as wn) corpus, run:

# In[3]:


print(wordnet.fileids())


# To get the list of words inside a corpus we use the .words() method:

# In[4]:


print(webtext.words())


# ### Wordnet (OMW) => Synset basics

# <p><b>Synset</b> are wordnet instances grouping synonymous words that express the same concept.</p>

# In[5]:


syn = wordnet.synsets('fantasma', lang='ita')
print(syn)


# In[6]:


print("NAME: ",syn[0].name())
print("DEFINITION: ",syn[0].definition())
print("EXAMPLES: ",syn[0].examples())


# In[7]:


print(syn[0].lemmas(lang='ita'))
print(syn[0].hypernyms())


# ### Nltk.collocations

# <b>Collocations</b> are terms that can co-occur inside a sentence because of some kind of relation given by the common use of the language.
# 
# A <b>bigram</b> is a sequence of two adjacent elements. What we are going to achieve with the following code is to list bigrams according to their collocation probability ranking / order within a given corpus (in our case webtext). So, <b>after stopwords filtering</b>, bigrams that co-occur more often than others are nearer to the top.

# In[8]:


from nltk.collocations import BigramCollocationFinder 
from nltk.metrics import BigramAssocMeasures


stopset = set(stopwords.words('english'))
filter_stops = lambda w: len(w) < 3 or w in stopset

# Load all words of webtext corpus
words = [w.lower() for w in webtext.words()] 

# Creating a BigramCollocationFinder object for later use
# from the previously created corpus words list
bigramColloc = BigramCollocationFinder.from_words(words)

# Filtering stopwords out from the BigramCollocationFinder object
bigramColloc.apply_word_filter(filter_stops)

# Get the first 20 results of two-word-sequences in our (already filtered)
# corpus according to the likelihood of both words to be found together
# in the shown sequence
bigramColloc.nbest(BigramAssocMeasures.likelihood_ratio, 20)


# In[9]:


# Here, instead, bigrams are ordered according to their frequency
# in the corpus
bigramColloc.nbest(BigramAssocMeasures.raw_freq, 20)

