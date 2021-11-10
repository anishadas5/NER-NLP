# NER-NLP
Named entity recognition is a process where the named entity gets identified and linked to its class. As we know that any given raw text data consists of various kinds of words like some of them are stopwords, part of speech words likewise there can be various kind words that can be presented in a text file which can be segregated as named entities. These words do not represent any feeling but they can represent the relationship between two sentences or two words.
Eg;
“Rahul sold his Maruti 800 at rupees 50000 in 2015”

And the named entity recognition system will give results as 

“Rahul(person) sold his Maruti 800 (car/object) at rupees 50000 (price) in 2015 (time)”

**Let’s start with the importing library.**

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

**NLTK provides some already tagged sentences, we can check it using the treebank package.**

nltk.download('treebank')
sent = nltk.corpus.treebank.tagged_sents()
print(nltk.ne_chunk(sent[0]))

**We can also use NLTK for NER in our sample text.

Before extracting the named entity we need to tokenize the sentence and give them part of the speech tag to the tokenized words.**

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
raw_words= word_tokenize(raw_text)
tags=pos_tag(raw_words)

**Now we can perform NER on the changed sample using the ne_chunk module of the NLTK.**

nltk.download('maxent_ne_chunker')
nltk.download('words')
ne = nltk.ne_chunk(tags,binary=True)
print(ne)

Finally we get the desired output of Name Entity recognition.
