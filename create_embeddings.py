from gensim.corpora import WikiCorpus
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import codecs
import sys
import os
import logging

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

EMBEDDING_DIR = "embeddings"
FILE_NAME = "nlwiki-20161101-pages-articles-multistream.xml.bz2"
WIKI_FILE = "{}/{}".format(EMBEDDING_DIR, FILE_NAME)


LANGUAGE = "ned"
output_file = "{}/{}_temp.bin".format(EMBEDDING_DIR, LANGUAGE)



print("converting wiki to readable output")
with codecs.open(output_file, "w") as output:
    print("reading corpus")
    wiki = WikiCorpus(WIKI_FILE, lemmatize=False, dictionary={})
    print("Writing corpus")
    for index, text in enumerate(wiki.get_texts()):
        output.write(" ".join(text) + "")
        if (index % 10000 == 0):
            logger.info("Saved {} articles".format(index))

print("Done")
