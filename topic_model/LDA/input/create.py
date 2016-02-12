import util

corpus = util.readlist('corpus.train')
vocab_list = util.readlist('vocab.txt')

new_corpus = []

for doc in corpus:
    doc = map(int, doc.split())
    new_doc = [vocab_list[word] for word in doc]
    new_corpus.append(' '.join(new_doc))

n_docs = len(corpus)
n_vocab = len(vocab_list)
header = str(n_docs) + ' ' + str(n_vocab)

util.write_corpus('corpus.txt', header, new_corpus)




