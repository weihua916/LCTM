import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import util
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', default='',help='input directory')
    parser.add_argument('--file', '-f', default='',help='filename of corpus')
    args = parser.parse_args()

    return args.input_dir, args.file


def main():
    input_dir, file = parse()
    
    #line of alphabet
    corpus = util.read_list(input_dir +  file)

    vocab_list = set([])
    for doc in corpus:
        vocab_list = vocab_list.union(set(doc.split()))

    vocab_list = list(vocab_list)
    vocab2index = dict(zip(vocab_list, map(str, range(len(vocab_list)))))

    #line of word index
    newcorpus = []

    for doc in corpus:
        new_doc = [vocab2index[word] for word in doc.split()]
        newcorpus.append(' '.join(new_doc))

    header = str(len(newcorpus)) + ' ' + str(len(vocab_list))

    util.write_corpus(input_dir + 'indexed_corpus.txt', header, newcorpus)
    util.write_list(input_dir + 'vocab.txt', vocab_list)


if __name__ == '__main__':
    main()

