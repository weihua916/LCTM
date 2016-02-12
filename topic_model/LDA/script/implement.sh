#!/bin/bash
# run an toy example for LDA

K=20   # number of topics
niter=100
input_dir=../input/
output_dir=../output/

#path to the input document
doc_file=corpus.txt

echo "=============== Index Docs ============="
python index_document.py -i ${input_dir} -f ${doc_file}
#indexed file will be called indexed_corpus.txt
#vocabulary file will be called vocab.txt

echo "=============== Inferreing Topics ============="
make -C ../src
../src/model -algo CGS -est_hyper 0 -n_topics $K -n_iter $niter -doc_sentence_file indexed_corpus.txt -input_dir $input_dir -output_dir $output_dir

echo "=============== print Topics ============="
python show_topics.py -o $output_dir -i $input_dir
