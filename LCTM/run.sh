n_iter=1500
wv_file=glove.txt
doc_file=indexed_corpus.txt
n_each=10_short
input_dir=input/sample/
output_dir=output/sample/
echo $output_dir
echo $input_dir

mkdir -p $output_dir

n_topics=20
n_concepts=1000
noise=0.5

concept_file=${n_concepts}concepts.txt
./model -n_topics $n_topics -n_iter $n_iter -noise $noise -wv_file $wv_file -doc_file $doc_file -concept_file $concept_file -input_dir $input_dir -output_dir $output_dir
