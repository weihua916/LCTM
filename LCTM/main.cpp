#include "LCTM.h"
#include "Utils.h"
#include "dataset.h"

int main(int argc, char* argv[]){
    
    //initialize with default values
    int n_topics = 50;
    int n_iter = 1500;
    string wv_file = "wv.txt";
    string doc_sentence_file = "doc.txt";
    string concept_file = "concept.txt";
    string input_dir = "input/";
    string output_dir = "output/";
    double noise = 0.5;
    
    Utils::proc_args(argc, argv, n_topics, n_iter, noise, wv_file, doc_sentence_file, concept_file, input_dir, output_dir);
    dataset data;
    data.load_wordvectors(input_dir + wv_file);
    data.load_train_doc_sentence(input_dir + doc_sentence_file);
    data.load_init_concepts(input_dir + concept_file);

    //hyper parameter
    //concentration parameters of dirichlet priors.
    double alpha = 0.1;
    double beta = 0.01;
    LCTM model = LCTM(n_topics, n_iter, noise, output_dir);
    model.fit(data);
    return 0;
}
