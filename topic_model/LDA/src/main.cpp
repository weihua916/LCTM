#include "LDA.h"
#include "LDA_CVB0.h"
#include "Utils.h"
#include "dataset.h"

int main(int argc, char* argv[]){
    
    //initialize with default values
    int n_topics = 20;
    int n_iter = 1500;
    std::string doc_sentence_file = "doc.txt";
    std::string input_dir = "input/";
    std::string output_dir = "output/";
    std::string algo = "";
    bool est_hyper;
    
    Utils::proc_args(argc, argv, algo, est_hyper, n_topics, n_iter, doc_sentence_file, input_dir, output_dir);
    dataset data;
    data.load_doc_sentence(input_dir + doc_sentence_file);
    
    //hyper parameter
    //concentration parameters of dirichlet priors
    double alpha = 0.1;
    double beta = 0.01;
    //if the inference algorithm is collapsed gibbs sampling
    if (algo == "CGS"){
        LDA model = LDA(n_topics, n_iter, alpha, beta, est_hyper, output_dir);
        model.fit(data);
    } else {
    //if the inference algorithm is collapsed variational bayes
        LDA_CVB0 model = LDA_CVB0(n_topics, n_iter, alpha, beta, est_hyper, output_dir);
        model.fit(data);
    }
    return 0;
}