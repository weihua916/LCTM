#pragma once

#include "Matrix.h"
#include <vector>
#include "dataset.h"
#include "Utils.h"

class LDA_CVB0 {
public:
    //constructor of the model
    LDA_CVB0(int n_topics, int n_iter, double alpha, double beta, bool est_hyper, std::string& output_dir);
    
    //X is the count matrix with (d,v)th element the number of times vocab v appears in d
    void fit(const dataset& data);
    //D X K matrix
    void write_theta(const std::string& filename);
    
    //K X S matrix
    void write_phi(const std::string& filename);
    
    //void write_alpha(const std::string& filename);
    
    //void write_topic_assignment(const std::string& filename);
    //theta is a learned document-topic distribution
    MatD theta;
    //phi is a learned topic-synonym distribution
    MatD phi;
    
    int n_docs; //number of documents
    int n_vocab; //vocabulary size
    int n_topics; //number of topics
    int n_iter;//number of iterations
    double alpha;
    VecD alpha_vec;
    double beta;
    
    int hyper_iter;
    bool est_hyper;
    
    std::string output_dir;
    //double train_proportion;
    //std::vector<int> doc_train_until;
    
private:
    
    std::vector<std::vector<int> > doc_sentence;
    //std::vector<std::vector<int> > test_doc_sentence;
    //bool test_perplexity;
    
    //the following three will concern matrix calculation
    MatD ndk; //ndk[d][k] is a number of words in document d that are assigned to topic k, size: M X K
    MatD nkw; //nks[k][s] is a number of words in topic k that are assigned to topic s, size: K X S
    VecD nk; //nk[k] is a number of words that are assigned to topic k, size: K
    
    int sum_words;
    
    std::vector<std::vector<VecD> > topic_ratio; //topics[d][i] is n_topics dimensional std::vector representing the ratio of each topic
    
    void initialize();
    
    
    //Use collapsed gibbs sampler to do bayesian inference.
    void infer();
    
    //functions for sampling latent topic k
    void downdate_ratio_for_z(int d, int w, VecD& qz_old);
    void update_ratio_for_z(int d, int w, VecD& qz_new);
    VecD estimate_topic_ratio(int d, int w);
    
    //calculate the maximal likelihood estimate of theta and phi
    void calculate_theta();
    void calculate_phi();
    void show_info();
   // void check_sum();
    
    void update_hyper_param();
    
    //calculate the ave_loglikelihood while training
    //double ave_loglikelihood();
    //void perplexity(ofstream& fout_perplexity);
};