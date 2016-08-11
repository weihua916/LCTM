#pragma once

#include "Matrix.h"
#include <vector>
#include "dataset.h"
#include "Utils.h"

using namespace std;

class LCTM {
public:
    //constructor of the model
    LCTM(int n_topics, int n_iter, double noise, string& output_dir);
    
    //X is the count matrix with (d,v)th element the number of times vocab v appears in d
    void fit(const dataset& data);
    
    //each row contains learned concept vector
    void write_mu_s(const string& filename);
    
    //each row contains learned concept noise
    void write_sigma_s(const string& filename);
    
    //D X K matrix
    void write_theta(const string& filename);
    
    //K X S matrix
    void write_phi(const string& filename);
    
    void write_topic_assignment(const string& filename);
    void write_concept_assignment(const string& filename);
    
    //theta is a learned document-topic distribution
    MatD theta;
    //MatD theta_test;
    //MatD accum_theta_test;
    //phi is a learned topic-concept distribution
    MatD phi;
    
    MatD accum_theta;
    MatD accum_phi;
    VecD accum_sigma;
    VecD sigma_average;
    int burn_in;
    
    int n_docs; //number of documents
    int n_vocab; //vocabulary size
    int n_concepts; //concept size
    int n_topics; //number of topics
    int n_iter;//number of iterations
    int wordvec_dim;
    double alpha;
    VecD alpha_vec;
    double beta;
    
    bool faster; //whether to use the heuristic that consequtive sampling of 5 times the same concept
    // allow us to avoid resampling again.
    int consec_num;
    int test_iteration;
    int revive_num;
    
    //if this is true, then we estimate the hyper parameters alpha and beta.
    bool est_hyper;
    int hyper_iter;
    
    string output_dir;
    
    //this parameter is initialized by the result of kmeans
    //actulally it can also be initialized using human cognition,
    //(i,e we can beforehand judge to what extent of noise we are able to call concepts.)
    double noise;
    
    vector<vector<int> > near_list;
    int near_num;
    
    //this parameter is initialized according to the actual input word vectors
    VecD mu_prior;//prior mean for concept vectors
    
    double sigma_prior;
    
    vector<vector<int> > doc_sentence;
    
    MatD wv; //word vectors, each columns contains a word vector: D X V
    
    MatD mu_s; //concept vectors: D X S
    
    MatD sum_mu_s; //sum of word vectors assigned to concept class s: D X S
    
    VecD mu_s_dot_mu_s; //a vector with s th element being an inner product of mu_s[s] and mu_s[s]
    
    VecD sigma_s; //a vector with s th element being a variance of concept s
    //Here, variance is the sum of posterior uncertainty of mu and noise parameter.
    //Usually, posterior uncertainty is quite small because ns[s] is usually very large.
    
    //the following three will concern matrix calculation
    MatI ndk; //ndk[d][k] is a number of words in document d that are assigned to topic k, size: M X K
    MatI nks; //nks[k][s] is a number of words in topic k that are assigned to topic s, size: K X S
    VecI nk; //nk[k] is a number of words that are assigned to topic k, size: K
    
    MatI ndk_test;
    
    int sum_words;
    
    int* ns; //ns[s] is a number of words that are assigned to topic s, size: S
    int** concepts; //concepts[d][i] is the latent concept class assigned to word (d,i)
    int** topics; //topics[d][i] is the latent topic assigned to word (d,i)
    int ** consec_sampled_num;
    
    //initialize counts and latent variables for the inference.
    //latent concepts are initialized before using kmeans algorithm
    void initialize(const vector<int>& init_concepts);
    

    void set_wordvector_priors(const VecI& nw);
    
    //Use collapsed gibbs sampler to do bayesian inference.
    void infer();
    
    //functions for sampling latent concept s
    Utils::mu_sigma_pair calculate_mu_and_sigma(int s);
    void downdate_param_for_s(int w, int s, int z);
    void update_param_for_s(int w, int s, int z);
    int sample_s(int w, int z);
    int sample_s_from_nearList(int w, int z);
    
    void create_near_list();
    
    //functions for sampling latent topic k
    void downdate_count_for_z(int d, int s, int z);
    void update_count_for_z(int d, int s, int z);
    int sample_z(int d, int s);
    
    //calculate the maximal likelihood estimate of theta and phi
    void calculate_theta();
    void calculate_phi();
    void show_info();
};
