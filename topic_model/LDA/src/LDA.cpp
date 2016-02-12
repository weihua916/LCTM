#include "LDA.h"
#include <iostream>
#include "dataset.h"
#include "Matrix.h"
#include <vector>
#include "Utils.h"
#include <cassert>
#include <math.h>
#include <fstream>

//Constructor
//est_hyper: if true, then estimate hyper paramters!
LDA::LDA(int n_topics, int n_iter,
                       double alpha, double beta, bool est_hyper, std::string& output_dir):
    n_topics(n_topics), n_iter(n_iter), alpha(alpha), beta(beta), est_hyper(est_hyper) ,output_dir(output_dir){
}

void LDA::fit(const dataset& data){
    n_docs = data.n_docs;
    n_vocab = data.n_vocab;
    
    doc_sentence = data.doc_sentence;

    initialize();
    show_info();
    infer();
    calculate_theta();
    calculate_phi();
    write_theta(output_dir + "theta.txt");
    write_phi(output_dir + "phi.txt");
}


/*parameters
 X: count matrix
*/
void LDA::initialize(){
    std::cout << "Start initializing..." << std::endl;
    //train_proportion = 0.8;
    
    hyper_iter = 100;
    
    ndk = MatI::Zero(n_docs, n_topics);
    nkw = MatI::Zero(n_topics, n_vocab);
    nk = VecI::Zero(n_topics);
    alpha_vec = alpha * VecD::Ones(n_topics);

    topics = new int*[n_docs];
    sum_words = 0;
    srandom(time(0));
    
    for (int d=0; d<n_docs; d++){
        //calculate the length of d th document
        std::vector<int> sentence = doc_sentence[d];
        int doc_len = sentence.size();
        topics[d] = new int[doc_len];
        
        for (int i=0; i<doc_len; i++){
            //initialize z with a random topic
            double r = (static_cast<double>(random())+1.0) / (static_cast<double>(RAND_MAX) + 2.0);
            int topic = static_cast<int>(floor(r * n_topics));
            int w = sentence[i];
            topics[d][i] = topic;
            ndk(d,topic)++;
            nkw(topic, w)++;
            nk(topic)++;
            sum_words++;
        }
    }
    std::cout << "finished initializing!" << std::endl;
}

//after the initialization, do inference (collapsed gibbs sampling)
void LDA::infer(){
    std::cout << "start training..." << std::endl;
    std::vector<int> sentence;
    int w;
    int z;
    int z_new;
    int doc_len;
    for (int it=0; it<n_iter; it++){
        std::cout << it << "th iteration..." << std::endl;
        //check_sum();
        for (int d=0; d < n_docs; d++){
            sentence = doc_sentence[d];
            doc_len = sentence.size();
            for (int i=0; i<doc_len; i++){
                w = sentence[i];
                z = topics[d][i];
                downdate_count_for_z(d, w, z);
                z_new = sample_z(d, w);
                topics[d][i] = z_new;
                update_count_for_z(d ,w ,z_new);
            }
        }

        if (est_hyper && it > 20)
            update_hyper_param();
        
        if (it % 10 == 0)
            std::cout << "average log likelihood: " << ave_loglikelihood() << std::endl;
        
        if (it % 25 == 0){
            calculate_theta();
            calculate_phi();
            write_theta(output_dir + "theta.txt");
            write_phi(output_dir + "phi.txt");
        }
    }
    std::cout << "finished training!" << std::endl;
}

inline void LDA::downdate_count_for_z(int d, int w, int z){
    ndk(d,z)--;
    nkw(z,w)--;
    nk[z]--;
}

inline void LDA::update_count_for_z(int d, int w, int z){
    ndk(d,z)++;
    nkw(z,w)++;
    nk[z]++;
}

int LDA::sample_z(int d, int w){
    VecD left = (nkw.col(w).cast<double>().array() + beta)/
        (nk.cast<double>().array() + beta*static_cast<double>(n_vocab));
    VecD nd_d = ndk.row(d).cast<double>();
    VecD right = nd_d.array() + alpha_vec.array();
    VecD prob = left.array()*right.array();
    prob /= prob.sum();
    return Utils::draw(prob);
}

double LDA::ave_loglikelihood(){
    calculate_theta();
    calculate_phi();
    double ll = 0.0;
    int num_words = 0;
    for (int d=0; d < n_docs; d++){
        std::vector<int> sentence = doc_sentence[d];
        int doc_len = sentence.size();
        for (int i=0; i<doc_len; i++){
            int w = sentence[i];
            int k = topics[d][i];
            ll += log(theta(d,k)) + log(phi(k,w));
            num_words++;
        }
    }
    return ll/(double)num_words;
}

void LDA::calculate_theta(){
    theta = ndk.cast<double>();
    for (int d=0; d<n_docs; d++){
        theta.row(d).array() += alpha_vec.array();
        theta.row(d) /= theta.row(d).sum();
    }
}

void LDA::calculate_phi(){
    phi = nkw.cast<double>().array() + beta;
    for (int k=0; k<n_topics; k++){
        phi.row(k) /= phi.row(k).sum();
    }
}

void LDA::write_theta(const std::string& filename){
    Utils::write_matrix(theta, filename);
}

void LDA::write_phi(const std::string& filename){
    Utils::write_matrix(phi, filename);
}

void LDA::update_hyper_param(){
    for (int it = 0; it < hyper_iter; it++){
        VecD alpha_new = VecD::Zero(n_topics);
        double beta_new;
        for (int k = 0; k < n_topics; k++){
            double denom = 0;
            double nom = 0;
            assert(ndk.col(k).sum()>0);
            for (int d = 0; d < n_docs; d++){
                denom += Utils::digamma(static_cast<double>(ndk.row(d).sum()) + alpha_vec.sum()) - Utils::digamma(alpha_vec.sum());
                nom += (Utils::digamma(static_cast<double>(ndk(d,k)) + alpha_vec(k)) - Utils::digamma(alpha_vec(k)))*alpha_vec(k);
            }
            alpha_new(k) = nom/denom;
        }
        double denom = 0;
        double nom = 0;
        for (int k = 0; k < n_topics; k++){
            denom += Utils::digamma(static_cast<double>(nkw.row(k).sum()) + beta*static_cast<double>(n_vocab)) - Utils::digamma(beta*static_cast<double>(n_vocab));
            for (int v = 0; v < n_vocab; v++)
                nom += Utils::digamma(static_cast<double>(nkw(k,v)) + beta) - Utils::digamma(beta);
        }
        nom *= beta;
        beta_new = nom/(denom*static_cast<double>(n_vocab));
        alpha_vec = alpha_new;
        beta = beta_new;
    }
}

void LDA::show_info(){
    std::cout << "LDA_CGS:" << std::endl;
    if (est_hyper){
        std::cout << "hyper parameter inference = true" << std::endl;
    } else {
        std::cout << "hyper parameter inference = false" << std::endl;
    }
    std::cout << "number of iterations: " << n_iter << std::endl;
    std::cout << "number of topics: " << n_topics << std::endl;
    std::cout << "number of unique vocabulary: " <<  n_vocab << std::endl;
    std::cout << "number of documents: " << n_docs << std::endl;
}

