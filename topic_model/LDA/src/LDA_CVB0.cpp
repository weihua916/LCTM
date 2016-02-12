#include "LDA_CVB0.h"
#include <iostream>
#include "dataset.h"
#include "Matrix.h"
#include <vector>
#include "Utils.h"
#include <cassert>
#include <math.h>

//Constructor
LDA_CVB0::LDA_CVB0(int n_topics, int n_iter,
         double alpha, double beta, bool est_hyper, std::string& output_dir):
n_topics(n_topics), n_iter(n_iter), alpha(alpha), beta(beta), est_hyper(est_hyper), output_dir(output_dir){
}

void LDA_CVB0::fit(const dataset& data){
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
    //write_topic_assignment(output_dir + "topic_assignment.txt");
    //write_alpha(output_dir + "alpha.txt");
}


/*parameters
 X: count matrix
 */
void LDA_CVB0::initialize(){
    std::cout << "Start initializing..." << std::endl;
    //train_proportion = 0.8;
    //number of fixed point iterations for estimating the hyper parameters
    hyper_iter = 50;
    
    ndk = MatD::Zero(n_docs, n_topics);
    nkw = MatD::Zero(n_topics, n_vocab);
    nk = VecD::Zero(n_topics);
    alpha_vec = alpha * VecD::Ones(n_topics);
    
    sum_words = 0;
    srandom(time(0));
    
    for (int d=0; d<n_docs; d++){
        //calculate the length of d th document
        std::vector<int> sentence = doc_sentence[d];
        int doc_len = sentence.size();
        //int until;
        std::vector<VecD> doc_topic_ratio;
        //initialize q(z) for each word as an uniform distribution on topics
        for (int i=0; i<doc_len; i++){
            VecD rand_ratio = Utils::random_ratio(n_topics);
            doc_topic_ratio.push_back(rand_ratio);
            int w = sentence[i];
            ndk.row(d).array() += rand_ratio.array();
            nkw.col(w).array() += rand_ratio.array();
            nk.array() += rand_ratio.array();
            sum_words++;
        }
        topic_ratio.push_back(doc_topic_ratio);
    }
    std::cout << "finished initializing!" << std::endl;
}

//after the initialization, do inference (collapsed gibbs sampling)
void LDA_CVB0::infer(){
    std::cout << "start training..." << std::endl;
    std::vector<int> sentence;
    int w;
    VecD qz_old;
    VecD qz_new;
    
    double changed_ratio_z;
    for (int it=0; it<n_iter; it++){
        std::cout << it << "th iteration..." << std::endl;
        //check_sum();
        for (int d=0; d < n_docs; d++){
            sentence = doc_sentence[d];
            int doc_len = sentence.size();
            for (int i=0; i<doc_len; i++){
                w = sentence[i];
                qz_old = topic_ratio[d][i];
                downdate_ratio_for_z(d, w, qz_old);
                qz_new = estimate_topic_ratio(d, w);
                topic_ratio[d][i] = qz_new;
                update_ratio_for_z(d ,w ,qz_new);
            }
        }
        if (est_hyper && it > 20)
            update_hyper_param();
        if (it % 25 == 0){
            calculate_theta();
            calculate_phi();
            write_theta(output_dir + "theta.txt");
            write_phi(output_dir + "phi.txt");
        }
    }
    std::cout << "finished training!" << std::endl;
}

inline void LDA_CVB0::downdate_ratio_for_z(int d, int w, VecD& qz_old){
    ndk.row(d).array() -= qz_old.array();
    nkw.col(w).array() -= qz_old.array();
    nk.array() -= qz_old.array();
}

inline void LDA_CVB0::update_ratio_for_z(int d, int w, VecD& qz_new){
    ndk.row(d).array() += qz_new.array();
    nkw.col(w).array() += qz_new.array();
    nk.array() += qz_new.array();
}

VecD LDA_CVB0::estimate_topic_ratio(int d, int w){
    VecD left = (nkw.col(w).array() + beta)/(nk.array() + beta*(double)n_vocab);
    VecD nd_d = ndk.row(d);
    VecD right = nd_d.array() + alpha_vec.array();
    VecD prob = left.array()*right.array();
    prob /= prob.sum();
    return prob;
}

void LDA_CVB0::calculate_theta(){
    theta = ndk;
    for (int d=0; d<n_docs; d++){
        theta.row(d).array() += alpha_vec.array();
        theta.row(d) /= theta.row(d).sum();
    }
}

void LDA_CVB0::calculate_phi(){
    phi = nkw.array() + beta;
    for (int k=0; k<n_topics; k++){
        phi.row(k) /= phi.row(k).sum();
    }
}

void LDA_CVB0::write_theta(const std::string& filename){
    Utils::write_matrix(theta, filename);
}

void LDA_CVB0::write_phi(const std::string& filename){
    Utils::write_matrix(phi, filename);
}

void LDA_CVB0::update_hyper_param(){
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
        beta_new = nom/(denom*(double)n_vocab);
        alpha_vec = alpha_new;
        beta = beta_new;
    }
}

void LDA_CVB0::show_info(){
    std::cout << "LDA_CVB0:" << std::endl;
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

