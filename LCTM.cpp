#include "LCTM.h"
#include <iostream>
#include "dataset.h"
#include "Matrix.h"
#include <vector>
#include "Utils.h"
#include <cassert>
#include <math.h>
#include <fstream>

using namespace std;

//Constructor
LCTM::LCTM(int n_topics, int n_iter, double noise, string& output_dir):
    n_topics(n_topics), n_iter(n_iter), noise(noise), output_dir(output_dir){
}

void LCTM::fit(const dataset& data){
    n_docs = data.n_docs;
    n_vocab = data.n_vocab;
    n_concepts = data.n_concepts;
    wordvec_dim = data.wordvec_dim;
    wv = data.wordvectors;
    alpha = 0.1;
    beta = 0.01;
    
    burn_in = 999;
    near_num = 300;
    faster = true;
    
    doc_sentence = data.train_doc_sentence;
   
    initialize(data.init_concepts);
    show_info();
    infer();
}


/*parameters
 X: count matrix
 init_concepts: an array whose w th element is the the initial concept class assigned to that word
*/
void LCTM::initialize(const vector<int>& init_concepts){
    cout << "Start initializing..." << endl;
    hyper_iter = 100;
    consec_num = 100;
    
    ndk = MatI::Zero(n_docs, n_topics);
    nks = MatI::Zero(n_topics, n_concepts);
    nk = VecI::Zero(n_topics);
    alpha_vec = alpha * VecD::Ones(n_topics);
    
    ns = new int[n_concepts];
    for (int s=0; s<n_concepts; s++)
        ns[s] = 0;
    
    //count of each words.
    VecI nw = VecI::Zero(n_vocab);
    
    concepts = new int*[n_docs];
    topics = new int*[n_docs];
    sum_mu_s = MatD::Zero(wordvec_dim, n_concepts);
    
    srandom(time(0));
    
    for (int d=0; d<n_docs; d++){
        //calculate the length of d th document
        vector<int> sentence = doc_sentence[d];
        int doc_len = sentence.size();
        topics[d] = new int[doc_len];
        concepts[d] = new int[doc_len];
        
        for (int i=0; i<doc_len; i++){
            //initialize z with a random topic
            int topic = (int)(((double)random() / RAND_MAX) * n_topics);
            int w = sentence[i];
            int s = init_concepts[w];
            topics[d][i] = topic;
            concepts[d][i] = s;
            ns[s]++;
            nw(w)++;
            ndk(d,topic)++;
            nks(topic, s)++;
            nk(topic)++;
            sum_mu_s.col(s) += wv.col(w);
        }
    }
    
    //if faster is true, heuristic is used.
    if (faster){
        consec_sampled_num = new int*[n_docs];
        for (int d = 0; d<n_docs; d++){
            consec_sampled_num[d] = new int[doc_sentence[d].size()];
            for (int i=0; i<doc_sentence[d].size(); i++){
                consec_sampled_num[d][i] = 0;
            }
        }
    }
    
    //number of words in total
    sum_words = nw.sum();
    set_wordvector_priors(nw);
    
    mu_s = MatD::Zero(wordvec_dim, n_concepts);
    sigma_s = VecD::Zero(n_concepts);
    mu_s_dot_mu_s = VecD::Zero(n_concepts);
    
    for (int s=0; s<n_concepts; s++){
        Utils::mu_sigma_pair pair = calculate_mu_and_sigma(s);
        mu_s.col(s) = pair.mu;
        sigma_s(s) = pair.sigma;
        mu_s_dot_mu_s(s) = pair.mu.squaredNorm();
    }
    cout << "finished initializing!" << endl;
}

//set mu_prior and sigma_prior
void LCTM::set_wordvector_priors(const VecI& nw){
    VecD sum_vectors = wv * nw.cast<double>();
    mu_prior = sum_vectors / (double)sum_words;
    sigma_prior = 0.0;
    for (int w = 0; w<n_vocab; w++){
        VecD diff = wv.col(w) - mu_prior;
        sigma_prior += diff.squaredNorm()*(double)nw(w);
    }
    
    //MLE of sigma
    sigma_prior /= (double)(sum_words*wordvec_dim);
    
    //expand a little bit
    //sigma_prior *= 4.0;
    sigma_prior = 1.0;
    cout << "sigma_prior is " << sigma_prior << endl;
    //cout << "mu_prior is " << mu_prior << endl;
}

//after the initialization, do inference (collapsed gibbs sampling)
void LCTM::infer(){
    cout << "start training..." << endl;
    vector<int> sentence;
    int w;
    int z;
    int z_new;
    int s;
    int s_new;
    
    int count = 0;
    
    accum_theta = MatD::Zero(n_docs, n_topics);
    accum_phi = MatD::Zero(n_topics, n_concepts);
    accum_sigma = VecD::Zero(n_concepts);
    
    //numbder of s and z changed after sampling
    int num_z_changed;
    int num_s_changed;
    double changed_ratio_z;
    double changed_ratio_s;
    int num_ommit;
    double ommit_ratio;
    
    string header = to_string(n_topics) + "K_" + to_string(n_concepts) + "S_" + to_string(noise).substr(0,4) + "noise_";
    output_dir += header;
    
    cout << output_dir << endl;
    cout << header << endl;
    
    //ofstream fout(output_dir + "avll_log.txt", ios::app);
    //ofstream fout_perplexity(output_dir + "perplexity.txt", ios::app);
    for (int it=0; it<n_iter; it++){
        num_z_changed = 0;
        num_s_changed = 0;
        num_ommit = 0;
        
        if (it % 5 == 0)
            create_near_list();
        
        for (int d=0; d < n_docs; d++){
            //cout << d << endl;
            sentence = doc_sentence[d];
            //double time_for_z = 0.0;
            double time_for_sample_s = 0.0;
            for (int i=0; i<sentence.size(); i++){
                //cout << i << endl;
                w = sentence[i];
                z = topics[d][i];
                s = concepts[d][i];
                
                assert(s>=0 && s<n_concepts);
                assert(z>=0 && z<n_topics);
                assert(w>=0 && w<n_vocab);
                
                downdate_count_for_z(d, s, z);
                //clock_t start = clock();
                z_new = sample_z(d, s);
                //clock_t end = clock();
                topics[d][i] = z_new;
                update_count_for_z(d ,s ,z_new);
                
                if(z != z_new) num_z_changed++;
                
                //if the word meet the criterion, there is no need to evaluate s.
                
                //if (faster && consec_sampled_num[d][i] > consec_num && consec_sampled_num[d][i] < revive_num){
                if (faster && consec_sampled_num[d][i] > consec_num){
                    //consec_sampled_num[d][i]++;
                    num_ommit++;
                    continue;
                }
                //reset
                //if (faster && consec_sampled_num[d][i] >= revive_num)
                //consec_sampled_num[d][i] = 0;
                    
                downdate_param_for_s(w, s, z_new);
                
                s_new = sample_s_from_nearList(w, z_new);
                //clock_t end = clock();
                //time_for_sample_s += (double)(end - start) / CLOCKS_PER_SEC;
                
                concepts[d][i] = s_new;
                update_param_for_s(w, s_new, z_new);
                
                if(s != s_new){
                    num_s_changed++;
                    //reset
                    if (faster)
                        consec_sampled_num[d][i] = 0;
                } else {
                    if(faster)
                        consec_sampled_num[d][i]++;
                }
            }
            //cout << "time for sample s: " << time_for_sample_s/(double)doc_train_until[d] << endl;
        }
        changed_ratio_z = 100*(double)num_z_changed/(double)sum_words;
        changed_ratio_s = 100*(double)num_s_changed/(double)sum_words;
        ommit_ratio = 100*(double)num_ommit/(double)sum_words;
        
        if (it % 50 == 0){
            cout << it << "th iteration..." << endl;
            //cout << "z changed " << num_z_changed << " (" << changed_ratio_z << "%)"<< endl;
            //cout << "s changed " << num_s_changed << " (" << changed_ratio_s << "%)"<< endl;
            //cout << "ommit ratio " << num_ommit << " (" << ommit_ratio << "%)"<< endl;
        }
        if (it % 50 == 0 && it > burn_in){
            count++;
            calculate_theta();
            calculate_phi();
            
            accum_theta += theta;
            accum_phi += phi;
            accum_sigma += sigma_s;
            
            theta = accum_theta/static_cast<double>(count);
            phi = accum_phi/static_cast<double>(count);
            sigma_average = accum_sigma/static_cast<double>(count);

            
            write_mu_s(output_dir + "mu.txt");
            write_theta(output_dir + "theta.txt");
            write_phi(output_dir + "phi.txt");
            write_sigma_s(output_dir + "variance.txt");
        }
    }
    
    calculate_theta();
    calculate_phi();
    count++;
    accum_theta += theta;
    accum_phi += phi;
    accum_sigma += sigma_s;
    
    theta = accum_theta/static_cast<double>(count);
    phi = accum_phi/static_cast<double>(count);
    sigma_average = accum_sigma/static_cast<double>(count);
    
    write_theta(output_dir + "theta.txt");
    write_phi(output_dir + "phi.txt");
    write_mu_s(output_dir + "mu.txt");
    write_sigma_s(output_dir + "variance.txt");
    
    cout << "finished training!" << endl;
}

//from the curent count and sum statistics of concept class, calculate mu_s and sigma_s
Utils::mu_sigma_pair LCTM::calculate_mu_and_sigma(int s){
    Utils::mu_sigma_pair pair;
    double variance_inverse = ((double)ns[s])/noise + 1.0/sigma_prior;
    //cout << "noise" << noise << endl;
    //cout << "sigma_prior" << sigma_prior<< endl;
    pair.sigma = noise + (1.0/variance_inverse);
    
    //in order to prevent overflow, calculate pair.mu and pair.sigma separately
    double coef_1 = (double)ns[s] + noise/sigma_prior;
    double coef_2 = 1.0 + (double)ns[s]*(sigma_prior/noise);
    pair.mu = sum_mu_s.col(s)/coef_1 + mu_prior/coef_2;
    return pair;
}

void LCTM::downdate_param_for_s(int w, int s, int z){
    sum_mu_s.col(s) -= wv.col(w);
    ns[s]--;
    nks(z,s)--;
    Utils::mu_sigma_pair pair = calculate_mu_and_sigma(s);
    mu_s.col(s) = pair.mu;
    sigma_s(s) = pair.sigma;
    mu_s_dot_mu_s(s) = pair.mu.squaredNorm();
}

void LCTM::update_param_for_s(int w, int s, int z){
    sum_mu_s.col(s) += wv.col(w);
    ns[s]++;
    nks(z,s)++;
    Utils::mu_sigma_pair pair = calculate_mu_and_sigma(s);
    mu_s.col(s) = pair.mu;
    sigma_s(s) = pair.sigma;
    mu_s_dot_mu_s(s) = pair.mu.squaredNorm();
}

int LCTM::sample_s(int w, int z){
    VecD log_left = Utils::log_vec(nks.cast<double>().row(z).array() + beta);
    VecD term1 = - 0.5 * (double)wordvec_dim * Utils::log_vec(sigma_s);
    VecD term2 = -(0.5/sigma_s.array())*((mu_s_dot_mu_s - 2.0*mu_s.transpose()*wv.col(w)).array());
    VecD prob = Utils::soft_max(log_left + term1 + term2);
    return Utils::draw(prob);
}

int LCTM::sample_s_from_nearList(int w, int z){
    VecD log_left = VecD::Zero(near_num);
    VecD term1 = VecD::Zero(near_num);
    VecD term2 = VecD::Zero(near_num);
    for (int i = 0; i < near_num; i++){
        int s = near_list[w][i];
        log_left(i) = log(static_cast<double>(nks(z, s))+ beta);
        term1(i) = - 0.5 * (double)wordvec_dim *log(sigma_s(s));
        term2(i) = -(0.5/sigma_s(s))*(mu_s_dot_mu_s(s) - 2.0*mu_s.col(s).transpose()*wv.col(w));
    }
    VecD prob = Utils::soft_max(log_left + term1 + term2);
    return near_list[w][Utils::draw(prob)];
}

void LCTM::create_near_list(){
    near_list.clear();
    for (int w = 0; w < n_vocab; w++){
        VecD term1 = - 0.5 * (double)wordvec_dim * Utils::log_vec(sigma_s);
        VecD term2 = -(0.5/sigma_s.array())*((mu_s_dot_mu_s - 2.0*mu_s.transpose()*wv.col(w)).array());
        VecD log_right = term1 + term2;
        vector<size_t> sorted_index = Utils::ArgSort(log_right, std::greater<double>());
        vector<int> top_index;
        for (int i = 0; i < near_num; i++){
            top_index.push_back(static_cast<int>(sorted_index[i]));
        }
        near_list.push_back(top_index);
    }
}


inline void LCTM::downdate_count_for_z(int d, int s, int z){
    ndk(d,z)--;
    nks(z,s)--;
    nk[z]--;
}

inline void LCTM::update_count_for_z(int d, int s, int z){
    ndk(d,z)++;
    nks(z,s)++;
    nk[z]++;
}

int LCTM::sample_z(int d, int s){
    VecD left = (nks.col(s).cast<double>().array() + beta)/
        (nk.cast<double>().array() + beta*(double)n_concepts);
    VecD nd_d = ndk.row(d).cast<double>();
    VecD right = nd_d.array() + alpha_vec.array();
    VecD prob = left.array()*right.array();
    prob /= prob.sum();
    return Utils::draw(prob);
}

void LCTM::calculate_theta(){
    theta = ndk.cast<double>();
    for (int d=0; d<n_docs; d++){
        theta.row(d).array() += alpha_vec.array();
        theta.row(d) /= theta.row(d).sum();
    }
}

void LCTM::calculate_phi(){
    phi = nks.cast<double>().array() + beta;
    for (int k=0; k<n_topics; k++){
        phi.row(k) /= phi.row(k).sum();
    }
}

void LCTM::write_mu_s(const string& filename){
    Utils::write_matrix_transpose(mu_s, filename);
}

void LCTM::write_sigma_s(const string& filename){
    Utils::write_vector(sigma_average, filename);
}

void LCTM::write_theta(const string& filename){
    Utils::write_matrix(theta, filename);
}

void LCTM::write_phi(const string& filename){
    Utils::write_matrix(phi, filename);
}

void LCTM::write_topic_assignment(const string& filename){
    ofstream fw;
    fw.open(filename, ios::out);
    if (fw.fail()){
        cerr << "error occurred writing data to " << filename << endl;
        exit(-1);
    }
    for (int d=0; d< n_docs; d++){
        int doc_len = doc_sentence[d].size();
        for (int i=0; i< doc_len-1; i++){
            fw << topics[d][i] << " ";
            //cout << topics[d][i] << endl;
            //assert(topics[d][i] < n_topics);
        }
        fw << topics[d][doc_len-1] << endl;
    }
}

void LCTM::write_concept_assignment(const string& filename){
    ofstream fw;
    fw.open(filename, ios::out);
    if (fw.fail()){
        cerr << "error occurred writing data to " << filename << endl;
        exit(-1);
    }
    for (int d=0; d< n_docs; d++){
        int doc_len = doc_sentence[d].size();
        for (int i=0; i< doc_len-1; i++){
            fw << concepts[d][i] << " ";
            //assert(concepts[d][i] < n_concepts);
        }
        fw << concepts[d][doc_len-1] << endl;
    }
}

void LCTM::show_info(){
    cout << "LCTM:" << endl;
    if (faster){
        cout << "faster = true" << endl;
        cout << "consec num = " << consec_num << endl;
    } else {
        cout << "faster = false" << endl;
    }
    cout << "number of iterations: " << n_iter << endl;
    cout << "number of topics: " << n_topics << endl;
    cout << "number of concepts: " << n_concepts << endl;
    cout << "word vector dimension: " << wordvec_dim << endl;
    cout << "number of unique vocabulary: " <<  n_vocab << endl;
    cout << "number of documents: " << n_docs << endl;
}

