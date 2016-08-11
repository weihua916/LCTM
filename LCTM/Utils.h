#pragma once
#include <cassert>
#include "Matrix.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#define logPI 1.1447298858494
#define PI 3.1415926535
#define epsilon 0.0001
using namespace std;

namespace Utils{
    
    typedef struct{
        double sigma;
        VecD mu;
    } mu_sigma_pair;
    
    inline VecD log_vec(const VecD& vec){
        return vec.array().log();
    }
    
    inline VecD soft_max(VecD vec){
        //cout << vec << endl;
        //cout << "--" << endl;
        
        //vec.array() -= vec.maxCoeff();
        //vec = vec.array().exp();
        
        vec = (vec.array() - vec.maxCoeff()).exp();
        
        //VecD v = vec/vec.sum();
        //double s = v.sum();
        //assert((1.0 - epsilon <s) && (s< 1.0 + epsilon));
        return vec/vec.sum();
    }
    
    inline int argmin(VecD& vec){
        int arg = 0;
        for (int i = 1; i < vec.size(); i++){
            if (vec(arg) > vec(i))
                arg = i;
        }
        return arg;
    }
    
    inline int argmax(VecD& vec){
        int arg = 0;
        for (int i = 1; i < vec.size(); i++){
            if (vec(arg) < vec(i))
                arg = i;
        }
        return arg;
    }
    
    inline int draw(const VecD& prob){
        double s = prob.sum();
        assert((1.0 - epsilon <s) && (s< 1.0 + epsilon));
        double r = ((double)random()+1.0) / ((double)RAND_MAX + 2.0);
        int i = 0;
        double accum = 0.0;
        while (accum < r && i < prob.size()){
            accum += prob(i);
            i++;
        }
        return i-1;
    }
    
    inline double gaussian_log_pdf(const VecD& v, const VecD& mu, double sigma){
        int dim = v.size();
        double coef = -0.5*(double)dim*(logPI + log(sigma));
        VecD dif = v.array() - mu.array();
        double inside = -(0.5/sigma) * dif.squaredNorm();
        return coef + inside;
    }
    
    inline vector<int> split_and_stoi(const string& str, char delim){
        vector<int> res;
        int current = 0, found;
        while((found = str.find_first_of(delim, current)) != string::npos){
            int num = stoi(string(str, current, found - current));
            res.push_back(num);
            current = found + 1;
        }
        int num = stoi(string(str, current, str.size() - current));
        res.push_back(num);
        return res;
    }
    
    inline vector<double> split_and_stod(const string& str, char delim){
        vector<double> res;
        int current = 0, found;
        while((found = str.find_first_of(delim, current)) != string::npos){
            double num = stod(string(str, current, found - current));
            res.push_back(num);
            current = found + 1;
        }
        double num = stod(string(str, current, str.size() - current));
        res.push_back(num);
        return res;
    }
    
    inline void write_matrix_transpose(const MatD& mat, const string& filename){
        ofstream fw;
        fw.open(filename, ios::out);
        if (fw.fail()){
            cerr << "error occurred writing data to " << filename << endl;
            exit(-1);
        }
        for (int i=0; i< mat.cols(); i++){
            VecD column = mat.col(i);
            for (int j=0; j< column.size()-1; j++){
                fw << column(j) << " ";
            }
            fw << column(column.size()-1) << endl;
        }
    }
    
    inline void write_matrix(const MatD& mat, const string& filename){
        ofstream fw;
        fw.open(filename, ios::out);
        if (fw.fail()){
            cerr << "error occurred writing data to " << filename << endl;
            exit(-1);
        }
        for (int i=0; i< mat.rows(); i++){
            VecD row = mat.row(i);
            for (int j=0; j< row.size()-1; j++){
                fw << row(j) << " ";
            }
            fw << row(row.size()-1) << endl;
        }
    }
    
    inline void write_int_matrix(const MatI& mat, const string& filename){
        ofstream fw;
        fw.open(filename, ios::out);
        if (fw.fail()){
            cerr << "error occurred writing data to " << filename << endl;
            exit(-1);
        }
        for (int i=0; i< mat.rows(); i++){
            VecI row = mat.row(i);
            for (int j=0; j< row.size()-1; j++){
                fw << row(j) << " ";
            }
            fw << row(row.size()-1) << endl;
        }
    }
    
    inline void write_vector(const VecD& vec, const string& filename){
        ofstream fw;
        fw.open(filename, ios::out);
        if (fw.fail()){
            cerr << "error occurred writing data to " << filename << endl;
            exit(-1);
        }
        for (int i=0; i<vec.size(); i++)
            fw << vec(i) << endl;
    }
    
    inline double digamma(double x) {
        double result = 0, xx, xx2, xx4;
        assert(x > 0);
        for ( ; x < 7; ++x)
            result -= 1/x;
        x -= 1.0/2.0;
        xx = 1.0/x;
        xx2 = xx*xx;
        xx4 = xx2*xx2;
        result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
        return result;
    }
    
    inline double log_T_distribution(double x, double nu, double mu, double variance){
        double normalized_constant = lgamma(0.5*nu + 0.5) - lgamma(0.5*nu) - 0.5*(log(nu) + logPI + log(variance));
        double density = -0.5*(nu + 1) * log(1 +  (x-mu)*(x-mu)/(variance*nu));
        return normalized_constant + density;
    }
    
    inline double diag_T_logpdf(const VecD& wv, double nu, const VecD& mu, const VecD& variance, double log_variance){
        int dim = wv.size();
        if (nu < 100){
            double n_const = static_cast<double>(dim)*(lgamma(0.5*nu + 0.5) - lgamma(0.5*nu) - 0.5*(log(nu) + logPI)) - 0.5*variance.array().log().sum();
            VecD diff = wv.array() - mu.array();
            double density = -0.5*(nu+1)*(1 + diff.array()*diff.array()/(variance.array()*nu)).array().log().sum();
            return  density + n_const;
        }else{
            VecD diff = wv.array() - mu.array();
            double density = -0.5*(diff.array()*diff.array()/variance.array()).sum();
            double n_const = -0.5*(double)dim * log(2*PI) - 0.5*log_variance;
            return density + n_const;
        }
    }
    
    inline VecD diag_T_logpdf_list(const VecD& wv, VecD& nu_list, const MatD& mu_list, const MatD& variance_list){
        int dim = wv.size();
        MatD mat = mu_list.colwise() - wv;
        mat = mat.array()*mat.array();
        mat.array() /= variance_list.array();
        
        for (int s = 0; s < nu_list.size(); s++)
            mat.col(s).array() /= nu_list(s);
        
        mat.array() += 1;
        mat = mat.array().log();
        VecD density = mat.colwise().sum();
        density.array() *= -0.5*(nu_list.array() + 1);
        
        VecD n_const = VecD::Zero(nu_list.size());
        for (int s = 0; s < nu_list.size(); s++)
            n_const(s) = (double)dim*(lgamma(0.5*nu_list(s) + 0.5) - lgamma(0.5*nu_list(s)) - 0.5*logPI);
        n_const.array() -= 0.5*(double)dim*(nu_list.array().log());
        n_const.array() -= 0.5*variance_list.array().log().colwise().sum();
        return density + n_const;
    }
    
    template <typename Sequence, typename BinaryPredicate>
    struct IndexCompareT {
        IndexCompareT(const Sequence& seq, const BinaryPredicate comp)
        : seq_(seq), comp_(comp) { }
        bool operator()(const size_t a, const size_t b) const
        {
            return comp_(seq_[a], seq_[b]);
        }
        const Sequence seq_;
        const BinaryPredicate comp_;
    };
    
    template <typename Sequence, typename BinaryPredicate>
    IndexCompareT<Sequence, BinaryPredicate>
    inline IndexCompare(const Sequence& seq, const BinaryPredicate comp)
    {
        return IndexCompareT<Sequence, BinaryPredicate>(seq, comp);
    }
    
    template <typename Sequence, typename BinaryPredicate>
    inline std::vector<size_t> ArgSort(const Sequence& seq, BinaryPredicate func)
    {
        std::vector<size_t> index(seq.size());
        for (int i = 0; i < index.size(); i++)
            index[i] = i;
        
        std::sort(index.begin(), index.end(), IndexCompare(seq, func));
        
        return index;
    }
    
    inline void proc_args(int argc, char** argv, int& n_topics, int& n_iter, double& noise, string& wv_file, string& doc_sentence_file, string& concept_file, string& input_dir, string& output_dir){
        
        for (int i = 1; i < argc; i+=2){
            string arg = (string)argv[i];
            
            if (arg == "-help"){
                cout << "### Options ###" << endl;
                cout << "-n_topics    the number of topics (default: 20)" << endl;
                cout << "-n_iter    the number of iterations (default: 1500)" << endl;
                cout << "-noise     the noise parameter of gaussian distribution" << endl;
                cout << "-wv_file   the file of wordvectors, each row contains one word vector." << endl;
                cout << "-doc_file  the document file, each line correpsponding to one document." << endl;
                cout << "-concept_file  the concept file, each row contains the concept class." << endl;
                cout << "-input_dir  the path to the output directory. (default: input/)" << endl;
                cout << "-output_dir  the path to the output directory. (default: output/)" << endl;
                exit(1);
            }
            else if (arg == "-n_topics"){
                assert(i+1 < argc);
                n_topics = atoi(argv[i+1]);
                assert(n_topics > 1);
            }
            else if (arg == "-n_iter"){
                assert(i+1 < argc);
                n_iter = atoi(argv[i+1]);
                assert(n_iter > 0);
            }
            else if (arg == "-noise"){
                assert(i+1 < argc);
                noise = atof(argv[i+1]);
                assert(n_iter > 0);
            }
            else if (arg == "-wv_file"){
                assert(i+1 < argc);
                wv_file = (string)argv[i+1];
            }
            else if (arg == "-doc_file"){
                assert(i+1 < argc);
                doc_sentence_file = (string)argv[i+1];
            }
            else if (arg == "-concept_file"){
                assert(i+1 < argc);
                concept_file = (string)argv[i+1];
            }
            else if (arg == "-input_dir"){
                assert(i+1 < argc);
                input_dir = (string)argv[i+1];
            }
            else if (arg == "-output_dir"){
                assert(i+1 < argc);
                output_dir = (string)argv[i+1];
            }
        }
    }

};
