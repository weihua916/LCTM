#pragma once
#include <cassert>
#include "Matrix.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

const double logPI = 1.1447298858494;
const double epsilon = 0.0001;

namespace Utils{
    inline VecD log_vec(const VecD& vec){
        return vec.array().log();
    }
    
    inline VecD soft_max(VecD vec){
        vec = (vec.array() - vec.maxCoeff()).exp();
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
        double r = (static_cast<double>(random())+1.0) / (static_cast<double>(RAND_MAX) + 2.0);
        int i = 0;
        double accum = 0.0;
        while (accum < r && i < prob.size()){
            accum += prob(i);
            i++;
        }
        return i-1;
    }
    
    inline VecD random_ratio(int dim){
        VecD ratio = VecD::Zero(dim);
        for (int i = 0; i < dim; i++){
            ratio(i) = (static_cast<double>(random())+1.0) / (static_cast<double>(RAND_MAX) + 2.0);
        }
        return ratio/ratio.sum();
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
    
    inline std::vector<int> split_and_stoi(const std::string& str, char delim){
        std::vector<int> res;
        int current = 0, found;
        while((found = str.find_first_of(delim, current)) != std::string::npos){
            int num = std::stoi(std::string(str, current, found - current));
            res.push_back(num);
            current = found + 1;
        }
        int num = std::stoi(std::string(str, current, str.size() - current));
        res.push_back(num);
        return res;
    }
    
    inline std::vector<double> split_and_stod(const std::string& str, char delim){
        std::vector<double> res;
        int current = 0, found;
        while((found = str.find_first_of(delim, current)) != std::string::npos){
            double num = std::stod(std::string(str, current, found - current));
            res.push_back(num);
            current = found + 1;
        }
        double num = std::stod(std::string(str, current, str.size() - current));
        res.push_back(num);
        return res;
    }
    
    inline void write_matrix_transpose(const MatD& mat, const std::string& filename){
        std::ofstream fw;
        fw.open(filename, std::ios::out);
        if (fw.fail()){
            std::cerr << "error occurred writing data to " << filename << std::endl;
            std::exit(-1);
        }
        /*
        for (int i=0; i< mat.cols(); i++){
            VecD column = mat.col(i);
            for (int j=0; j< column.size()-1; j++){
                fw << column(j) << " ";
            }
            fw << column(column.size()-1) << std::endl;
        }*/
        fw << mat.transpose() << std::endl;
    }
    
    inline void write_matrix(const MatD& mat, const std::string& filename){
        std::ofstream fw;
        fw.open(filename, std::ios::out);
        if (fw.fail()){
            std::cerr << "error occurred writing data to " << filename << std::endl;
            std::exit(-1);
        }
        /*
        for (int i=0; i< mat.rows(); i++){
            VecD row = mat.row(i);
            for (int j=0; j< row.size()-1; j++){
                fw << row(j) << " ";
            }
            fw << row(row.size()-1) << std::endl;
        }*/
        fw << mat << std::endl;
    }
    
    inline void write_int_matrix(const MatI& mat, const std::string& filename){
        std::ofstream fw;
        fw.open(filename, std::ios::out);
        if (fw.fail()){
            std::cerr << "error occurred writing data to " << filename << std::endl;
            std::exit(-1);
        }
        /*
        for (int i=0; i< mat.rows(); i++){
            VecI row = mat.row(i);
            for (int j=0; j< row.size()-1; j++){
                fw << row(j) << " ";
            }
            fw << row(row.size()-1) << std::endl;
        }*/
        fw << mat << std::endl;
    }
    
    inline void write_vector(const VecD& vec, const std::string& filename){
        std::ofstream fw;
        fw.open(filename, std::ios::out);
        if (fw.fail()){
            std::cerr << "error occurred writing data to " << filename << std::endl;
            std::exit(-1);
        }
        /*
        for (int i=0; i<vec.size(); i++)
            fw << vec(i) << std::endl;
         */
        fw << vec << std::endl;
    }
    
    inline void proc_args(int argc, char** argv, std::string& algo, bool& est_hyper, int& n_topics, int& n_iter, std::string& doc_sentence_file, std::string& input_dir, std::string& output_dir){
        
        for (int i = 1; i < argc; i+=2){
            std::string arg = static_cast<std::string>(argv[i]);
            
            if (arg == "-help"){
                std::cout << "### Options ###" << std::endl;
                std::cout << "-algo    the algorithm for the inference (CGS or CVB0)" << std::endl;
                std::cout << "-est_hyper   whether to estimate hyper parameter or not. 1 or 0" << std::endl;
                std::cout << "-n_topics    the number of topics (default: 20)" << std::endl;
                std::cout << "-n_iter    the number of iterations (default: 1500)" << std::endl;
                std::cout << "-doc_sentence_file  the document file, each line correpsponding to one document." << std::endl;
                std::cout << "-input_dir  the path to the output directory. (default: output/)" << std::endl;
                std::cout << "-output_dir  the path to the output directory. (default: input/)" << std::endl;
                exit(1);
            }
            else if (arg == "-n_topics"){
                assert(i+1 < argc);
                n_topics = std::atoi(argv[i+1]);
                assert(n_topics > 1);
            }
            else if (arg == "-est_hyper"){
                assert(i+1 < argc);
                est_hyper = static_cast<bool>(std::atoi(argv[i+1]));
                assert(std::atoi(argv[i+1])==0 || std::atoi(argv[i+1])==1);
            }
            else if (arg == "-n_iter"){
                assert(i+1 < argc);
                n_iter = std::atoi(argv[i+1]);
                assert(n_iter > 0);
            }
            else if (arg == "-doc_sentence_file"){
                assert(i+1 < argc);
                doc_sentence_file = static_cast<std::string>(argv[i+1]);
            }
            else if (arg == "-algo"){
                assert(i+1 < argc);
                algo = static_cast<std::string>(argv[i+1]);
                assert(algo == "CGS" || algo == "CVB0");
            }
            else if (arg == "-input_dir"){
                assert(i+1 < argc);
                input_dir = static_cast<std::string>(argv[i+1]);
            }
            else if (arg == "-output_dir"){
                assert(i+1 < argc);
                output_dir = static_cast<std::string>(argv[i+1]);
            }
        }
    }

};
