#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "Utils.h"
#include "dataset.h"

using namespace std;

//constructor
dataset::dataset(){
    has_test = false;
}

void dataset::load_wordvectors(const string& wv_file){
    ifstream ifs(wv_file, ios::in);
    string str;
    if (ifs.fail()){
        cerr << "wordvector file not found." << endl;
        exit(-1);
    }
    getline(ifs, str);
    vector<int> res = Utils::split_and_stoi(str,' ');
    
    if(res.size()!=2){
        cerr << "file format is incorrect." << endl;
        exit(-1);
    }
    n_vocab = res[0];
    wordvec_dim = res[1];
    
    wordvectors = MatD::Zero(wordvec_dim, n_vocab);
    
    int i = 0;
    while (getline(ifs, str)){
        vector<double> vec = Utils::split_and_stod(str,' ');
        if (vec.size() != wordvec_dim){
            cerr << "vector size is incorrect." << endl;
            exit(-1);
        }
        wordvectors.col(i++) = Map<VecD>(&vec[0],wordvec_dim);
    }
}

void dataset::load_train_doc_sentence(const string& doc_sentence_file){
    ifstream ifs(doc_sentence_file,ios::in);
    string str;
    if (ifs.fail()){
        cout << doc_sentence_file << endl;
        cerr << "train doc_sentence file not found." << endl;
        exit(-1);
    }
    while (getline(ifs, str)){
        vector<int> sentence = Utils::split_and_stoi(str, ' ');
        train_doc_sentence.push_back(sentence);
    }
    n_docs = train_doc_sentence.size();
}

void dataset::load_init_concepts(const string& concept_file){
    ifstream ifs(concept_file,ios::in);
    string str;
    if (ifs.fail()){
        cerr << "concept_init file not found." << endl;
        exit(-1);
    }
    while (getline(ifs, str)){
        init_concepts.push_back(stoi(str));
    }
    n_concepts = *max_element(init_concepts.begin(),init_concepts.end()) + 1;
}




