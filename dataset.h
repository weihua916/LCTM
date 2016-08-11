#pragma once

#include "Matrix.h"
#include <vector>
#include <string>

using namespace std;
using namespace Eigen;

class dataset {
    
public:
    //constructor
    dataset();
    
    int n_concepts;
    int n_docs;
    int n_vocab;
    int wordvec_dim;
    
    bool has_test; //whether the dataset contains test data or not.
    
    vector<int> init_concepts; //an array of length n_vocab
    vector<vector<int> > train_doc_sentence;
    vector<vector<int> > test_doc_sentence;
    MatD wordvectors; // D X V matrix, each column is a word vector.
    MatD init_concept_vecs;
    
    /*
     Format of wordvector file:
     [n_vocab] [wordvec_dim]
     [w1] [w2] [w3] ...
     [w1] [w2] [w3] ...
     
     each row corresponds to a word vector.
    */
    void load_wordvectors(const string& wv_file);
    
    /*
    Format of doc_sentence_file:
     [id] [id] [id] ...
     [id] [id] [id] ...
     [id] [id] [id] ...
     
     each line corresponds to word ids.
    */
    void load_train_doc_sentence(const string& doc_sentence_file);
    
    /*
    Format of init_concept_file:
     [concept_id]
     [concept_id]
     [concept_id]
     [concept_id]
     .
     .
     .
     
     each line contains concept identification number of corresponding words.
     */
    void load_init_concepts(const string& concept_file);
};