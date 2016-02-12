#pragma once

#include "Matrix.h"
#include <vector>
#include <string>

class dataset {
public:
    int n_docs;
    int n_vocab;

    std::vector<std::vector<int> > doc_sentence;
    
    /*
    Format of doc_sentence_file:
     [num_docs] [num_vocab]
     [id] [id] [id] ...
     [id] [id] [id] ...
     [id] [id] [id] ...
     
     each line corresponds to word ids.
    */
    void load_doc_sentence(const std::string& doc_sentence_file);
};