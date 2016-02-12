#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "Utils.h"
#include "dataset.h"

void dataset::load_doc_sentence(const std::string& doc_sentence_file){
    std::ifstream ifs(doc_sentence_file,std::ios::in);
    std::string str;
    if (ifs.fail()){
        std::cout << doc_sentence_file << std::endl;
        std::cerr << "train doc_sentence file not found." << std::endl;
        exit(-1);
    }
    
    //parse the first line
    //[num docs] [num vocab]
    std::getline(ifs, str);
    std::vector<int> info = Utils::split_and_stoi(str, ' ');
    n_docs = info[0];
    n_vocab = info[1];
    
    //parse the documents
    while (std::getline(ifs, str)){
        std::vector<int> sentence = Utils::split_and_stoi(str, ' ');
        doc_sentence.push_back(sentence);
    }
    n_docs = doc_sentence.size();
}






