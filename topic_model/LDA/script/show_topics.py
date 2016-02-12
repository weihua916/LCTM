import numpy as np
import argparse
import util

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', default='',help='input directory')
    parser.add_argument('--output_dir', '-o', default='',help='input directory')
    args = parser.parse_args()
    
    return args.input_dir, args.output_dir

def main():
    input_dir, output_dir = parse()
    
    #topic word distribution
    phi = np.loadtxt(output_dir + 'phi.txt')
    
    #vocabulary file
    vocab_list = util.read_list(input_dir + 'vocab.txt')

    #number of top words to be printed
    top_num = 10
    
    for k, phi_k in enumerate(phi):
        top_arg = np.argsort(-phi_k)[:top_num]
        top_wordlist = [vocab_list[arg] for arg in top_arg]
        
        print str(k) + ' th topic: '
        print ' '.join(top_wordlist) + '\n'

if __name__ == '__main__':
    main()

