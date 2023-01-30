import time
import torch

import GraphCodeBERT
import CodeBERT

def CodeBERT_predict():
    print(CodeBERT.prediction('''
    // An vulnerable example form ReVeal dataset
    int getulong(const char * numstr, unsigned long int * result) {
    long long int val;
    char * endptr;
    errno = 0;
    val = strtoll(numstr, & endptr, 0);
    if (('\\0' == * numstr) || ('\\0' != * endptr) || (ERANGE == errno) || (val != (unsigned long int) val)) {
        return 0;
    }
    * result = (unsigned long int) val;
    return 1;
}
    '''))

def GraphCodeBERT_predict():
    print(GraphCodeBERT.prediction('''
    // An vulnerable example form ReVeal dataset
    int getulong(const char * numstr, unsigned long int * result) {
    long long int val;
    char * endptr;
    errno = 0;
    val = strtoll(numstr, & endptr, 0);
    if (('\\0' == * numstr) || ('\\0' != * endptr) || (ERANGE == errno) || (val != (unsigned long int) val)) {
        return 0;
    }
    * result = (unsigned long int) val;
    return 1;
}
    '''))


def CodeBERT_compressor_predict():
    print(CodeBERT.distill_pred('''
    // An vulnerable example form ReVeal dataset
    int getulong(const char * numstr, unsigned long int * result) {
    long long int val;
    char * endptr;
    errno = 0;
    val = strtoll(numstr, & endptr, 0);
    if (('\\0' == * numstr) || ('\\0' != * endptr) || (ERANGE == errno) || (val != (unsigned long int) val)) {
        return 0;
    }
    * result = (unsigned long int) val;
    return 1;
}
    '''))

def GraphCodeBERT_compressor_predict():
    print(GraphCodeBERT.distill_pred('''
    // An vulnerable example form ReVeal dataset
    int getulong(const char * numstr, unsigned long int * result) {
    long long int val;
    char * endptr;
    errno = 0;
    val = strtoll(numstr, & endptr, 0);
    if (('\\0' == * numstr) || ('\\0' != * endptr) || (ERANGE == errno) || (val != (unsigned long int) val)) {
        return 0;
    }
    * result = (unsigned long int) val;
    return 1;
}
    '''))

# GraphCodeBERT_compressor_predict()
CodeBERT_compressor_predict()