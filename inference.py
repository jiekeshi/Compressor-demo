import time
import torch

import GraphCodeBERT
import CodeBERT

def CodeBERT_predict(text):
    latency, pred = CodeBERT.prediction(text)
    return latency, pred

def GraphCodeBERT_predict(text):
    latency, pred = GraphCodeBERT.prediction(text)
    return latency, pred

def CodeBERT_compressor_predict(text):
    _, pred = CodeBERT.prediction(text)
    latency, _ = CodeBERT.distill_pred(text)
    return latency, pred

def GraphCodeBERT_compressor_predict(text):
    _, pred = GraphCodeBERT.prediction(text)
    latency, _ = GraphCodeBERT.distill_pred(text)
    return latency, pred