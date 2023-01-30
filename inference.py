import time
import torch

import GraphCodeBERT
import CodeBERT

def CodeBERT_predict(text):
    if len(text) == 0:
        return 0, 0
    latency, pred = CodeBERT.prediction(text)
    return latency, pred

def GraphCodeBERT_predict(text):
    if len(text) == 0:
        return 0, 0
    latency, pred = GraphCodeBERT.prediction(text)
    return latency, pred

def CodeBERT_compressor_predict(text):
    if len(text) == 0:
        return 0, 0
    _, pred = CodeBERT.prediction(text)
    latency, _ = CodeBERT.distill_pred(text)
    return latency, pred

def GraphCodeBERT_compressor_predict(text):
    if len(text) == 0:
        return 0, 0
    _, pred = GraphCodeBERT.prediction(text)
    latency, _ = GraphCodeBERT.distill_pred(text)
    return latency, pred