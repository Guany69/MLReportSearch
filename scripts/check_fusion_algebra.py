"""Numerically verify that geometric softmax fusion equals weighted-logit fusion."""
import numpy as np
from scipy.special import logsumexp

dense = np.array([.2, .7, -.1]); lexical = np.array([.8, .1, .3])
alpha, td, tl = .6, .05, .08
ld = dense/td-logsumexp(dense/td); ll = lexical/tl-logsumexp(lexical/tl)
geometric = np.exp(alpha*ld+(1-alpha)*ll-logsumexp(alpha*ld+(1-alpha)*ll))
logits = alpha*dense/td+(1-alpha)*lexical/tl
weighted = np.exp(logits-logsumexp(logits))
assert np.allclose(geometric, weighted)
print("max absolute difference:", np.max(np.abs(geometric-weighted)))
