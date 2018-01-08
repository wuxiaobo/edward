#!/usr/bin/env python
"""Normal-normal model using Hamiltonian Monte Carlo."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Empirical, Normal
from edward.util import get_session, Progbar

ed.set_seed(42)

# DATA
x_data = np.array([0.0] * 50)

# MODEL: Normal-Normal with known variance
mu = Normal(loc=0.0, scale=1.0)
x = Normal(loc=tf.ones(50) * mu, scale=1.0)

# INFERENCE
qmu = Normal(loc=tf.Variable(0.0), scale=tf.nn.softplus(tf.Variable(1.0))+1e-3)

# analytic solution: N(loc=0.0, scale=\sqrt{1/51}=0.140)
loss, grads_and_vars = ed.klqp({mu: qmu}, data={x: x_data})

train_op = tf.train.AdamOptimizer().apply_gradients(grads_and_vars)
progbar = Progbar(1000)
sess = get_session()
tf.global_variables_initializer().run()
for t in range(1, 1001):
  loss_val, _ = sess.run([loss, train_op])
  if t % 50 == 0:
    progbar.update(t, {"Loss": loss_val})

# # CRITICISM
sess = get_session()
mean, stddev = sess.run([qmu.mean(), qmu.stddev()])
print("Inferred posterior mean:")
print(mean)
print("Inferred posterior stddev:")
print(stddev)

# Check convergence with visual diagnostics.
# samples = sess.run(qmu.params)

# # Plot histogram.
# plt.hist(samples, bins='auto')
# plt.show()

# # Trace plot.
# plt.plot(samples)
# plt.show()
