from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.inference import transform
from edward.models import RandomVariable, Trace
from edward.util import copy, get_descendants

try:
  from edward.models import Normal
  from tensorflow.contrib.distributions import kl_divergence
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))

tfd = tf.contrib.distributions


def klqp(model, variational, align_latent, align_data,
         scale=None, n_samples=1, kl_scaling=None, auto_transform=True,
         summary_key=None, *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective by automatically selecting from a
  variety of black box inference techniques.

  Args:
    model: function whose inputs are a subset of `args` (e.g., for
      discriminative). Output is not used.
      TODO auto_transform docstring
      Collection of random variables to perform inference on.
      If list, each random variable will be implictly optimized using
      a `Normal` random variable that is defined internally with a
      free parameter per location and scale and is initialized using
      standard normal draws. The random variables to approximate must
      be continuous.
    variational: function whose inputs are a subset of `args` (e.g.,
      for amortized). Output is not used.
    align_latent: function of string, aligning `model` latent
      variables with `variational`. It takes a model variable's name
      as input and returns a string, indexing `variational`'s trace;
      else identity.
    align_data: function of string, aligning `model` observed
      variables with data. It takes a model variable's name as input
      and returns an integer, indexing `args`; else identity.
    scale: function of string, aligning `model` observed
      variables with scale factors. It takes a model variable's name
      as input and returns a scale factor; else 1.0. The scale
      factor's shape must be broadcastable; it is multiplied
      element-wise to the random variable. For example, this is useful
      for mini-batch scaling when inferring global variables, or
      applying masks on a random variable.
    n_samples: int, optional.
      Number of samples from variational model for calculating
      stochastic gradients.
    kl_scaling: function of string, aligning `model` latent
      variables with KL scale factors. This provides option to scale
      terms when using ELBO with KL divergence. If the KL divergence
      terms are

      $\\alpha_p \mathbb{E}_{q(z\mid x, \lambda)} [
            \log q(z\mid x, \lambda) - \log p(z)],$

      then pass {$p(z)$: $\\alpha_p$} as `kl_scaling`,
      where $\\alpha_p$ is a tensor. Its shape must be broadcastable;
      it is multiplied element-wise to the batchwise KL terms.
    args: data inputs. It is passed at compile-time in Graph
      mode or runtime in Eager mode.

  #### Notes

  `KLqp` also optimizes any model parameters $p(z \mid x;
  \\theta)$. It does this by variational EM, maximizing

  $\mathbb{E}_{q(z; \lambda)} [ \log p(x, z; \\theta) ]$

  with respect to $\\theta$.

  In conditional inference, we infer $z$ in $p(z, \\beta
  \mid x)$ while fixing inference over $\\beta$ using another
  distribution $q(\\beta)$. During gradient calculation, instead
  of using the model's density

  $\log p(x, z^{(s)}), z^{(s)} \sim q(z; \lambda),$

  for each sample $s=1,\ldots,S$, `KLqp` uses

  $\log p(x, z^{(s)}, \\beta^{(s)}),$

  where $z^{(s)} \sim q(z; \lambda)$ and $\\beta^{(s)}
  \sim q(\\beta)$.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.

  ##

  $-\\text{ELBO} =
      -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]$

  KLqp supports

  1. score function gradients [@paisley2012variational]
  2. reparameterization gradients [@kingma2014auto]

  of the loss function.

  If the KL divergence between the variational model and the prior
  is tractable, then the loss function can be written as

  $-\mathbb{E}_{q(z; \lambda)}[\log p(x \mid z)] +
      \\text{KL}( q(z; \lambda) \| p(z) ),$

  where the KL term is computed analytically [@kingma2014auto]. We
  compute this automatically when $p(z)$ and $q(z; \lambda)$ are
  Normal.
  """
  p_log_prob = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  return loss
  # TODO reg penalty inside gradients and not just model
  # TODO implement after rb
  # is_reparameterizable = all([
  #     rv.reparameterization_type ==
  #     tf.contrib.distributions.FULLY_REPARAMETERIZED
  #     for rv in six.itervalues(latent_vars)])
  # is_analytic_kl = all([isinstance(z, Normal) and isinstance(qz, Normal)
  #                       for z, qz in six.iteritems(latent_vars)])
  if scale is None:
    scale = lambda name: 1.0
  def _intercept(f, *fargs, **fkwargs):
    """Set model's sample values to variational distribution's and data."""
    name = fkwargs.get('name', None)
    key = align_data(name)
    if isinstance(key, int):
      fkwargs['value'] = args[key]
    elif kwargs.get(key, None) is not None:
      fkwargs['value'] = kwargs.get(key)
    else:
      qz = posterior_trace[align_latent(name)].value
      fkwargs['value'] = qz.value
    return f(*fargs, **fkwargs)

  # In each trace, collect key-value pairs of (rv, rv's (scaled) log prob).
  # We use this collection to build individual gradients (surrogate
  # losses) for each model/variational parameter.
  p_log_probs = [{}] * n_samples
  q_log_probs = [{}] * n_samples
  for s in range(n_samples):
    with Trace() as posterior_trace:
      call_function_up_to_args(variational, *args, **kwargs)
    with Trace(intercept=_intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    for name, node in six.iteritems(model_trace):
      rv = node.value
      scale_factor = scale(name)
      p_log_prob[s][rv] = tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
      posterior_node = posterior_trace.get(align_latent(name), None)
      if posterior_node is not None:
        qz = posterior_node.value
        q_log_prob[s][qz] = tf.reduce_sum(scale_factor *
            qz.log_prob(tf.stop_gradient(qz.value)))

  # Take gradients of Rao-Blackwellized loss for each variational parameter.
  # TODO get rvs for each trace
  p_rvs = list(six.iterkeys(latent_vars)) + \
      [x for x in six.iterkeys(data) if isinstance(x, RandomVariable)]
  q_rvs = list(six.itervalues(latent_vars))
  reverse_latent_vars = {v: k for k, v in six.iteritems(latent_vars)}
  grads = []
  grads_vars = []
  # TODO var_list
  for var in var_list:
    # Get all variational factors depending on the parameter.
    descendants = get_descendants(tf.convert_to_tensor(var), q_rvs)
    if len(descendants) == 0:
      continue  # skip if not a variational parameter
    # Get p and q's Markov blanket wrt these latent variables.
    var_p_rvs = set()
    for qz in descendants:
      z = reverse_latent_vars[qz]
      var_p_rvs.update(z.get_blanket(p_rvs) + [z])

    var_q_rvs = set()
    for qz in descendants:
      var_q_rvs.update(qz.get_blanket(q_rvs) + [qz])

    pi_log_prob = [0.0] * n_samples
    qi_log_prob = [0.0] * n_samples
    for s in range(n_samples):
      pi_log_prob[s] = tf.reduce_sum([p_log_probs[s][rv] for rv in var_p_rvs])
      qi_log_prob[s] = tf.reduce_sum([q_log_probs[s][rv] for rv in var_q_rvs])

    pi_log_prob = tf.stack(pi_log_prob)
    qi_log_prob = tf.stack(qi_log_prob)
    grad = tf.gradients(
        -tf.reduce_mean(qi_log_prob *
                        tf.stop_gradient(pi_log_prob - qi_log_prob)) +
        tf.reduce_sum(tf.losses.get_regularization_losses()),
        var)
    grads.extend(grad)
    grads_vars.append(var)

  # Take gradients of total loss function for model parameters.
  p_log_prob = tf.reduce_mean([tf.reduce_sum(list(six.itervalues(p_log_prob)))
                               for p_log_prob in p_log_probs])
  q_log_prob = tf.reduce_mean([tf.reduce_sum(list(six.itervalues(q_log_prob)))
                               for q_log_prob in q_log_probs])
  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses()))
  if summary_key is not None:
    tf.summary.scalar("loss/p_log_prob", p_log_prob,
                      collections=[summary_key])
    tf.summary.scalar("loss/q_log_prob", q_log_prob,
                      collections=[summary_key])
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=[summary_key])

  loss = q_log_prob - p_log_prob + reg_penalty
  model_vars = [v for v in var_list if v not in grads_vars]
  model_grads = tf.gradients(loss, model_vars)
  grads.extend(model_grads)
  grads_vars.extend(model_vars)
  grads_and_vars = list(zip(grads, grads_vars))
  return loss, grads_and_vars


def klqp_reparameterization(
    model, variational, align_latent, align_data,
    scale=None, n_samples=1, auto_transform=True, summary_key=None,
    *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the reparameterization
  gradient.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.

  Build loss function equal to KL(q||p) up to a constant. Its
  automatic differentiation is a stochastic gradient of

  $-\\text{ELBO} =
      -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]$

  based on the reparameterization trick [@kingma2014auto].

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.

  Note if user defines constrained posterior, then auto_transform
  can do inference on real-valued; then test time user can use
  constrained. If user defines unconstrained posterior, then how to
  work with constrained at test time? For now, user must manually
  write the bijectors according to transform.
  """
  if scale is None:
    scale = lambda name: 1.0
  def _intercept(f, *fargs, **fkwargs):
    """Set model's sample values to variational distribution's and data."""
    name = fkwargs.get('name', None)
    key = align_data(name)
    if isinstance(key, int):
      fkwargs['value'] = args[key]
    elif kwargs.get(key, None) is not None:
      fkwargs['value'] = kwargs.get(key)
    else:
      qz = posterior_trace[align_latent(name)].value
      fkwargs['value'] = qz.value
    # if auto_transform and 'qz' in locals():
    #   # TODO for generation to work, must output original dist. to
    #   keep around TD? must maintain another stack to write to as a
    #   side-effect (or augment the original stack).
    #   return transform(f, qz, *fargs, **fkwargs)
    return f(*fargs, **fkwargs)

  p_log_prob = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  for s in range(n_samples):
    with Trace() as posterior_trace:
      call_function_up_to_args(variational, *args, **kwargs)
    with Trace(intercept=_intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    for name, node in six.iteritems(model_trace):
      rv = node.value
      scale_factor = scale(name)
      p_log_prob[s] += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
      posterior_node = posterior_trace.get(align_latent(name), None)
      if posterior_node is not None:
        qz = posterior_node.value
        q_log_prob[s] += tf.reduce_sum(scale_factor * qz.log_prob(qz.value))

  p_log_prob = tf.reduce_mean(p_log_prob)
  q_log_prob = tf.reduce_mean(q_log_prob)
  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  if summary_key is not None:
    tf.summary.scalar("loss/p_log_prob", p_log_prob,
                      collections=[summary_key])
    tf.summary.scalar("loss/q_log_prob", q_log_prob,
                      collections=[summary_key])
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=[summary_key])
  loss = q_log_prob - p_log_prob + reg_penalty
  return loss


def klqp_reparameterization_kl(
    model, variational, align_latent, align_data,
    scale=None, n_samples=1, kl_scaling=None, auto_transform=True,
    summary_key=None, *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the reparameterization
  gradient and an analytic KL term.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.

  Build loss function. Its automatic differentiation
  is a stochastic gradient of

  .. math::

    -\\text{ELBO} =  - ( \mathbb{E}_{q(z; \lambda)} [ \log p(x \mid z) ]
          + \\text{KL}(q(z; \lambda) \| p(z)) )

  based on the reparameterization trick [@kingma2014auto].

  It assumes the KL is analytic.

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.
  """
  if scale is None:
    scale = lambda name: 1.0
  if kl_scaling is None:
    kl_scaling = lambda name: 1.0
  def _intercept(f, *fargs, **fkwargs):
    """Set model's sample values to variational distribution's and data."""
    name = fkwargs.get('name', None)
    key = align_data(name)
    if isinstance(key, int):
      fkwargs['value'] = args[key]
    elif kwargs.get(key, None) is not None:
      fkwargs['value'] = kwargs.get(key)
    else:
      qz = posterior_trace[align_latent(name)].value
      fkwargs['value'] = qz.value
    return f(*fargs, **fkwargs)

  p_log_lik = [0.0] * n_samples
  for s in range(n_samples):
    with Trace() as posterior_trace:
      call_function_up_to_args(variational, *args, **kwargs)
    with Trace(intercept=_intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    for name, node in six.iteritems(model_trace):
      rv = node.value
      scale_factor = scale(name)
      p_log_lik[s] += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))

  p_log_lik = tf.reduce_mean(p_log_lik)

  kl_penalty = 0.0
  for name, node in six.iteritems(model_trace):
    rv = node.value
    posterior_node = posterior_trace.get(align_latent(name), None)
    if posterior_node is not None:
      qz = posterior_node.value
      kl_penalty += tf.reduce_sum(kl_scaling(name) * kl_divergence(qz, rv))

  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  if summary_key is not None:
    tf.summary.scalar("loss/p_log_lik", p_log_lik,
                      collections=[summary_key])
    tf.summary.scalar("loss/kl_penalty", kl_penalty,
                      collections=[summary_key])
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=[summary_key])
  loss = -p_log_lik + kl_penalty + reg_penalty
  return loss


def klqp_score(
    model, variational, align_latent, align_data,
    scale=None, n_samples=1, auto_transform=True, summary_key=None,
    *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the score function
  gradient.

  Build loss function equal to KL(q||p) up to a constant. It
  returns an surrogate loss function whose automatic differentiation
  is based on the score function estimator [@paisley2012variational].

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  if scale is None:
    scale = lambda name: 1.0
  def _intercept(f, *fargs, **fkwargs):
    """Set model's sample values to variational distribution's and data."""
    name = fkwargs.get('name', None)
    key = align_data(name)
    if isinstance(key, int):
      fkwargs['value'] = args[key]
    elif kwargs.get(key, None) is not None:
      fkwargs['value'] = kwargs.get(key)
    else:
      qz = posterior_trace[align_latent(name)].value
      fkwargs['value'] = qz.value
    return f(*fargs, **fkwargs)

  p_log_prob = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  for s in range(n_samples):
    with Trace() as posterior_trace:
      call_function_up_to_args(variational, *args, **kwargs)
    with Trace(intercept=_intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    for name, node in six.iteritems(model_trace):
      rv = node.value
      scale_factor = scale(name)
      p_log_prob[s] += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
      posterior_node = posterior_trace.get(align_latent(name), None)
      if posterior_node is not None:
        qz = posterior_node.value
        q_log_prob[s] += tf.reduce_sum(
            scale_factor * qz.log_prob(tf.stop_gradient(qz.value)))

  p_log_prob = tf.stack(p_log_prob)
  q_log_prob = tf.stack(q_log_prob)
  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  if summary_key is not None:
    tf.summary.scalar("loss/p_log_prob", tf.reduce_mean(p_log_prob),
                      collections=[summary_key])
    tf.summary.scalar("loss/q_log_prob", tf.reduce_mean(q_log_prob),
                      collections=[summary_key])
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=[summary_key])
  losses = q_log_prob - p_log_prob
  loss = tf.reduce_mean(losses) + reg_penalty
  surrogate_loss = (tf.reduce_mean(q_log_prob * tf.stop_gradient(losses)) +
                    reg_penalty)
  return loss, surrogate_loss


def klqp_score_rb(
    latent_vars=None, data=None, n_samples=1,
    auto_transform=True, scale=None, summary_key=None):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This class minimizes the objective using the score function gradient
  and Rao-Blackwellization.

  Build loss function and gradients based on the score function
  estimator [@paisley2012variational] and Rao-Blackwellization
  [@ranganath2014black].

  Computed by sampling from :math:`q(z;\lambda)` and evaluating the
  expectation using Monte Carlo sampling and Rao-Blackwellization.

  #### Notes

  Current Rao-Blackwellization is limited to Rao-Blackwellizing across
  stochastic nodes in the computation graph. It does not
  Rao-Blackwellize within a node such as when a node represents
  multiple random variables via non-scalar batch shape.

  The objective function also adds to itself a summation over all
  tensors in the `REGULARIZATION_LOSSES` collection.
  """
  # TODO rewrite score rb
  return build_score_rb_loss(
      latent_vars, data, scale, n_samples, summary_key)


def default_constructor(latent_vars):
  if isinstance(latent_vars, list):
    with tf.variable_scope(None, default_name="posterior"):
      latent_vars_dict = {}
      continuous = \
          ('01', 'nonnegative', 'simplex', 'real', 'multivariate_real')
      for z in latent_vars:
        if not hasattr(z, 'support') or z.support not in continuous:
          raise AttributeError(
              "Random variable {} is not continuous or a random "
              "variable with supported continuous support.".format(z))
        batch_event_shape = z.batch_shape.concatenate(z.event_shape)
        loc = tf.Variable(tf.random_normal(batch_event_shape))
        scale = tf.nn.softplus(
            tf.Variable(tf.random_normal(batch_event_shape)))
        latent_vars_dict[z] = Normal(loc=loc, scale=scale)
      latent_vars = latent_vars_dict
  return latent_vars


def call_function_up_to_args(f, *args, **kwargs):
  import inspect
  if hasattr(f, "_func"):  # make_template()
    argspec = inspect.getargspec(f._func)
  else:
    argspec = inspect.getargspec(f)
  num_kwargs = len(argspec.defaults) if argspec.defaults is not None else 0
  num_args = len(argspec.args) - num_kwargs
  if num_args > 0:
    return f(args[:num_args], **kwargs)
  elif num_kwargs > 0:
    return f(**kwargs)
  return f()
