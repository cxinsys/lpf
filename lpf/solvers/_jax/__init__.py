"""JAX-accelerated solvers.

These solvers compile the entire integration loop with ``@jax.jit`` +
``lax.scan`` so each call reduces to a single XLA program launch on the
selected device (CPU / GPU / TPU).

The auto-dispatch in ``lpf.solvers.solver.Solver`` picks these up whenever
a model's array module is a ``JaxModule`` (i.e. when the user requests
``device="jax:cpu"``, ``"jax:gpu:0"``, ``"jax:tpu:0"`` etc.).

Supported reaction-diffusion models:
    - LiawModel
    - GrayScottModel
    - BrusselatorModel
    - FitzHughNagumoModel
    - SchnakenbergModel
    - GiererMeinhardtModel
"""
