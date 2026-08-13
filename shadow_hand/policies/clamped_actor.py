from collections.abc import Callable, Sequence
from dataclasses import field

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax.nn.initializers import constant
from sbx.common.jax_layers import NatureCNN
from sbx.common.policies import Flatten

tfd = tfp.distributions


class ClampedActor(nn.Module):
    action_dim: int
    net_arch: Sequence[int]
    log_std_init: float = 0.0
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
    num_discrete_choices: int | Sequence[int] | None = None
    max_num_choices: int = 0
    split_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    ortho_init: bool = False
    features_extractor: type[NatureCNN] | None = None
    features_dim: int = 512
    log_std_min: float = -3.0
    log_std_max: float = 0.0

    def get_std(self) -> jnp.ndarray:
        return jnp.array(0.0)

    def __post_init__(self) -> None:
        if isinstance(self.num_discrete_choices, np.ndarray):
            self.max_num_choices = max(self.num_discrete_choices)
            self.split_indices = np.cumsum(self.num_discrete_choices[:-1])

        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> "tfd.Distribution":  # type: ignore[name-defined]
        if self.features_extractor is not None:
            x = self.features_extractor(self.features_dim, self.activation_fn)(x)

        x = Flatten()(x)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)

        if self.ortho_init:
            orthogonal_init = nn.initializers.orthogonal(scale=0.01)
            bias_init = nn.initializers.zeros
            action_logits = nn.Dense(
                self.action_dim, kernel_init=orthogonal_init, bias_init=bias_init
            )(x)
        else:
            action_logits = nn.Dense(self.action_dim)(x)

        if self.num_discrete_choices is not None:
            raise NotImplementedError(
                "ClampedActor only supports continuous Box action spaces; "
                "discrete actions don't have log_std and don't need clamping."
            )

        log_std_param = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
        log_std = jnp.clip(log_std_param, self.log_std_min, self.log_std_max)
        return tfd.MultivariateNormalDiag(loc=action_logits, scale_diag=jnp.exp(log_std))


def make_clamped_actor(log_std_min: float, log_std_max: float) -> type[nn.Module]:
    if log_std_min >= log_std_max:
        raise ValueError(f"log_std_min ({log_std_min}) must be < log_std_max ({log_std_max})")

    _min = float(log_std_min)
    _max = float(log_std_max)

    class _ClampedActor(ClampedActor):
        log_std_min: float = _min
        log_std_max: float = _max

    _ClampedActor.__name__ = f"ClampedActor_{_min:+.2f}_{_max:+.2f}".replace(".", "p").replace(
        "+", ""
    )
    _ClampedActor.__qualname__ = _ClampedActor.__name__
    return _ClampedActor
