from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ghmvae.priors.base import Prior


def _as_float(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    return float(arr.reshape(-1)[0])


def _as_vector(value: Any) -> Tensor:
    return torch.as_tensor(np.asarray(value, dtype=np.float64).reshape(-1), dtype=torch.float32)


def _as_spd(value: Any, eps: float = 1.1e-5) -> Tensor:
    """Return a float32 SPD matrix inside the declared regularity region.

    The safe GH protocols require covariance eigenvalues of at least ``1e-5``.
    A slightly larger conversion floor keeps the float32 result on the regular
    side of that boundary instead of manufacturing an anchor that is rejected
    immediately after a numerically successful fit.
    """

    mat_np = np.asarray(value, dtype=np.float64)
    mat_np = 0.5 * (mat_np + mat_np.T)
    eigvals = np.linalg.eigvalsh(mat_np)
    min_eig = float(np.min(eigvals))
    if min_eig < eps:
        mat_np = mat_np + (eps - min_eig) * np.eye(mat_np.shape[0])
    return torch.as_tensor(mat_np, dtype=torch.float32)


class _LogBesselK(torch.autograd.Function):
    """Fixed-order log K_v(x) with gradient with respect to x.

    PyTorch exposes K_0 and K_1 but not arbitrary orders. GH densities require
    arbitrary fixed orders, so we use SciPy for the scalar special function
    values and its analytic derivative. Mixture parameters are fixed during VAE
    training; gradients are only needed with respect to latent points.
    """

    @staticmethod
    def forward(ctx: Any, x: Tensor, order: float) -> Tensor:
        from scipy.special import kve

        x_cpu = x.detach().double().cpu().numpy()
        order_f = float(order)
        k_v = kve(order_f, x_cpu)
        log_value = np.log(k_v) - x_cpu
        deriv = -0.5 * (kve(order_f - 1.0, x_cpu) + kve(order_f + 1.0, x_cpu)) / k_v
        deriv = np.nan_to_num(deriv, nan=0.0, posinf=0.0, neginf=0.0)
        deriv_t = torch.as_tensor(deriv, device=x.device, dtype=x.dtype)
        ctx.save_for_backward(deriv_t)
        return torch.as_tensor(log_value, device=x.device, dtype=x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        (deriv,) = ctx.saved_tensors
        return grad_output * deriv, None


def log_bessel_k(order: float, x: Tensor, eps: float = 1e-8) -> Tensor:
    x_safe = x.clamp_min(eps)
    if abs(abs(float(order)) - 0.5) < 1e-12:
        return 0.5 * (math.log(math.pi / 2.0) - torch.log(x_safe)) - x_safe
    return _LogBesselK.apply(x_safe, float(order))


class Component(nn.Module):
    family: str

    def log_prob(self, z: Tensor) -> Tensor:
        raise NotImplementedError


class MultivariateNormalComponent(Component):
    family = "mvn"

    def __init__(self, mu: Any, sigma: Any) -> None:
        super().__init__()
        mu_t = _as_vector(mu)
        sigma_t = _as_spd(sigma)
        self.register_buffer("mu", mu_t)
        self.register_buffer("sigma", sigma_t)
        self.register_buffer("inv_sigma", torch.linalg.inv(sigma_t))
        self.register_buffer("logdet_sigma", torch.linalg.slogdet(sigma_t).logabsdet)

    def log_prob(self, z: Tensor) -> Tensor:
        diff = z - self.mu.to(z)
        inv_sigma = self.inv_sigma.to(z)
        q = ((diff @ inv_sigma) * diff).sum(dim=1)
        d = z.shape[1]
        return -0.5 * (d * math.log(2.0 * math.pi) + self.logdet_sigma.to(z) + q)


class MultivariateTComponent(Component):
    family = "mvt"

    def __init__(self, mu: Any, sigma: Any, df: Any) -> None:
        super().__init__()
        mu_t = _as_vector(mu)
        sigma_t = _as_spd(sigma)
        self.df = float(max(_as_float(df), 1e-4))
        self.register_buffer("mu", mu_t)
        self.register_buffer("sigma", sigma_t)
        self.register_buffer("inv_sigma", torch.linalg.inv(sigma_t))
        self.register_buffer("logdet_sigma", torch.linalg.slogdet(sigma_t).logabsdet)

    def log_prob(self, z: Tensor) -> Tensor:
        diff = z - self.mu.to(z)
        inv_sigma = self.inv_sigma.to(z)
        q = ((diff @ inv_sigma) * diff).sum(dim=1)
        d = z.shape[1]
        nu = torch.as_tensor(self.df, device=z.device, dtype=z.dtype)
        return (
            torch.lgamma((nu + d) / 2.0)
            - torch.lgamma(nu / 2.0)
            - 0.5 * (self.logdet_sigma.to(z) + d * torch.log(nu * math.pi))
            - 0.5 * (nu + d) * torch.log1p(q / nu)
        )


class SkewNormalComponent(Component):
    family = "mvsn"

    def __init__(self, mu: Any, sigma: Any, shape: Any) -> None:
        super().__init__()
        mu_t = _as_vector(mu)
        sigma_t = _as_spd(sigma)
        shape_t = _as_vector(shape)
        eigvals, eigvecs = torch.linalg.eigh(sigma_t)
        inv_half = eigvecs @ torch.diag(eigvals.clamp_min(1e-8).rsqrt()) @ eigvecs.T
        a_t = inv_half.T @ shape_t
        self.register_buffer("mu", mu_t)
        self.register_buffer("sigma", sigma_t)
        self.register_buffer("inv_sigma", torch.linalg.inv(sigma_t))
        self.register_buffer("logdet_sigma", torch.linalg.slogdet(sigma_t).logabsdet)
        self.register_buffer("a", a_t)

    def log_prob(self, z: Tensor) -> Tensor:
        diff = z - self.mu.to(z)
        inv_sigma = self.inv_sigma.to(z)
        q = ((diff @ inv_sigma) * diff).sum(dim=1)
        d = z.shape[1]
        mvn_lp = -0.5 * (d * math.log(2.0 * math.pi) + self.logdet_sigma.to(z) + q)
        cdf_arg = diff @ self.a.to(z)
        return math.log(2.0) + mvn_lp + torch.special.log_ndtr(cdf_arg)


class SkewLaplaceComponent(Component):
    family = "mvsl"

    def __init__(self, mu: Any, sigma: Any, gamma: Any) -> None:
        super().__init__()
        mu_t = _as_vector(mu)
        sigma_t = _as_spd(sigma)
        gamma_t = _as_vector(gamma)
        inv_sigma = torch.linalg.inv(sigma_t)
        gamma_norm = gamma_t @ inv_sigma @ gamma_t
        a = torch.sqrt(1.0 + gamma_norm)
        log_const = (
            -0.5 * torch.linalg.slogdet(sigma_t).logabsdet
            - (
                len(mu_t) * math.log(2.0)
                + (len(mu_t) - 1) * 0.5 * math.log(math.pi)
                + torch.log(a)
                + torch.lgamma(torch.as_tensor((len(mu_t) + 1.0) / 2.0))
            )
        )
        self.register_buffer("mu", mu_t)
        self.register_buffer("inv_sigma", inv_sigma)
        self.register_buffer("gamma", gamma_t)
        self.register_buffer("a", a)
        self.register_buffer("log_const", log_const)

    def log_prob(self, z: Tensor) -> Tensor:
        diff = z - self.mu.to(z)
        inv_sigma = self.inv_sigma.to(z)
        dist_sq = ((diff @ inv_sigma) * diff).sum(dim=1).clamp_min(1e-14)
        dist = torch.sqrt(dist_sq)
        lin = (diff @ inv_sigma) @ self.gamma.to(z)
        return self.log_const.to(z) - self.a.to(z) * dist + lin


class GeneralizedHyperbolicComponent(Component):
    family = "gh"

    def __init__(
        self,
        lmbda: Any,
        chi: Any,
        psi: Any,
        mu: Any,
        sigma: Any,
        gamma: Any,
        family: str = "mghp",
    ) -> None:
        super().__init__()
        self.family = family
        self.lmbda = _as_float(lmbda)
        self.chi = max(_as_float(chi), 0.0)
        self.psi = max(_as_float(psi), 0.0)
        mu_t = _as_vector(mu)
        sigma_t = _as_spd(sigma)
        gamma_t = _as_vector(gamma)
        inv_sigma = torch.linalg.inv(sigma_t)
        skewness_norm = gamma_t @ inv_sigma @ gamma_t
        self.register_buffer("mu", mu_t)
        self.register_buffer("sigma", sigma_t)
        self.register_buffer("gamma", gamma_t)
        self.register_buffer("inv_sigma", inv_sigma)
        self.register_buffer("logdet_sigma", torch.linalg.slogdet(sigma_t).logabsdet)
        self.register_buffer("skewness_norm", skewness_norm)

    @property
    def dimension(self) -> int:
        return int(self.mu.numel())

    def _sym_student_log_prob(self, q: Tensor) -> Tensor:
        d = self.dimension
        lmd = self.lmbda - d / 2.0
        chi = max(self.chi, 1e-12)
        log_const_top = -self.lmbda * math.log(chi) + math.lgamma(-lmd)
        log_const_bottom = d / 2.0 * math.log(math.pi) + 0.5 * self.logdet_sigma.to(q) + math.lgamma(-self.lmbda)
        return log_const_top + lmd * torch.log(chi + q) - log_const_bottom

    def log_prob(self, z: Tensor) -> Tensor:
        d = self.dimension
        diff = z - self.mu.to(z)
        inv_sigma = self.inv_sigma.to(z)
        q = ((diff @ inv_sigma) * diff).sum(dim=1).clamp_min(0.0)
        gamma = self.gamma.to(z)
        skewness_norm = self.skewness_norm.to(z).clamp_min(0.0)
        skewness_scaled = (diff @ inv_sigma) @ gamma
        symmetric = bool(float(self.skewness_norm.detach().cpu()) < 1e-12)
        lmd = self.lmbda - d / 2.0

        if self.psi <= 0.0:
            if symmetric:
                return self._sym_student_log_prob(q)
            chi = max(self.chi, 1e-12)
            skew_norm = skewness_norm.clamp_min(1e-12)
            arg = torch.sqrt((chi + q) * skew_norm).clamp_min(1e-8)
            log_const_top = -self.lmbda * math.log(chi) - lmd * torch.log(skew_norm)
            log_const_bottom = (
                d / 2.0 * math.log(2.0 * math.pi)
                + 0.5 * self.logdet_sigma.to(z)
                + math.lgamma(-self.lmbda)
                - (self.lmbda + 1.0) * math.log(2.0)
            )
            return log_const_top + log_bessel_k(lmd, arg) + skewness_scaled - log_const_bottom + lmd * torch.log(arg)

        psi = max(self.psi, 1e-12)
        psi_plus = psi + skewness_norm
        if self.chi > 0.0:
            chi = self.chi
            arg = torch.sqrt(psi_plus * (chi + q)).clamp_min(1e-8)
            log_const_top = (
                -self.lmbda / 2.0 * math.log(psi * chi)
                + d / 2.0 * math.log(psi)
                - lmd * torch.log1p(skewness_norm / psi)
            )
            const_arg = torch.as_tensor(math.sqrt(chi * psi), device=z.device, dtype=z.dtype)
            log_const_bottom = (
                d / 2.0 * math.log(2.0 * math.pi)
                + log_bessel_k(self.lmbda, const_arg)
                + 0.5 * self.logdet_sigma.to(z)
            )
            return log_const_top + log_bessel_k(lmd, arg) + skewness_scaled - log_const_bottom + lmd * torch.log(arg)

        q_safe = q.clamp_min(1e-8)
        arg = torch.sqrt(psi_plus * q_safe).clamp_min(1e-8)
        log_const_top = (
            d * math.log(psi) / 2.0
            + (1.0 - self.lmbda) * math.log(2.0)
            - lmd * torch.log1p(skewness_norm / psi)
        )
        log_const_bottom = d / 2.0 * math.log(2.0 * math.pi) + math.lgamma(self.lmbda) + 0.5 * self.logdet_sigma.to(z)
        return log_const_top + log_bessel_k(lmd, arg) + skewness_scaled - log_const_bottom + lmd * torch.log(arg)


class TorchMixturePrior(Prior):
    name = "torch_mixture"

    def __init__(self, log_weights: Tensor, components: Sequence[Component], name: str = "torch_mixture") -> None:
        super().__init__()
        self.name = name
        self.components = nn.ModuleList(list(components))
        self.register_buffer("log_weights", torch.log_softmax(log_weights.detach().float(), dim=0))

    @property
    def n_components(self) -> int:
        return len(self.components)

    def component_log_prob(self, z: Tensor) -> Tensor:
        return torch.stack([component.log_prob(z) for component in self.components], dim=1)

    def log_prob(self, z: Tensor) -> Tensor:
        return torch.logsumexp(self.component_log_prob(z) + self.log_weights.to(z), dim=1)

    def responsibilities(self, z: Tensor) -> Tensor:
        return torch.softmax(self.component_log_prob(z) + self.log_weights.to(z), dim=1)


def dist_short_name(dist_module: Any) -> str:
    name = getattr(dist_module, "__name__", str(dist_module)).split(".")[-1]
    mapping = {
        "multivariate_norm": "mvn",
        "multivariate_genhyperbolic": "mghp",
        "multivariate_genskewt": "mgst",
        "multivariate_hyperbolic": "mvhb",
        "multivariate_norminvgauss": "mnig",
        "multivariate_t": "mvt",
        "multivariate_vargamma": "mvvg",
        "multivariate_skewnorm": "mvsn",
        "multivariate_skew_laplace": "mvsl",
    }
    return mapping.get(name, name)


def component_from_fmvmm(dist_module: Any, alpha: Any) -> Component:
    dist_name = dist_short_name(dist_module)
    if dist_name == "mvn":
        mu, sigma = alpha
        return MultivariateNormalComponent(mu, sigma)
    if dist_name == "mvt":
        mu, sigma, df = alpha
        return MultivariateTComponent(mu, sigma, df)
    if dist_name == "mvsn":
        mu, sigma, shape = alpha
        return SkewNormalComponent(mu, sigma, shape)
    if dist_name == "mvsl":
        mu, sigma, gamma = alpha
        return SkewLaplaceComponent(mu, sigma, gamma)
    if dist_name == "mghp":
        lmbda, chi, psi, mu, sigma, gamma = alpha
        return GeneralizedHyperbolicComponent(lmbda, chi, psi, mu, sigma, gamma, family=dist_name)
    if dist_name == "mgst":
        lmbda, chi, mu, sigma, gamma = alpha
        return GeneralizedHyperbolicComponent(lmbda, chi, 0.0, mu, sigma, gamma, family=dist_name)
    if dist_name == "mvhb":
        chi, psi, mu, sigma, gamma = alpha
        return GeneralizedHyperbolicComponent((len(mu) + 1.0) / 2.0, chi, psi, mu, sigma, gamma, family=dist_name)
    if dist_name == "mnig":
        chi, psi, mu, sigma, gamma = alpha
        return GeneralizedHyperbolicComponent(-0.5, chi, psi, mu, sigma, gamma, family=dist_name)
    if dist_name == "mvvg":
        lmbda, psi, mu, sigma, gamma = alpha
        return GeneralizedHyperbolicComponent(lmbda, 0.0, psi, mu, sigma, gamma, family=dist_name)
    raise NotImplementedError(f"No torch component adapter for FMVMM distribution {dist_name!r}.")


def torch_prior_from_fmvmm(
    weights: Any,
    alphas: Sequence[Any],
    dist_comb: Sequence[Any],
    name: str | None = None,
) -> TorchMixturePrior:
    weights_t = torch.as_tensor(np.asarray(weights, dtype=np.float64), dtype=torch.float32).log()
    components = [component_from_fmvmm(dist, alpha) for dist, alpha in zip(dist_comb, alphas)]
    family_names = [dist_short_name(dist) for dist in dist_comb]
    prior_name = name or "fmvmm_" + "_".join(family_names)
    return TorchMixturePrior(weights_t, components, name=prior_name)


@dataclass
class TorchPriorCheck:
    max_abs_diff: float
    mean_abs_diff: float
    n: int
