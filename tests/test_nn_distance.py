import pytest
import torch
from torch import Tensor

from flair.nn.distance.hyperbolic import EPSILON, HyperbolicDistance, project


def _reference_squared_hyperbolic_distance(mat_1: Tensor, mat_2: Tensor) -> Tensor:
    projected_1 = project(mat_1)
    projected_2 = project(mat_2)
    time_product = projected_1[:, None, 0] * projected_2[None, :, 0]
    spatial_product = (projected_1[:, None, 1:] * projected_2[None, :, 1:]).sum(dim=-1)
    return torch.acosh((time_product - spatial_product).clamp(min=1 + EPSILON)).pow(2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n_features", [1, 2, 7])
def test_hyperbolic_distance_matches_projected_lorentz_product(dtype, n_features):
    generator = torch.Generator().manual_seed(1729)
    mat_1 = torch.randn(3, n_features, generator=generator, dtype=dtype) / 2
    mat_2 = torch.randn(4, n_features, generator=generator, dtype=dtype) / 2

    distances = HyperbolicDistance()(mat_1, mat_2)

    assert distances.shape == (3, 4)
    assert distances.dtype == dtype
    torch.testing.assert_close(distances, _reference_squared_hyperbolic_distance(mat_1, mat_2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hyperbolic_distance_identical_points_retain_numerical_floor(dtype):
    points = torch.tensor([[3.0, 0.0], [0.0, 0.0], [0.5, -0.25]], dtype=dtype)

    distances = HyperbolicDistance()(points, points)
    numerical_floor = torch.acosh(points.new_tensor(1 + EPSILON)).pow(2)

    torch.testing.assert_close(distances.diagonal(), numerical_floor.expand(len(points)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hyperbolic_distance_selects_identical_prototype(dtype):
    query = torch.tensor([[3.0, 0.0]], dtype=dtype)
    prototypes = torch.tensor([[0.1, 0.0], [3.0, 0.0]], dtype=dtype)

    distances = HyperbolicDistance()(query, prototypes)

    assert distances.argmin(dim=-1).item() == 1
    assert distances[0, 1] < distances[0, 0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hyperbolic_distance_is_invariant_to_spatial_coordinate_permutation(dtype):
    mat_1 = torch.tensor([[0.4, 0.1, -0.3], [1.2, -0.4, 0.6]], dtype=dtype)
    mat_2 = torch.tensor([[-0.2, 0.5, 0.7], [0.4, 0.1, -0.3], [0.1, -0.8, 0.2]], dtype=dtype)
    permutation = [2, 0, 1]
    distance = HyperbolicDistance()

    torch.testing.assert_close(distance(mat_1, mat_2), distance(mat_1[:, permutation], mat_2[:, permutation]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hyperbolic_distance_gradients_match_projected_lorentz_product(dtype):
    mat_1 = torch.tensor([[0.5, -0.25], [0.75, 0.5]], dtype=dtype, requires_grad=True)
    mat_2 = torch.tensor([[-0.25, 0.5], [0.25, -0.75], [0.75, -0.5]], dtype=dtype, requires_grad=True)

    gradients = torch.autograd.grad(HyperbolicDistance()(mat_1, mat_2).sum(), (mat_1, mat_2))
    reference_gradients = torch.autograd.grad(
        _reference_squared_hyperbolic_distance(mat_1, mat_2).sum(), (mat_1, mat_2)
    )

    for gradient, reference_gradient in zip(gradients, reference_gradients):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, reference_gradient)


def test_hyperbolic_distance_is_symmetric():
    mat_1 = torch.tensor([[0.5, -0.25], [0.75, 0.5]], dtype=torch.float64)
    mat_2 = torch.tensor([[-0.25, 0.5], [0.25, -0.75], [0.75, -0.5]], dtype=torch.float64)
    distance = HyperbolicDistance()

    torch.testing.assert_close(distance(mat_1, mat_2), distance(mat_2, mat_1).t())


def test_hyperbolic_distance_passes_gradcheck():
    mat_1 = torch.tensor([[0.5, -0.25], [0.75, 0.5]], dtype=torch.float64, requires_grad=True)
    mat_2 = torch.tensor([[-0.25, 0.5], [0.25, -0.75], [0.75, -0.5]], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(HyperbolicDistance(), (mat_1, mat_2))
