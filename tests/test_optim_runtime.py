"""Runtime boundary checks for the compiled NorMuonH bucket update."""

from copy import deepcopy

import torch

from delta_feedback_experiment.optim import NorMuonH, _normuonh_bucket_step


def test_bucket_compilation_preserves_updates_and_tensor_storage():
    """Trace the mutation boundary and compare with independent matrix steps."""
    torch.manual_seed(29)
    parameters = [torch.nn.Parameter(torch.randn(7, 5)) for _ in range(3)]
    references = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    optimizers = [NorMuonH([p], lr=0.03) for p in references]
    momenta = [torch.zeros_like(p) for p in parameters]
    row_moments = [torch.zeros(p.shape[0], 1) for p in parameters]
    radii = [p.detach().norm() for p in parameters]
    storage = [t.data_ptr() for t in (*parameters, *momenta, *row_moments, *radii)]
    compiled = torch.compile(_normuonh_bucket_step, backend="eager", fullgraph=True)

    for lr in (0.03, 0.0, 0.01):
        gradients = [torch.randn_like(p) for p in parameters]
        with torch.no_grad():
            compiled(
                parameters,
                gradients,
                momenta,
                row_moments,
                radii,
                torch.tensor(lr),
                0.95,
                0.95,
                1e-8,
                5,
            )
        for index, (reference, optimizer) in enumerate(zip(references, optimizers)):
            reference.grad = gradients[index].clone()
            optimizer.param_groups[0]["lr"] = lr
            optimizer.step()
            torch.testing.assert_close(parameters[index], reference)
            torch.testing.assert_close(
                momenta[index], optimizer.state[reference]["momentum"]
            )
            torch.testing.assert_close(
                row_moments[index], optimizer.state[reference]["row_moment"]
            )
            torch.testing.assert_close(parameters[index].norm(), radii[index])

    assert [
        t.data_ptr() for t in (*parameters, *momenta, *row_moments, *radii)
    ] == storage


def test_bucket_optimizer_preserves_absent_gradients_and_resume():
    """Changing active sets must neither update missing state nor break resume."""
    torch.manual_seed(31)
    parameters = [
        torch.nn.Parameter(torch.randn(*shape)) for shape in ((7, 5), (7, 5), (5, 7))
    ]
    optimizer = NorMuonH(parameters, lr=0.02)
    initial = [parameter.detach().clone() for parameter in parameters]
    parameters[0].grad = torch.randn_like(parameters[0])
    optimizer.step()
    assert set(optimizer.state) == {parameters[0]}
    for index in (1, 2):
        assert torch.equal(parameters[index], initial[index])

    restored_parameters = [
        torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters
    ]
    restored = NorMuonH(restored_parameters, lr=0.02)
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    untouched = deepcopy(optimizer.state[parameters[0]])
    saved_weight = parameters[0].detach().clone()

    for index, (parameter, other) in enumerate(zip(parameters, restored_parameters)):
        gradient = None if index == 0 else torch.randn_like(parameter)
        parameter.grad = gradient
        other.grad = None if gradient is None else gradient.clone()
    optimizer.step()
    restored.step()

    assert torch.equal(parameters[0], saved_weight)
    for name, value in untouched.items():
        assert torch.equal(optimizer.state[parameters[0]][name], value)
    assert set(optimizer.state_dict()) == {"state", "param_groups"}
    for parameter, other in zip(parameters, restored_parameters):
        assert torch.equal(parameter, other)
        assert set(optimizer.state[parameter]) == {"momentum", "row_moment", "radius"}
        for name, value in optimizer.state[parameter].items():
            assert torch.equal(value, restored.state[other][name])


def test_bounded_expert_buckets_match_independent_matrix_updates():
    torch.manual_seed(87)
    parameters = [torch.nn.Parameter(torch.randn(7, 5)) for _ in range(7)]
    references = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    optimizer = NorMuonH(parameters, lr=0.02, max_bucket_elements=70)
    separate = [NorMuonH([p], lr=0.02) for p in references]
    assert [len(batch) for batch in optimizer._batches(parameters)] == [2, 2, 2, 1]
    for _ in range(3):
        for parameter, reference in zip(parameters, references):
            gradient = torch.randn_like(parameter)
            parameter.grad = gradient
            reference.grad = gradient.clone()
        optimizer.step()
        for reference, single in zip(references, separate):
            single.step()
        for parameter, reference in zip(parameters, references):
            torch.testing.assert_close(parameter, reference)
