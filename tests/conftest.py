"""Tiny numerical tests run faster without a large CPU thread pool."""

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)
