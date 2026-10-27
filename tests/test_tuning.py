import json
from dataclasses import replace

import pytest
import torch

from memfast import Config, Simulator
from memfast.simulator import default_schedule, schedule_grid

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_small_workspace_uses_admitted_reference(tmp_path):
    config = Config(
        activation_bits=4, weight_bits=4, array_size=(8, 8), rows_per_read=4, adc_bits=(6, 7, 8, 9)
    )
    torch.manual_seed(135)
    weight, x = torch.randn(19, 8), torch.randn(3, 8, device="cuda:0")
    sim = Simulator(config, mode="tuned", device="cuda:0", cache_dir=tmp_path, workspace_mb=0.002)
    array = sim.map(weight, name="budget")
    from memfast.cuda import ResourceError
    from memfast.quantization import input_tiles

    q, _ = input_tiles(x, config)
    with pytest.raises(ResourceError, match="workspace"):
        sim._admit(array, q, default_schedule(config))
    expected = Simulator(config, device="cuda:0").map(weight, name="budget")(x)
    assert torch.equal(array(x), expected)
    record = sim.records[0]
    assert record["complete"] and record["selected"]["arrays"] == 1
    assert len(record["candidates"]) == len(list(schedule_grid(config, 3, array.p)))
    fresh = Simulator(config, mode="tuned", device="cuda:0", cache_dir=tmp_path, workspace_mb=0.002)
    assert torch.equal(fresh.map(weight, name="budget")(x), expected)
    assert fresh.records[0]["cache_hit"] and not fresh.records[0]["candidates"]


@pytest.mark.parametrize("failure", ["oom", "resource"])
def test_cached_schedule_resource_failure_retraverses(tmp_path, monkeypatch, failure):
    from memfast.cuda import ResourceError

    config = Config(activation_bits=2, weight_bits=2, array_size=(8, 8))
    torch.manual_seed(136)
    weight, x = torch.randn(13, 8), torch.randn(2, 8, device="cuda:0")
    first = Simulator(config, mode="tuned", device="cuda:0", cache_dir=tmp_path)
    expected = first.map(weight, name="cache")(x)
    fresh = Simulator(config, mode="tuned", device="cuda:0", cache_dir=tmp_path)
    array = fresh.map(weight, name="cache")
    original = array._execute
    calls = 0

    def fail_cached(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            if failure == "oom":
                raise torch.cuda.OutOfMemoryError("injected cached allocation failure")
            raise ResourceError("injected cached resource failure")
        return original(*args)

    monkeypatch.setattr(array, "_execute", fail_cached)
    assert torch.equal(array(x), expected)
    assert not fresh.records[0]["cache_hit"]
    assert fresh.records[0]["complete"] and fresh.records[0]["candidates"]
    assert array._event == 1


def test_corrupt_cache_and_adc_fingerprint(tmp_path):
    config = Config(activation_bits=2, weight_bits=2, array_size=(8, 8), adc_bits=(6, 7))
    weight, x = torch.randn(9, 8), torch.randn(2, 8, device="cuda:0")
    sim = Simulator(config, mode="tuned", device="cuda:0", cache_dir=tmp_path)
    matrix = sim.map(weight)
    key, payload = sim._key(matrix, x)
    path = tmp_path / f"{key}.json"
    path.write_text("{broken", encoding="utf-8")
    assert torch.isfinite(matrix(x)).all()
    assert sim.records[0]["complete"] and not sim.records[0]["cache_hit"]
    assert json.loads(path.read_text())["complete"]
    other = Simulator(
        replace(config, adc_bits=(7, 6)), mode="tuned", device="cuda:0", cache_dir=tmp_path
    )
    other_key, other_payload = other._key(other.map(weight), x)
    assert other_key != key and other_payload != payload
