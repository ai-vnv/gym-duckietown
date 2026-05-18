"""Smoke tests for the scenario registry and factories (no GL, no env)."""
from __future__ import annotations

import pytest

from gym_duckietown.multiview import scenarios as scen


def test_builtin_scenarios_registered():
    names = {n for n, _ in scen.list_scenarios()}
    assert {"small_loop_bare", "small_loop_kfupm", "small_loop_jisr3"} <= names


def test_get_scenario_returns_same_object():
    a = scen.get_scenario("small_loop_jisr3")
    b = scen.get_scenario("small_loop_jisr3")
    assert a is b
    assert a.name == "small_loop_jisr3"


def test_get_scenario_unknown_raises():
    with pytest.raises(KeyError):
        scen.get_scenario("does-not-exist")


def test_register_scenario_rejects_duplicate():
    s = scen.Scenario(name="small_loop_bare", description="dup")
    with pytest.raises(ValueError):
        scen.register_scenario(s)


def test_jisr3_factories_produce_objects(stub_sim):
    s = scen.get_scenario("small_loop_jisr3")
    decals = s.floor_decals(stub_sim)
    bbs = s.billboards(stub_sim)
    sss = s.stop_signs(stub_sim)
    tls = s.traffic_lights(stub_sim)
    tlvs = s.traffic_light_visuals(stub_sim)

    assert len(decals) == 1
    assert len(bbs) == 1  # single billboard outside the south road
    assert len(sss) == 1  # single stop sign on the south straight
    assert len(tls) == 1
    assert len(tlvs) == 1
    # tuple unpacking shape: (x, z, y_base, w, h, rot)
    assert len(tlvs[0]) == 6


def test_bare_scenario_has_no_props(stub_sim):
    s = scen.get_scenario("small_loop_bare")
    assert s.floor_decals(stub_sim) == []
    assert s.billboards(stub_sim) == []
    assert s.stop_signs(stub_sim) == []
    assert s.traffic_lights(stub_sim) == []
    assert s.scenery(stub_sim) == []


def test_jisr3_scenery_returns_non_empty_with_expected_kinds(stub_sim):
    s = scen.get_scenario("small_loop_jisr3")
    items = s.scenery(stub_sim)
    assert len(items) > 0
    kinds = {item["kind"] for item in items}
    assert kinds == {"building", "house", "tree"}
    # every entry has the required fields
    for item in items:
        assert {"kind", "position"} <= item.keys()
        assert len(item["position"]) == 2
