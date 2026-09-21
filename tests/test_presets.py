"""Presets, Permalink-Angaben und Regler-Grenzen sind untereinander stimmig."""

import pytest

import dj_constants as C
import dj_presets as P
from dj_scenario import make_network


def test_every_preset_sets_every_control_within_bounds_and_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 5
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and p["net"] in C.NETS
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
        make_network(p["net"], p["side"], p["reach"], p["spread"], p["blocked"], p["walls"], p["seed"])


def test_setting_specs_and_kept_keys_are_consistent():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) and set(P.KEPT) <= set(P.SETTING_SPECS)
    for spec in P.SETTING_SPECS.values():
        assert spec.lo is None or spec.lo < spec.hi                                 # kein Regler mit gleichen Grenzen (Streamlit bricht ab)
    assert len({s.url_param for s in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_defaults_lie_inside_the_bounds_and_the_default_net_is_the_first_preset():
    assert C.SIDE_MIN <= C.DEFAULT_SIDE <= C.SIDE_MAX and C.REACH_MIN <= C.DEFAULT_REACH <= C.REACH_MAX and C.SPREAD_MIN <= C.DEFAULT_SPREAD <= C.SPREAD_MAX
    assert C.BLOCKED_MIN <= C.DEFAULT_BLOCKED <= C.BLOCKED_MAX and C.WALLS_MIN <= C.DEFAULT_WALLS <= C.WALLS_MAX
    assert C.DEFAULT_NET in C.NETS and C.DEFAULT_NET == list(C.PRESETS.values())[0]["net"]


def test_net_choice_caster_rejects_unknown_nets():
    cast = P.SETTING_SPECS["net_select"].caster
    assert cast("exchange") == "exchange"
    with pytest.raises(ValueError):
        cast("ring")
