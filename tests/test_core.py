import numpy as np
import pytest

from ai.player_behavior import BehaviorStore, PlayerBehaviorModel
from environment import maps, observations
from environment.config import N_ACTIONS, WALL, EnvConfig
from environment.neurohunt_env import NeuroHuntEnv
from environment.world import World

CH1 = "assets/maps/chapter1.json"


@pytest.mark.parametrize("seed", range(25))
def test_procedural_maps_are_connected(seed):
    m = maps.generate(seed)
    w = World(m)
    for target in [m.keycard, m.terminal, *m.exits, *m.lockers, m.hunter_start]:
        assert w.path_dist(m.player_start, target) < 10_000


def test_eval_seeds_disjoint_from_training():
    env = NeuroHuntEnv(eval_mode=True)
    env.reset(seed=0)
    assert env.world.map.name.startswith("proc-")
    assert int(env.world.map.name.split("-")[1]) >= maps.EVAL_SEED_OFFSET


def test_chapter1_map_valid():
    m = maps.load_json(CH1)
    assert m.grid.shape == (15, 21)
    assert m.keycard and m.terminal and len(m.exits) == 2 and m.player_start and m.hunter_start


def test_hunter_cannot_see_through_walls():
    m = maps.load_json(CH1)
    w = World(m)
    w.hunter, w.hunter_dir = (2, 2), (1, 0)
    w.player = (7, 2)            # behind the wall at x=5
    w.update_perception()
    assert m.grid[2, 5] == WALL
    assert not w.player_visible


def test_hunter_sees_in_cone():
    m = maps.load_json(CH1)
    w = World(m)
    w.hunter, w.hunter_dir, w.player = (1, 2), (1, 0), (4, 2)
    w.update_perception()
    assert w.player_visible
    w.hunter_dir = (-1, 0)       # facing away
    w.update_perception()
    assert not w.player_visible


def test_hidden_player_only_found_by_adjacent_scan():
    m = maps.load_json(CH1)
    w = World(m)
    w.player = (2, 3)            # locker
    w.toggle_hide()
    w.hunter, w.hunter_dir = (2, 1), (0, 1)
    w.update_perception()
    assert not w.player_visible
    w.hunter = (2, 2)
    w.hunter_scanning = True
    w.update_perception()
    assert w.player_visible


def test_env_api_and_observation_bounds():
    env = NeuroHuntEnv()
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    for _ in range(400):
        obs, r, te, tr, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
        if te or tr:
            assert info["outcome"] in ("caught", "escaped", "timeout")
            obs, _ = env.reset()
    assert env.action_space.n == N_ACTIONS


def test_observation_never_leaks_hidden_player():
    env = NeuroHuntEnv()
    env.reset(seed=3)
    w = env.world
    w.player_visible = False
    obs = observations.build(w, env.behavior, 0, 300)
    assert obs["grid"][3].sum() == 0


def test_behavior_model_learns_favourite_locker(tmp_path):
    m = maps.load_json(CH1)
    b = PlayerBehaviorModel()
    for _ in range(4):
        b.record_hide(m.lockers[0])
        b.end_episode()
    b.record_hide(m.lockers[3])
    scores = b.hide_scores(m.lockers)
    assert max(scores, key=scores.get) == m.lockers[0]
    w = World(m)
    target = w.resolve_target(8, b)   # check_hiding_spot
    assert target == m.lockers[0]
    store = BehaviorStore(tmp_path / "t.sqlite")
    store.save("p", m.name, b)
    b2 = store.load("p", m.name)
    assert b2.hide_scores(m.lockers) == pytest.approx(scores)


def test_predicted_exit_feeds_intercept():
    m = maps.load_json(CH1)
    b = PlayerBehaviorModel()
    for _ in range(3):
        b.record_exit(m.exits[1])
    w = World(m)
    assert w.resolve_target(7, b) == m.exits[1]


def test_config_roundtrip():
    cfg = EnvConfig.from_json("experiments/configs/ppo_cnn_v1.json")
    assert cfg.rewards.catch == 100.0 and cfg.max_steps == 300
