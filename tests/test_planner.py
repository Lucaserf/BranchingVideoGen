from bvg.config import Costs, Profile, load_config, load_profile
from bvg.planner import Shot, default_shots, finish_after_last_choice, lock_times, plan_film

COSTS = Costs(keyframe=2, shot={"pan": 2, "standard": 25, "hero": 120}, finish=60)


def shots(acts=6, per_act=6):
    return [Shot(act=a, index=i) for a in range(acts) for i in range(per_act)]


def test_queue_waits_for_each_act_to_lock():
    # One act, one pan shot locked at t=90: done at 90 + 2 + 2 + 60, so 64 s after the last choice.
    assert finish_after_last_choice([Shot(0, 0)], COSTS, lock_times(1, 90)) == 64


def test_tight_budget_keeps_last_act_as_pans():
    # 90 s leaves no room to upgrade act 6, but earlier acts still render video during play.
    plan = plan_film(shots(), COSTS, Profile("x", target_wait_minutes=1.5, max_hero_shots=0), 90)
    assert all(s.render_type == "pan" for s in plan.shots if s.act == 5)
    assert plan.counts()["standard"] > 0
    assert plan.fits


def test_more_wait_means_more_video_shots():
    counts = [plan_film(shots(), COSTS, Profile("x", m, 36), 90).counts() for m in (3, 20, 60)]
    assert counts[0]["pan"] >= counts[1]["pan"] >= counts[2]["pan"]
    assert counts[2]["hero"] > counts[1]["hero"]
    assert all(plan_film(shots(), COSTS, Profile("x", m, 36), 90).fits for m in (3, 20, 60))


def test_hero_cap_is_respected_and_priority_goes_first():
    s = shots()
    s[7] = Shot(act=1, index=1, priority=1.0)
    plan = plan_film(s, COSTS, Profile("x", 60, max_hero_shots=1), 90)
    assert plan.counts()["hero"] == 1
    assert plan.shots[7].render_type == "hero"


def test_shipped_profiles_load_and_fit_with_estimates():
    config = load_config()
    from bvg.config import load_costs

    costs = load_costs(config)
    for name in config["profiles"]:
        plan = plan_film(default_shots(config), costs, load_profile(config, name),
                         config["film"]["seconds_per_choice"])
        assert len(plan.shots) == 36
        assert plan.fits, name
