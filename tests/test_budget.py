from rewardsmith.search import survivor_stretch
from rewardsmith.sim import budget
from rewardsmith.sim import scene as scene_mod

BALL_TO_GOAL = scene_mod.parse({
    "objects": [
        {"name": "ball", "kind": "ball", "color": "orange", "size": [0.15], "distance": [1, 2], "movable": True},
        {"name": "goal", "kind": "zone", "color": "green", "size": [0.5], "distance": [2.5, 3.5]},
    ],
    "target": "ball",
})
BLUE_BOX = scene_mod.parse({
    "objects": [{"name": "box", "kind": "box", "color": "blue", "size": [0.4, 0.4, 0.3], "distance": [2, 3]}],
    "target": "box",
})


def test_plans_by_scene_and_words():
    assert budget.plan("make robot move ball to a target destination", BALL_TO_GOAL).kind == "complex"
    assert budget.plan("walk to the blue box and stop beside it", BLUE_BOX).kind == "object"
    assert budget.plan("walk to the box, then turn around", BLUE_BOX).kind == "complex"
    simple = budget.plan("walk backwards", scene_mod.EMPTY)
    assert (simple.kind, simple.family, simple.steps) == ("simple", "gait", budget.BUDGETS["simple"])


def test_poses_drop_the_level_body_terms():
    for text in ("stand on your hind legs", "sit down", "jump in place", "roll over", "Make the robot dog stand on two legs"):
        assert budget.plan(text, scene_mod.EMPTY).family == "pose", text
    for text in ("walk backwards", "spin in place", "trot forward", "push the ball into the goal"):
        assert budget.plan(text, scene_mod.EMPTY).family == "gait", text


def test_survivor_stretch_keeps_the_round_cost():
    assert survivor_stretch(4, [2], 5) == 1.6  # (4 + 2 + 2 + 2) / 4 = 2.5 runs' work for 4 runs' budget
    assert survivor_stretch(8, [4, 2], 5) == 2.0
    assert survivor_stretch(4, [4], 5) == 1.0


def test_two_legs_is_a_pose_for_the_go1_but_the_gait_for_a_biped():
    assert budget.plan("walk forward on two legs", scene_mod.EMPTY).family == "pose"
    assert budget.plan("walk forward on two legs", scene_mod.EMPTY, bipedal=True).family == "gait"
    assert budget.plan("jump on two legs", scene_mod.EMPTY, bipedal=True).family == "pose"
