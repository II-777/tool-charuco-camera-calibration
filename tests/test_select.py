from charuco_calib.select import Pose, covered_cells, is_novel, is_steady


def test_steady_requires_a_still_run():
    moving = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0), (20.0, 0.0), (25.0, 0.0)]
    assert is_steady(moving, tolerance_px=4, count=6) is False
    still = [(10.0, 10.0), (11.0, 10.0), (10.0, 11.0), (10.5, 10.5), (11.0, 11.0), (10.0, 10.2)]
    assert is_steady(still, tolerance_px=4, count=6) is True


def test_novel_rejects_a_repeat_pose():
    first = Pose((100.0, 100.0), frozenset({(1, 1)}), 200.0)
    same = Pose((110.0, 100.0), frozenset({(1, 1)}), 200.0)
    moved = Pose((200.0, 100.0), frozenset({(2, 1)}), 200.0)
    tilted = Pose((100.0, 100.0), frozenset({(1, 1)}), 280.0)
    assert is_novel(same, [first], min_shift_px=30) is False
    assert is_novel(moved, [first], min_shift_px=30) is True
    assert is_novel(tilted, [first], min_shift_px=30) is True


def test_coverage_is_the_union_of_cells():
    poses = [
        Pose((0.0, 0.0), frozenset({(0, 0), (1, 0)}), 40.0),
        Pose((10.0, 10.0), frozenset({(1, 0), (2, 2)}), 80.0),
    ]
    assert covered_cells(poses) == frozenset({(0, 0), (1, 0), (2, 2)})
