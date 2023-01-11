# Unit tests for inversion helper functions
import ildars.clustering.inversion as inversion

# We consider number n,m equal if abs(n-m) < EPSILON
EPSILON = 0.0000001


def test_is_point_on_finite_line():
    origin = [0, 0, 0]
    line1 = inversion.Line(origin, [2, 2, 2], None)
    assert inversion.is_point_on_finite_line(
        line1, [1, 1, 1]
    ), "Point [1,1,1] should be on line " + str(line1)
    assert not inversion.is_point_on_finite_line(
        line1, [3, 3, 3]
    ), "Point [3,3,3] should not be on line " + str(line1)
    assert not inversion.is_point_on_finite_line(
        line1, [2.00001, 2.00001, 2.00001]
    ), "Point [2.00001,2.00001,2.00001] should not be on line " + str(line1)
    assert not inversion.is_point_on_finite_line(
        line1, [1, 1, 0.9999]
    ), "Point [1,1,0.9999] should not be on line " + str(line1)


def test_bin_get_distance():
    # non intersecting cross
    l1 = inversion.Line([0, 0, -1], [0, 0, 1], None)
    l2 = inversion.Line([-1, 1, 0], [1, 1, 0], None)
    test_bin = inversion.Bin(l1)
    dist = test_bin.get_distance_to_line(l2).distance
    goal_dist = 1
    assert dist == goal_dist, (
        "Distance of "
        + str(l1)
        + " and "
        + str(l2)
        + " should be "
        + str(goal_dist)
        + " but was "
        + str(dist)
    )

    # intersecting cross
    l1 = inversion.Line([0, 0, -1], [0, 0, 1], None)
    l2 = inversion.Line([-1, 0, 0], [1, 0, 0], None)
    test_bin = inversion.Bin(l1)
    dist = test_bin.get_distance_to_line(l2).distance
    goal_dist = 0
    assert dist == goal_dist, (
        "Distance of "
        + str(l1)
        + " and "
        + str(l2)
        + " should be "
        + str(goal_dist)
        + " but was "
        + str(dist)
    )

    # non intersecting parallel lines
    l1 = inversion.Line([0, 0, -1], [0, 0, 1], None)
    l2 = inversion.Line([0, 1, -1], [0, 1, 1], None)
    test_bin = inversion.Bin(l1)
    dist = test_bin.get_distance_to_line(l2).distance
    goal_dist = 1
    assert dist == goal_dist, (
        "Distance of "
        + str(l1)
        + " and "
        + str(l2)
        + " should be "
        + str(goal_dist)
        + " but was "
        + str(dist)
    )

    # parallel lines which do not touch but are "aligned", i.e. are on the
    # same infinite line
    l1 = inversion.Line([0, 0, -2], [0, 0, -1], None)
    l2 = inversion.Line([0, 0, 1], [0, 0, 2], None)
    test_bin = inversion.Bin(l1)
    dist = test_bin.get_distance_to_line(l2).distance
    goal_dist = 2
    assert dist == goal_dist, (
        "Distance of "
        + str(l1)
        + " and "
        + str(l2)
        + " should be "
        + str(goal_dist)
        + " but was "
        + str(dist)
    )
    # TODO: more tests, cover all possible cases of relations of finite lines
    # in 3D.


def main():
    test_is_point_on_finite_line()
    test_bin_get_distance()
    print("All inversion tests passed successfully")


if __name__ == "__main__":
    main()
