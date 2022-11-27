# Unit tests for inversion helper functions
import inversion

def test_is_point_on_finite_line():
    origin = [0,0,0]
    line1 = inversion.Line(origin, [2,2,2])
    assert inversion.is_point_on_finite_line(line1, [1,1,1]), "Point [1,1,1] should be on line " + str(line1)
    assert not inversion.is_point_on_finite_line(line1, [3,3,3]), "Point [3,3,3] should not be on line " + str(line1)
    assert not inversion.is_point_on_finite_line(line1, [2.00001,2.00001,2.00001]), "Point [2.00001,2.00001,2.00001] should not be on line " + str(line1)
    assert not inversion.is_point_on_finite_line(line1, [1,1,0.9999]), "Point [1,1,0.9999] should not be on line " + str(line1)

if __name__ == "__main__":
    test_is_point_on_finite_line()
    print("All tests passed successfully")