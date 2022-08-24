import numpy
import numpy as np

from src.genetic.genetic import get_number_of_weights, verify_same_number_of_weights, get_row_and_index, \
    parent_child_weights, random_crossover_of_weights, get_sorted_crossover_points, crossover_weights, mutate_weights, \
    unravel_weights


def test_get_number_of_weights():
    for _ in range(10000):
        arrays = [numpy.array([]), numpy.array([])]
        assert get_number_of_weights(arrays) == 0
        arrays = [numpy.array([1, 2, 3]), numpy.array([4, 5])]
        assert get_number_of_weights(arrays) == 5


def test_verify_same_number_of_weights():
    for _ in range(10000):
        arrays = [numpy.array([1, 2, 3]), numpy.array([4, 5])]
        assert verify_same_number_of_weights(arrays, arrays) == 5
        try:
            verify_same_number_of_weights(arrays, numpy.array([1, 2, 3]))
            assert False
        except ValueError:
            pass


def test_get_row_and_index():
    for _ in range(10000):
        arrays = [numpy.array([1, 2, 3]), numpy.array([4, 5])]
        assert get_row_and_index(arrays, 0) == (0, 0)
        assert get_row_and_index(arrays, 1) == (0, 1)
        assert get_row_and_index(arrays, 2) == (0, 2)
        assert get_row_and_index(arrays, 3) == (1, 0)
        assert get_row_and_index(arrays, 4) == (1, 1)


def test_parent_child_weights():
    for _ in range(10000):
        arrays = [numpy.array([1, 2, 3]), numpy.array([4, 5])]
        arrays2 = [numpy.array([6, 7]), numpy.array([8])]
        copy, copy2 = parent_child_weights(arrays, arrays2)
        assert all(arrays[0] == copy[0])
        assert all(arrays[1] == copy[1])
        assert all(arrays2[0] == copy2[0])
        assert all(arrays2[1] == copy2[1])
        arrays[0][0] = 0
        arrays2[0][0] = 0
        assert any(arrays[0] != copy[0])
        assert any(arrays2[0] != copy2[0])


def test_random_crossover_of_weights():
    for _ in range(10000):
        arrays = [numpy.array([1, 2, 3]), numpy.array([4, 5])]
        arrays2 = [numpy.array([6, 7]), numpy.array([8])]
        arrays3 = random_crossover_of_weights(arrays, arrays2)
        for a in arrays3:
            for e in a:
                found = False
                for a2 in arrays:
                    for e2 in a2:
                        if e2 == e:
                            found = True
                for a2 in arrays2:
                    for e2 in a2:
                        if e2 == e:
                            found = True
                assert found
        equal = True
        arrays = unravel_weights(arrays)
        arrays3 = unravel_weights(arrays3)
        for a in arrays3:
            if a not in arrays:
                equal = False
        assert not equal
        equal = True
        for a in arrays3:
            if a not in arrays:
                equal = False
        assert not equal


def test_get_sorted_crossover_points():
    for _ in range(10000):
        number_of_weights = 20
        n_points = 10
        ps = get_sorted_crossover_points(number_of_weights, n_points)
        assert len(ps) == n_points
        for p in ps:
            assert p > -1
            assert p < number_of_weights


def test_crossover_weights():
    arrays = [
        numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        numpy.array([11, 12, 153, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    ]
    arrays2 = [
        numpy.array([31, 32, 33, 34, 35, 36, 37, 38, 39, 310]),
        numpy.array([311, 312, 315, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325])
    ]
    arrays3 = crossover_weights(arrays, arrays2, n_points=0)
    arrays = list(unravel_weights(arrays))
    arrays3 = list(unravel_weights(arrays3))
    for a in arrays3:
        assert a in arrays
    n_points = 2
    for _ in range(10000):
        arrays = [
            numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            numpy.array([11, 12, 153, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
        ]
        arrays3 = crossover_weights(arrays, arrays2, n_points)
        flat_array = unravel_weights(arrays)
        flat_array2 = unravel_weights(arrays2)
        flat_array3 = unravel_weights(arrays3)
        assert len(flat_array3) == len(flat_array)
        in_first = True
        i = 0
        for j, e in enumerate(flat_array3):
            if in_first:
                if e != flat_array[j]:
                    in_first = False
                    i += 1
            else:
                if e != flat_array2[j]:
                    in_first = True
                    i += 1
        assert i < (n_points + 1)


def test_mutate_weights():
    arrays = [numpy.array([1, 2, 3]), numpy.array([4, 5])]
    n_points = 2
    for _ in range(10000):
        arrays2 = mutate_weights(arrays, n_points)
        flat_array = unravel_weights(arrays)
        flat_array2 = unravel_weights(arrays2)
        assert flat_array != flat_array2
        mutations = 0
        for j, e in enumerate(flat_array2):
            if e != flat_array[j]:
                mutations += 1
        assert n_points >= mutations > 1


test_get_number_of_weights()
test_verify_same_number_of_weights()
test_get_row_and_index()
test_parent_child_weights()
test_random_crossover_of_weights()
test_get_sorted_crossover_points()
test_crossover_weights()
test_mutate_weights()