import numpy as np

from stonesoup.functions import jacobian, gm_reduce_single


def test_jacobian():
    """ jacobian function test """

    # State related variables
    state_mean = np.array([[3.0], [1.0]])

    def f(x):
        return np.array([[1, 1], [0, 1]])@x

    jac = jacobian(f, state_mean)
    jac = jac  # Stop flake8 unused warning


def test_jacobian2():
    """ jacobian function test """

    # Sample functions to compute Jacobian on
    def fun(x):
        """ function for testing scalars i.e. scalar input, scalar output"""
        return 2*x**2

    def fun1d(vec):
        """ test function with vector input, scalar output"""
        out = 2*vec[0]+3*vec[1]
        return out

    def fun2d(vec):
        """ test function with 2d input and 2d output"""
        out = np.empty((2, 1))
        out[0] = 2*vec[0]**2 + 3*vec[1]**2
        out[1] = 2*vec[0]+3*vec[1]
        return out
        x = 3
        jac = jacobian(fun, x)
        assert jac == 4*x

    x = np.array([[1], [2]])
    # Tolerance value to use to test if arrays are equal
    tol = 1.0e-9

    jac = jacobian(fun1d, x)
    T = np.array([2.0, 3.0])

    FOM = np.where(np.abs(jac-T) > tol)
    # Check # of array elements bigger than tol
    assert len(FOM[0]) == 0

    jac = jacobian(fun2d, x)
    T = np.array([[4.0*x[0], 6*x[1]],
                  [2, 3]])
    FOM = np.where(np.abs(jac - T) > tol)
    # Check # of array elements bigger than tol
    assert len(FOM[0]) == 0

    return


def test_gm_reduce_single():

    means = np.array([[1, 2], [3, 4], [5, 6]], np.float)
    covars = np.array([[[1, 1], [1, 0.7]], [[1.2, 1.4], [1.3, 2]],
                       [[2, 1.4], [1.2, 1.2]]], np.float)
    weights = np.array([1, 2, 5], np.float)

    mean, covar = gm_reduce_single(means, covars, weights)

    assert np.array_equal(mean, np.array([[4], [5]], np.float))
    assert np.array_equal(covar, np.array([[5.675, 5.35], [5.2, 5.3375]],
                                          np.float))
