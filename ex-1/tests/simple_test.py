from solver import solver


def simple_test(QFunc, alphas, n, min_x=-100, max_x=100):
    """
    Perform a simple test to check if the solver works.

    Args:
        QFunc (class): The QFunc class representing the objective function.
        alphas (list): A list of alpha values to test.
        n (int): The number of dimensions in the function (default is 10).
        min_x (float, optional): The minimum value for input x (default is -100).
        max_x (float, optional): The maximum value for input x (default is 100).
    """
    print("\tstarting simple test to check if solver works")
    for alpha in alphas:
        print(f"\t• testing for alpha = {alpha}")
        objective_func = QFunc(alpha=alpha, min_x=min_x, max_x=max_x, n=n)
        starting_point = objective_func.gen_input()
        # solved_data = solver(objective_func.get_func, np.ones(n))
        solved_data = solver(objective_func.get_func, starting_point)

        print(f"\t\tbeta={solved_data.beta}")
        print(f"\t\tstarting point={solved_data.starting_point}")
        print(f"\t\tsteps={solved_data.get_run_steps()}")
        print(f"\t\tq(xt)={solved_data.get_final_value()}\n")
