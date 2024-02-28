from autograd import grad

max_steps = 1000
beta = 0.01


def solver(object, x0, method="gradient-descent", verbose=True):
    if method == "gradient-descent":
        current_value = x0
        steps_counter = 0

        while steps_counter <= max_steps:
            grad_f = grad(object)
            diff = beta * grad_f(current_value)
            current_value -= diff

            if verbose:
                print(f"step {steps_counter:2} q(xt)={object(current_value)}")
                # print(f"\n{current_value}")
            steps_counter += 1
    else:
        pass  # not yet implemented
