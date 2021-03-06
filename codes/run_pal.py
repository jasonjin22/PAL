import read_data as rd
import plot_pareto as pp

from training_set import TrainingSet
from point import Point
from state import State
from state import sampled_set_to_numpy_array


def run(mode):
    model_mode = mode
    print("model_mode: ", model_mode)
    # 0. hyper-parameters
    epsilon = 0.1
    delta = 0.1

    # 1. load data from CSV files
    # noc has small y values
    X_full, X, Y = rd.read_from_csv("snw")
    true_pareto = pp.find_pareto(Y)
    range1 = max(Y[:, 0]) - min(Y[:, 0])
    range2 = max(Y[:, 1]) - min(Y[:, 1])
    # 2. build the training set, points and state
    training_set = TrainingSet(X_full, X, Y)
    state = State(training_set, epsilon, len(X_full), delta)
    for i in range(len(X_full)):
        the_point = Point(X_full[i], X[i], Y[i])
        state.add_point(the_point)
    # 3. do the pareto front finding
    if model_mode == 0:
        state.decide_hyper_para_normal()  # randomly sample some points and decide the hyper parameters
    else:
        state.decide_hyper_para_coregion()

    for i in range(40):
        state.update_beta_t()
        state.sampling()
        print("state.sampled_points)", len(state.sampled_points))
        state.modeling(model_mode)
        if i >= 5:
            state.discard()
            state.covering()
            # if state.check_stop_criteria():
            #     break
        state.compute_error(true_pareto, range1, range2)
    pp.plot_pareto(Y, state.training_set, sampled_set_to_numpy_array(state.P_t))
    pp.plot_error(state.error)
    print(state.error)
    return state.error

if __name__ == '__main__':
    iters = 10
    error_list_mode_0 = []
    error_list_mode_1 = []
    mean_error_0 = []
    mean_error_1 = []
    for i in range(iters):
        error_list_mode_0.append(run(0))
        error_list_mode_1.append(run(1))
    for i in range(len(error_list_mode_0[0])):
        tem = 0
        for j in range(iters):
            tem += error_list_mode_0[j][i]
        mean_error_0.append(tem/iters)
    for i in range(len(error_list_mode_1[0])):
        tem = 0
        for j in range(iters):
            tem += error_list_mode_1[j][i]
        mean_error_1.append(tem/iters)
    pp.plot_error_both_mode(mean_error_0, mean_error_1)
    print("0\n", mean_error_0)
    print("1\n", mean_error_1)