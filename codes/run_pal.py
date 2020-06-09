import read_data as rd


from training_set import TrainingSet
from point import Point
from state import State


def main():
    # 0. hyper-parameters
    epsilon = 0.1
    delta = 0.1

    # 1. load data from CSV files
    # noc has small y values
    X_full, X, Y = rd.read_from_csv("snw")
    # 2. build the training set, points and state
    training_set = TrainingSet(X_full, X, Y)
    state = State(training_set, epsilon, len(X_full), delta)
    for i in range(len(X_full)):
        the_point = Point(X_full[i], X[i], Y[i])
        state.add_point(the_point)
    # 3. do the pareto front finding
    state.decide_hyper_para()  # randomly sample some points and decide the hyper parameters
    for i in range(30):
        state.update_beta_t()
        state.sampling()
        print("state.sampled_points)", len(state.sampled_points))
        state.modeling()
        if i >= 5:
            state.discard()
            state.covering()

if __name__ == '__main__':
    main()