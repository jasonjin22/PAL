import random
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.dpi'] = 150


def plot_points(X_axis, Y_axis):
    """
    copyright https://www.cnblogs.com/cloud-ken/p/11323955.html
    plot all the points
    """
    plt.scatter(X_axis, Y_axis, s=10, c="blue", alpha=0.25)
    plt.show()


def find_pareto(Y):
    sorted_Y = sorted(Y, key=lambda x: x[0], reverse=True)
    new_Y = []
    curr_max = float('-inf')
    for i in range(len(sorted_Y)):
        if sorted_Y[i][1] > curr_max:
            curr_max = sorted_Y[i][1]
            new_Y.append(sorted_Y[i])
        else:
            pass
    return np.array(new_Y)


def plot_points_with_pareto(X_all, Y_all, X_pareto, Y_pareto):
    """
    plot all the points with the pareto frontier highlighted
    """
    plt.scatter(X_all, Y_all, s=10, c="blue", alpha=0.25)
    plt.scatter(X_pareto, Y_pareto, s=40, c="red", alpha=0.5)
    x = [X_pareto[0], X_pareto[0]]
    y = [Y_pareto[0]]
    for i in X_pareto[1:]:
        x.append(i)
        x.append(i)
    for i in Y_pareto:
        y.append(i)
        y.append(i)
    plt.plot(x, y[:-1], color='r', alpha=0.25)
    plt.show()


def plot_sampled_points(training_set, sampled_points, sampling_point ,U_t):
    plt.scatter(training_set.Y[:, 0], training_set.Y[:, 1], s=10, c="blue", alpha=0.25)
    plt.scatter(sampled_points[1], sampled_points[2], s=10, c="red", alpha=1)
    plt.scatter(sampling_point.y[0], sampling_point.y[1], s=50, c="orange", alpha=1, marker='X')
    plot_R_t(sampling_point)
    # random_sample_and_plot_R_t(U_t)
    # random_sample_and_plot_R_t(U_t)
    plt.show()


def plot_discarded_points(training_set, sampled_points, sampling_point , discarded_points, P_t, U_t):
    plt.scatter(training_set.Y[:, 0], training_set.Y[:, 1], s=10, c="blue", alpha=0.25)
    plt.scatter(sampled_points[1], sampled_points[2], s=10, c="red", alpha=1)
    plt.scatter(discarded_points[1], discarded_points[2], s=10, c="black", alpha=1)
    plt.scatter(P_t[1], P_t[2], s=50, c="green", alpha=1)
    plt.scatter(sampling_point.y[0], sampling_point.y[1], s=50, c="orange", alpha=1, marker='X')
    random_sample_and_plot_R_t(U_t)
    plot_R_t(sampling_point)

    plt.show()


def plot_R_t(point):
    if point.R_t is None:
        print("The R_t is None")
    else:
        # print(point.R_t, point.mu_1, point.sigma_1, point.mu_2, point.sigma_2)
        L, R = point.R_t
        left_bottom = np.squeeze(L)
        right_upper = np.squeeze(R)

        x = [left_bottom[0], right_upper[0]]
        y = [left_bottom[1], right_upper[1]]
        plt.plot(point.mu_1, point.mu_2, c="cyan", marker='o', markerfacecolor='none')
        plt.plot(x, y, color='r', alpha=0.5)


def random_sample_and_plot_R_t(U_t):
    random_point = random.sample(U_t, 1)[0]
    plt.scatter(random_point.y[0], random_point.y[1], s=80, c="purple", alpha=1, marker='X')
    plot_R_t(random_point)


def plot_pessimistic_set_and_discarded_points(training_set, pessimistic_set, discard_set, discarded_points, set1,
                                              sampling_point):
    plt.scatter(training_set.Y[:, 0], training_set.Y[:, 1], s=10, c="blue", alpha=0.25)
    plt.scatter(pessimistic_set[1], pessimistic_set[2], s=10, c="red", alpha=1)
    plt.scatter(discarded_points[1], discarded_points[2], s=10, c="black", alpha=1)

    plt.scatter(discard_set[1], discard_set[2], s=10, c="orange", alpha=1)
    for point in set1:
        plot_R_t(point)
    plt.scatter(sampling_point.y[0], sampling_point.y[1], s=80, c="purple", alpha=1, marker='X')
    plt.show()


def plot_covered(training_set, P_t, the_point, small_point_set):
    plt.scatter(training_set.Y[:, 0], training_set.Y[:, 1], s=10, c="blue", alpha=0.25)
    plt.scatter(P_t[1], P_t[2], s=80, c="springgreen", alpha=1, marker="^")
    plt.scatter(the_point.y[0], the_point.y[1], s=30, c="orange", alpha=1)
    plt.scatter(small_point_set[1], small_point_set[2], s=10, c="red", alpha=1)
    plot_R_t(the_point)
    plt.show()


def plot_pareto(Y, training_set, P_t):
    plt.scatter(training_set.Y[:, 0], training_set.Y[:, 1], s=10, c="blue", alpha=0.25)
    plt.scatter(P_t[1], P_t[2], s=80, c="springgreen", alpha=1, marker="^")
    print(P_t)
    x = P_t[1]
    y = P_t[2]
    nl = []
    for i in range(len(x)):
        nl.append((x[i], y[i]))
    print(nl)
    nl = sorted(nl, key=lambda x: x[1])
    print(nl)
    X_pareto = []
    Y_pareto = []
    for i in nl:
        X_pareto.append(i[0])
        Y_pareto.append(i[1])
    x = [X_pareto[0], X_pareto[0]]
    y = [Y_pareto[0]]
    for i in X_pareto[1:]:
        x.append(i)
        x.append(i)
    for i in Y_pareto:
        y.append(i)
        y.append(i)
    plt.plot(x, y[:-1], color='r', alpha=0.25)

    true_pareto = find_pareto(Y)
    X_pareto = true_pareto[:, 0]
    Y_pareto = true_pareto[:, 1]
    x = [X_pareto[0], X_pareto[0]]
    y = [Y_pareto[0]]
    for i in X_pareto[1:]:
        x.append(i)
        x.append(i)
    for i in Y_pareto:
        y.append(i)
        y.append(i)
    plt.plot(x, y[:-1], color='green', alpha=0.25)

    # plt.plot(x, y[:-1], color='r', alpha=0.25)
    plt.show()


def plot_error(error_list):
    # count number of zeros
    c = 0
    for i in error_list:
        if i == 0:
            c += 1
    x = np.arange(c, len(error_list))
    plt.errorbar(x, error_list[c:], color='slateblue')
    plt.title("Error of the predicted Pareto frontier")
    plt.xlabel("number of iterations")
    plt.ylabel("error")
    plt.show()

def plot_error_both_mode(e1, e2):
    # count number of zeros
    c = 0
    for i in e1:
        if i == 0:
            c += 1
    x = np.arange(c, len(e1))
    plt.errorbar(x, e1[c:], color='orangered')
    plt.errorbar(x, e2[c:], color='slateblue')
    plt.title("Error of the predicted Pareto frontier")
    plt.xlabel("number of iterations")
    plt.ylabel("error")
    plt.show()


