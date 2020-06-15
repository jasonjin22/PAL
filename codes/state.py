import random
import GPy
import numpy as np

import plot_pareto as pp

sample_ratio = 0.5
beta_ratio = 3


def tem(X, y1, y2):
    X_train = X
    ad_0 = np.zeros((X_train.shape[0], X_train.shape[1] + 1))
    ad_0[:, :-1] = X_train
    ad_1 = np.ones((X_train.shape[0], X_train.shape[1] + 1))
    ad_1[:, :-1] = X_train
    X_train_0 = ad_0
    X_train_1 = ad_1
    X_train_full = np.concatenate((X_train_0, X_train_1), axis=0)
    y_train_full = np.concatenate((y1, y2), axis=0)
    return X_train_full, y_train_full


class State(object):
    def __init__(self, training_set, epsilon, space_size, delta):
        self.training_set = training_set
        self.epsilon = epsilon
        self.t = 0
        self.beta_t = None
        self.P_t = set()
        self.U_t = set()
        self.beta_t = None
        self.sampled_points = set()
        self.discarded_points = set()
        self.x_dimension = None
        self.space_size = space_size  # the size of E, the design space
        self.delta = delta
        self.kernel1 = None
        self.kernel2 = None
        self.kern = None
        self.next_point_to_sample = None
        self.error = []

    def decide_hyper_para_normal(self):
        self.kernel1 = GPy.kern.RBF(input_dim=self.x_dimension, variance=1., lengthscale=1.)
        self.kernel2 = GPy.kern.RBF(input_dim=self.x_dimension, variance=1., lengthscale=1.)
        random_sample_points = random.sample(self.U_t, int(np.floor(sample_ratio * self.space_size)))
        random_sample_points_set = set(random_sample_points)
        # print(random_sample_points_set)
        X, y1, y2 = sampled_set_to_numpy_array(random_sample_points_set)
        m1 = GPy.models.GPRegression(X, y1, self.kernel1)
        m1.optimize(messages=False)
        m2 = GPy.models.GPRegression(X, y2, self.kernel2)
        m2.optimize(messages=False)

        print("The hyper parameters are: ")
        print(self.kernel1)
        print(self.kernel2)

    def decide_hyper_para_coregion(self):
        self.kernel1 = GPy.kern.RBF(self.x_dimension, lengthscale=100)
        self.kernel2 = GPy.kern.Coregionalize(1, output_dim=2, rank=2)
        self.kern = self.kernel1 ** self.kernel2

        random_sample_points = random.sample(self.U_t, int(np.floor(sample_ratio * self.space_size)))
        random_sample_points_set = set(random_sample_points)
        # print(random_sample_points_set)
        X, y1, y2 = sampled_set_to_numpy_array(random_sample_points_set)

        X_train_full, y_train_full = tem(X, y1, y2)
        model = GPy.models.GPRegression(X_train_full, y_train_full, self.kern)
        model.optimize()

    def sampling(self):
        if self.t == 1:
            sampled_point = self.U_t.pop()
            self.U_t.add(sampled_point)  # the point should not be discarded
            self.sampled_points.add(sampled_point)
            self.next_point_to_sample = sampled_point
            sampled_point.being_sampled(self.beta_t)
        else:
            work_set = self.P_t.union(self.U_t)
            next_point_to_sample = None
            max_w_t = 0
            for point in work_set:
                if point.w_t > max_w_t:
                    max_w_t = point.w_t
                    next_point_to_sample = point
            self.sampled_points.add(next_point_to_sample)
            self.next_point_to_sample = next_point_to_sample
            next_point_to_sample.being_sampled(self.beta_t)

    def modeling(self, mode):
        # do GP regression on all the sampled points
        # change the sampled_points set to numpy array for GP regression
        X, y1, y2 = sampled_set_to_numpy_array(self.sampled_points)
        # do GP regression
        if mode == 0:
            m1 = GPy.models.GPRegression(X, y1, self.kernel1)
            m2 = GPy.models.GPRegression(X, y2, self.kernel2)
            # update mu_t, sigma_t
            work_set = self.P_t.union(self.U_t)
            for point in work_set:
                point.update_mu_sigma(m1, m2)
        elif mode == 1:
            X_train_full, y_train_full = tem(X, y1, y2)
            model = GPy.models.GPRegression(X_train_full, y_train_full, self.kern)
            # update mu_t, sigma_t
            work_set = self.P_t.union(self.U_t)
            for point in work_set:
                point.update_mu_sigma_coregion(model)
        # update R_t
        for point in work_set:
            point.update_R_t(self.beta_t)
        # pp.plot_sampled_points(self.training_set, sampled_set_to_numpy_array(self.sampled_points),
        #                        self.next_point_to_sample, self.U_t)
        self.count_high_error_points_in_U_t()

    def discard(self):
        P_t_ps = pessimistic_set(self.P_t)
        U_t_P_t_ps = pessimistic_set(self.U_t.union(self.P_t))

        # discard points in x in U that are dominated by some point x′ in p_pess(P_t)
        work_set = self.U_t
        discard_set = set()
        for point in work_set:
            for another_point in P_t_ps:
                if point.being_epsilon_dominated(another_point, self.epsilon):
                    discard_set.add(point)
                    self.discarded_points.add(point)
                    break
        self.U_t = self.U_t - discard_set

        # discard points x in U_t - p_pess(P_t U U_t) that are dominated by some point x′ in p_pess(P_t U U_t)
        work_set = self.U_t - U_t_P_t_ps
        discard_set = set()
        for point in work_set:
            for another_point in U_t_P_t_ps:
                if point.being_epsilon_dominated(another_point, self.epsilon):
                    discard_set.add(point)
                    self.discarded_points.add(point)
                    break
        self.U_t = self.U_t - discard_set
        # pp.plot_pessimistic_set_and_discarded_points(self.training_set, sampled_set_to_numpy_array(U_t_P_t_ps),
        #                                              sampled_set_to_numpy_array(discard_set),
        #                                              sampled_set_to_numpy_array(self.discarded_points), U_t_P_t_ps,
        #                                              self.next_point_to_sample)
        # print(self.sampled_points)
        # self.U_t = self.U_t - self.discard_point(P_t_ps, self.U_t)
        # self.U_t = self.U_t - self.discard_point(U_t_P_t_ps, self.U_t - U_t_P_t_ps)
        # pp.plot_discarded_points(self.training_set, sampled_set_to_numpy_array(self.sampled_points),
        #                          self.next_point_to_sample, sampled_set_to_numpy_array(self.discarded_points),
        #                          sampled_set_to_numpy_array(self.P_t), self.U_t)

    def covering(self):
        cover_set = set()
        discard_set = set()  # the set is used to contain the points that are dominated by newly added pareto points
        for point in self.U_t:
            tem_set = set()
            tem_set.add(point)
            work_set = self.U_t.union(self.P_t) - tem_set
            if point.should_to_P_t(work_set, self.epsilon):
                self.P_t.add(point)
                cover_set.add(point)
                # self.U_t.remove(point)
                print("COVERING: ", point)
                # discard the points in U_t that are dominated by the point newly moved to P_t
                discard_set_in_loop = set()
                for small_point in work_set:
                    if small_point.being_epsilon_dominated(point, self.epsilon):
                        discard_set_in_loop.add(small_point)
                        print("dominated by the new P_t point: ", small_point, " by ", point)
                # pp.plot_covered(self.training_set, sampled_set_to_numpy_array(self.P_t), point,
                #                 sampled_set_to_numpy_array(discard_set_in_loop))
                discard_set = discard_set.union(discard_set_in_loop)
            else:
                print("should_to_P_t: NOT")
        self.U_t = self.U_t - cover_set
        self.U_t = self.U_t - discard_set

    def count_high_error_points_in_U_t(self):
        cc = 0
        for point in self.U_t:
            # print(point in self.sampled_points)
            if abs(point.y[0] - point.mu_1) > 0.1:
                # print("point.y1", point.y[0] - point.mu_1, "sigma1: ", point.sigma_1)
                cc += 1
            if abs(point.y[1] - point.mu_2) > 0.1:
                # print("point.y2", point.y[1] - point.mu_2, "sigma2: ", point.sigma_2)
                cc += 1
        print("high_error_points: ", cc)
        print("len(U)", len(self.U_t))
        print("len(discarded_points)", len(self.discarded_points))

    def not_global_optimal(self, m1, m2):
        for point in self.sampled_points:
            mean1, var1 = m1.predict(np.array([point.x]))
            mean2, var2 = m2.predict(np.array([point.x]))
            if var1 > 1 or var2 > 1:
                # print(var1, var2)
                return True
        return False

    def add_point(self, point):
        """
        add a point into the U_t set when initialize the State
        @param point: the point
        @return: nothing
        """
        self.U_t.add(point)
        if self.x_dimension is None:
            self.x_dimension = point.dimension
            print("x_dimension: ", self.x_dimension)

    def update_beta_t(self):
        self.t += 1
        print("t = ", self.t)
        pi_t = (np.pi ** 2 * self.t**2) / 6
        beta_t_tem = 2 * self.space_size * pi_t / self.delta  # m|E|pi_t/delta
        self.beta_t = np.sqrt(2 * np.log(beta_t_tem)) / beta_ratio

    def check_stop_criteria(self):
        stop = True
        for point in self.U_t:
            if point.w_t > self.epsilon:
                stop = False
                break
        return stop

    def compute_error(self, true_pareto, range1, range2):
        error_sum = 0
        for pareto_point in true_pareto:
            error_list_for_each_point = []
            for predicted_point in self.P_t:
                error_list_for_each_point.append(max((pareto_point[0] - predicted_point.mu_1) * 100 / range1,
                                                     (pareto_point[1] - predicted_point.mu_2) * 100 / range2))
            if len(error_list_for_each_point) == 0:
                pass
            else:
                error_sum += min(error_list_for_each_point)
        self.error.append(error_sum / len(true_pareto))


def sampled_set_to_numpy_array(the_set):
    X_list = None
    y1_list = None
    y2_list = None
    for point in the_set:
        if X_list is None:
            X_list = np.array([point.x])
        else:
            X_list = np.append(X_list, [point.x], axis=0)
        if y1_list is None:
            y1_list = np.array([point.y[0]])
            y1_list = y1_list[:, np.newaxis]
        else:
            y1_list = np.append(y1_list, point.y[0])
            y1_list = y1_list[:, np.newaxis]

        if y2_list is None:
            y2_list = np.array([point.y[1]])
            y2_list = y2_list[:, np.newaxis]
        else:
            y2_list = np.append(y2_list, point.y[1])
            y2_list = y2_list[:, np.newaxis]

    # print("X_list: ", X_list)
    # print("y1_list: ", y1_list)
    # print("y2_list: ", y2_list)
    return X_list, y1_list, y2_list


def pessimistic_set(the_set):
    the_list = list(the_set)
    s = sorted(the_list, key=lambda x: x.R_t[0][0], reverse=True)
    pessimistic_list = []
    curr_max = float('-inf')
    for i in range(len(s)):
        if s[i].R_t[0][1] > curr_max:
            curr_max = s[i].R_t[0][1]
            pessimistic_list.append(s[i])
        else:
            pass
    return set(pessimistic_list)


# def discard_point(self, pessimistic_set, U):
#     removed_points = set()
#     U_prime = set()
#     for point in pessimistic_set:
#         a = float(point.R_t[0][0])
#         b = float(point.R_t[0][1])
#         U_prime.add((point, (a + self.epsilon, b + self.epsilon)))
#     for point in self.U_t - pessimistic_set:
#         # print("point.R_t[1][0]: ", point.R_t[1][0], type(point.R_t[1][0]) == np.float64)
#         a = float(point.R_t[1][0])
#         b = float(point.R_t[1][1])
#         U_prime.add((point, (a, b)))
#     sortU = sorted(U_prime, key=lambda xx: xx[1][0])
#     current_max = float('-inf')
#     for x, (f1, f2) in sortU:
#         if x in pessimistic_set:
#             current_max = f2
#         else:
#             if f2 < current_max:
#                 U.remove(x)
#                 self.discarded_points.add(x)
#                 removed_points.add(x)
#                 print("REMOVED: ", x, "len(U)", len(self.U_t))
#     return removed_points