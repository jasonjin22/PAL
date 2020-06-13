import numpy as np

sample_error = 0.0000001


class Point(object):
    def __init__(self, x_full, x, y):
        self.x_full = x_full
        self.x = x
        self.y = y
        self.dimension = x.shape[0]
        self.mu_1 = None
        self.sigma_1 = None
        self.mu_2 = None
        self.sigma_2 = None
        self.R_t = None  # None represents the R^n space
        self.w_t = None  # w_t(x) = max |y - y'|_2
        self.sampled = False

    def __str__(self):
        return "Point " + "x: " + str(self.x) + " y: " + str(self.y)

    def __repr__(self):
        return "Point " + "x: " + str(self.x) + " y: " + str(self.y)

    def update_mu_sigma(self, model1, model2):
        # if not self.sampled:
        mean1, var1 = model1.predict_noiseless(np.array([self.x]))
        mean2, var2 = model2.predict_noiseless(np.array([self.x]))
        self.mu_1 = mean1
        self.sigma_1 = np.sqrt(var1)
        self.mu_2 = mean2
        self.sigma_2 = np.sqrt(var2)

    def update_R_t(self, beta_t):
        # if not self.sampled:
        if self.sampled:
            return
        left_bottom = [self.mu_1 - beta_t * self.sigma_1,
                       self.mu_2 - beta_t * self.sigma_2]
        right_upper = [self.mu_1 + beta_t * self.sigma_1,
                       self.mu_2 + beta_t * self.sigma_2]
        if self.R_t is None:
            # the original R_t is the whole space
            self.R_t = [left_bottom, right_upper]
            # self.check_R_t_valid()
        else:
            # print("last RT: ", self.R_t)
            # self.R_t = [left_bottom, right_upper]
            intersection = [[max(left_bottom[0], self.R_t[0][0]),
                             max(left_bottom[1], self.R_t[0][1])],
                            [min(right_upper[0], self.R_t[1][0]),
                             min(right_upper[1], self.R_t[1][1])]]
            new_R_t = [left_bottom, right_upper]

            self.check_R_t_valid(intersection, new_R_t)
        self.updated_w_t()

    def show_mu_sigma(self):
        print(self.x, self.y, self.mu_1, self.sigma_1, self.mu_2, self.sigma_2)

    def check_R_t_valid(self, intersection, new_R_t):
        left_bottom = intersection[0]
        right_upper = intersection[1]
        if left_bottom[0] < right_upper[0] and left_bottom[1] < right_upper[1]:
            self.R_t = intersection
        else:
            # print("The R_t is: ", self.R_t)
            # print("The R_t is not valid")
            self.R_t = new_R_t
            # raise Exception("The R_t is not valid")

    def updated_w_t(self):
        self.w_t = np.sqrt((2 * self.sigma_1) ** 2 + (2 * self.sigma_2) ** 2)

    def being_sampled(self, beta_t):
        self.sampled = True
        self.mu_1 = self.y[0]
        self.sigma_1 = sample_error
        self.mu_2 = self.y[1]
        self.sigma_2 = sample_error
        left_bottom = [self.mu_1 - beta_t * self.sigma_1,
                       self.mu_2 - beta_t * self.sigma_2]
        right_upper = [self.mu_1 + beta_t * self.sigma_1,
                       self.mu_2 + beta_t * self.sigma_2]
        self.R_t = [left_bottom, right_upper]
        self.updated_w_t()

    def being_epsilon_dominated(self, another_point, epsilon):
        return (self.R_t[1][0] + epsilon < another_point.R_t[0][0])\
               and (self.R_t[1][1] + epsilon < another_point.R_t[0][1])

    def should_to_P_t(self, work_set, epsilon):
        result = True
        for point in work_set:
            if (point.R_t[1][0] + epsilon > self.R_t[0][0]) and (point.R_t[1][1] + epsilon > self.R_t[0][1]):
                result = False
                break
        return result


