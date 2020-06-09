class TrainingSet(object):
    def __init__(self, X_full, X, Y):
        self.X_full = X_full
        self.X = X
        self.Y = Y
        self.num_samples = X.shape[0]  # the size of the design space
        self.dimension = X.shape[1]

