
import numpy as np
from time import time
import statsmodels.api as sm
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras



class CodaCore:
    def __init__(
            self,
            objective,
            random_state=None,
            type='balance',
            regularization=1.0, # lambda parameter
            ensemble_max_size=10,
            cv_params=None,
            opt_params=None,
            verbose=False
        ):
        """
        A CodaCore model
        :param objective:
        :param random_state:
        :param type:
        :param regularization:
        :param ensemble_max_size:
        :param cv_params:
        :param opt_params:
        :param verbose:
        """
        self.objective=objective
        self.random_state = random_state
        self.type = self.set_type(type)
        self.regularization = regularization
        self.ensemble_max_size=ensemble_max_size
        self.cv_params = self.set_cv_params(cv_params)
        self.opt_params = self.set_opt_params(opt_params)
        self.verbose = verbose
        self.ensemble = []
        if objective != 'binary_classification':
            raise NotImplementedError

    def set_type(self, type):
        if type in ['balances', 'balance', 'B', 'ILR']:
            return 'B'
        elif type in ['amalgamations', 'amalgam', 'A', 'SLR']:
            return 'A'
        else:
            raise ValueError("Invalid 'type' argument given", type)

    def set_cv_params(self, cv_params):
        """Overrides the defaults with any user-specified params"""
        default_params = {
            'num_folds': 5,
            'num_thresholds': 20,
        }

        if cv_params is not None:
            default_params.update(cv_params)
        return default_params

    def set_opt_params(self, opt_params):
        """Overrides the defaults with any user-specified params"""
        default_params = {
            'epochs': 100,
            'batch_size': None,
            'adaptive_lr': 0.5,
            'momentum': 0.9,
            'epsilon_a': 1e-6,
            'epsilon_b': 1e-2,
        }

        if opt_params is not None:
            default_params.update(opt_params)
        return default_params

    def fit(self, x, y):
        """
        Fits CodaCore to some data

        :param x: Compositional data (rowsums will be normalized to 1)
        :type x: np.ndarray
        :param y: A response variable
        :type y: np.ndarray
        :return:
        """

        if np.any(x == 0):
            raise ValueError("The data contain zeros. Please impute prior to modeling.")

        if np.any(x.sum(axis=1) != 1.0):
            x = x / x.sum(axis=1, keepdims=True)

        current_estimate = y * 0.0
        for i in range(self.ensemble_max_size):
            start_time = time()

            base_learner = CodaCoreBase(
                    objective=self.objective,
                    random_state=self.random_state,
                    type=self.type,
                    regularization=self.regularization,
                    cv_params=self.cv_params,
                    opt_params=self.opt_params,
                    verbose=self.verbose
                )
            base_learner.fit(x, y, current_estimate)

            end_time = time()

            if self.verbose:
                print("Stage ", i)
                print("Time taken: ", end_time - start_time)

            # If base learner is empty, we stop (no further gain in CV score):
            if not (np.any(base_learner.numerator_parts) or np.any(base_learner.denominator_parts)):
                break

            # Else we append base learner to ensemble and update predictions
            self.ensemble.append(base_learner)

            current_estimate += base_learner.predict(x)

        return self

    def predict(self, x, return_logits=True):
        y_pred = np.zeros(x.shape[0])
        for base_learner in self.ensemble:
            y_pred += base_learner.predict(x)

        if return_logits:
            return y_pred
        else:
            return 1 / (1 + np.exp(-y_pred))

    def summary(self):
        """Prints a summary of the fitted model"""
        if self.type == 'B':
            print("Number of balances found: ", len(self.ensemble))
        elif self.type == 'A':
            print("Number of SLRs found:", len(self.ensemble))
        for i in range(len(self.ensemble)):
            base_learner = self.ensemble[i]
            print("Log-ratio ", i)
            print("Numerator parts:", self.get_numerator_parts(i))
            print("Denominator parts:", self.get_denominator_parts(i))
            print("Slope:", base_learner.slope)
            if self.objective == 'binary_classification':
                print("Accuracy:", base_learner.metrics['acc'])
                print("ROC AUC:", base_learner.metrics['auc'])
                print("Cross-entropy:", base_learner.metrics['xe'])
        return

    def get_numerator_parts(self, base_learner_index):
        return list(np.where(self.ensemble[base_learner_index].numerator_parts)[0])

    def get_denominator_parts(self, base_learner_index):
        return list(np.where(self.ensemble[base_learner_index].denominator_parts)[0])

    def get_logratio(self, x):
        """
        Takes a set of compositional inputs (of the same shape
        as the original training data) and produces the
        corresponding logratio features under the trained model.

        :param x:
        :return:
        """
        logratios = np.zeros([x.shape[0], len(self.ensemble)])
        for j in range(len(self.ensemble)):
            logratios[:, j] = self.ensemble[j].get_logratio(x)
        return logratios

class CodaCoreBase:
    def __init__(self,
                 objective,
                 random_state,
                 type,
                 regularization,
                 cv_params,
                 opt_params,
                 verbose
                 ):
        """
        A CodaCore base learner object.

        Runs gradient descent on our relaxed problem,
        and applies the the discretization procedure.

        :param objective:
        :param random_state:
        :param type:
        :param regularization:
        :param cv_params:
        :param opt_params:
        :param verbose:
        """
        self.objective = objective
        self.random_state = random_state
        self.type = type
        self.regularization = regularization
        self.cv_params = cv_params
        self.opt_params = opt_params
        self.verbose = verbose
        self.intercept = None
        self.slope = None
        self.weights = None
        self.soft_assignment = None
        self.numerator_parts = None
        self.denominator_parts = None
        self.metrics = None

    def fit(self, x, y, current_estimate):
        self.train_relaxation(x, y, current_estimate)
        self.set_threshold_cv(x, y, current_estimate)

        # Store some metrics
        y_pred = self.predict(x) + current_estimate
        y_pred = 1 / (1 + np.exp(-y_pred))
        acc = np.mean(y == (y_pred > 0.5) * 1)
        auc = roc_auc_score(y, y_pred)
        xe = log_loss(y, y_pred)
        self.metrics = {'acc': acc, 'auc': auc, 'xe': xe}
        return

    def train_relaxation(self, x, y, current_estimate):

        num_obs, input_dim = x.shape

        # Find a good initialization for GD
        if self.objective == 'binary_classification':
            loss = keras.losses.BinaryCrossentropy(from_logits=True)
            if abs(np.mean(1 / (1 + np.exp(-current_estimate))) - np.mean(y)) < 0.001:
                # Protect against numerical errors in glm() call
                intercept_init = 0.0
            else:
                # log_reg = LogisticRegression(fit_intercept=False).fit(np.ones([num_obs, 1]), y)
                # intercept_init = log_reg.coef_[0, 0]
                # log_reg = sm.Logit(y, np.ones([num_obs, 1]), offset=current_estimate).fit()
                log_reg = sm.GLM(y, np.ones([num_obs, 1]), offset=current_estimate, family=sm.families.Binomial()).fit()
                intercept_init = log_reg.params[0]
        elif self.objective == 'regression':
            loss = keras.losses.MSE()
            intercept_init = np.mean(y - current_estimate)

        if self.opt_params['batch_size'] is None:
            batch_size = num_obs
        else:
            raise NotImplementedError

        def gradient_descent(lr, epochs):

            if self.type == 'B':
                relaxation_layer = BalanceRelaxation(epsilon=self.opt_params['epsilon_b'])
            elif self.type == 'A':
                relaxation_layer = AmalgamationRelaxation(epsilon=self.opt_params['epsilon_a'])
            else:
                raise ValueError("Unknown type.")

            # Compute learning rate using adaptive strategy
            optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=self.opt_params['momentum'])
            model = ModelRelaxation(relaxation_layer, current_estimate, intercept_init)
            model.compile(optimizer, loss)
            model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=False)
            return model

        # First train for 1 epoch with lr = 1 to find adaptive lr
        model = gradient_descent(1, 1)
        lr = self.opt_params['adaptive_lr'] / np.max(np.abs(model.relaxation_layer.get_weights()[0]))
        # Now we can use our adaptive lr to retrain
        model = gradient_descent(lr, self.opt_params['epochs'])

        self.weights = model.relaxation_layer.get_weights()[0].reshape([-1])
        soft_assignment = 2 / (1 + np.exp(-self.weights)) - 1

        # Equalize most positive and most negative weights (for slightly more stable features)
        equalization_ratio = np.max(soft_assignment) / min(soft_assignment) * (-1)
        soft_assignment[soft_assignment < 0] = soft_assignment[soft_assignment < 0] * equalization_ratio
        self.soft_assignment = soft_assignment

        return

    def set_threshold_cv(self, x, y, current_estimate):

        if np.any(np.abs(self.soft_assignment) > 0.999999):
            Warning("Large weights encountered in gradient descent; vanishing gradients likely.")

        candidate_thresholds = -np.sort(-np.abs(self.soft_assignment))
        num_thresholds = self.cv_params['num_thresholds']
        num_folds = self.cv_params['num_folds']

        candidate_thresholds = candidate_thresholds[1:min(num_thresholds, len(candidate_thresholds))]

        start_time = time()
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=self.random_state)

        scores = np.zeros([len(candidate_thresholds), num_folds])
        i = 0
        for threshold in candidate_thresholds:
            self.discretize(threshold=threshold)
            j = 0
            for train_index, test_index in skf.split(x, y):
                logratio = self.get_logratio(x[train_index])
                # log_reg = LogisticRegression(random_state=self.random_state).fit(logratio, y[train_index])
                # log_reg = sm.Logit(y[train_index], logratio, offset=current_estimate[train_index]).fit()
                log_reg = sm.GLM(y[train_index], sm.add_constant(logratio), offset=current_estimate[train_index],
                                  family=sm.families.Binomial()).fit()
                logratio = self.get_logratio(x[test_index])
                y_pred = log_reg.predict(sm.add_constant(logratio), offset=current_estimate[test_index])
                cross_entropy = log_loss(y[test_index], y_pred)
                scores[i, j] = -cross_entropy
                j += 1
            i += 1

        means = np.mean(scores, axis=1)
        stds = np.std(scores, axis=1) / np.sqrt(num_folds)
        se_rule = np.max(means) - self.regularization * stds[np.argmax(means)]
        optimal_threshold = candidate_thresholds[means >= se_rule][0]

        end_time = time()
        if self.verbose:
            print("CV time:", end_time - start_time)

        no_improvement = se_rule < -log_loss(y, 1 / (1 + np.exp(-current_estimate)))
        if no_improvement:
            optimal_threshold = 1.1 # bigger than the soft assignments, i.e., empty logratio

        self.discretize(threshold=optimal_threshold)
        logratio = self.get_logratio(x)
        log_reg = sm.GLM(y, sm.add_constant(logratio), offset=current_estimate, family=sm.families.Binomial()).fit()
        self.intercept, self.slope = log_reg.params

        return

    def discretize(self, threshold):
        """Selects covariates for logratio based on a threshold value."""
        self.numerator_parts = self.soft_assignment >= threshold
        self.denominator_parts = self.soft_assignment <= -threshold
        return

    def get_logratio(self, x):
        """Computes the logratio on data x, given some selection of covariates."""
        if not (np.any(self.numerator_parts) or np.any(self.denominator_parts)):
            logratio = np.zeros([x.shape[0]])
        elif self.type == 'B':
            positive_part = np.mean(np.log(x[:, self.numerator_parts]), axis=1)
            negative_part = np.mean(np.log(x[:, self.denominator_parts]), axis=1)
            logratio = positive_part - negative_part
        elif self.type == 'A':
            positive_part = np.sum(x[:, self.numerator_parts], axis=1)
            negative_part = np.sum(x[:, self.denominator_parts], axis=1)
            epsilon = self.opt_params['epsilon_a']
            logratio = np.log(positive_part + epsilon) - np.log(negative_part + epsilon)
        else:
            raise ValueError("Unknown type given:", self.type)

        return logratio

    def predict(self, x, return_logits=True):
        logratio = self.get_logratio(x)
        logits = self.intercept + self.slope * logratio
        if return_logits:
            return logits
        else:
            return 1 / (1 + np.exp(-logits))

class BalanceRelaxation(keras.layers.Layer):
    def __init__(self, epsilon):
        super(BalanceRelaxation, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        soft_assignment = 2 * tf.sigmoid(self.w) - 1
        # Add the small value to ensure gradient flows at exact zeros (initial values)
        pve_assignments = tf.nn.relu(soft_assignment + 1e-20)
        nve_assignments = keras.activations.relu(-soft_assignment)
        pve_part = tf.matmul(tf.math.log(x), pve_assignments) / \
                   tf.maximum(tf.reduce_sum(pve_assignments), self.epsilon)
        nve_part = tf.matmul(tf.math.log(x), nve_assignments) / \
                   tf.maximum(tf.reduce_sum(nve_assignments), self.epsilon)
        logratio = pve_part - nve_part
        return logratio


class AmalgamationRelaxation(keras.layers.Layer):
    def __init__(self, epsilon):
        super(AmalgamationRelaxation, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        soft_assignment = 2 * tf.sigmoid(self.w) - 1
        # Add the small value to ensure gradient flows at exact zeros (initial values)
        pve_assignments = tf.nn.relu(soft_assignment + 1e-20)
        nve_assignments = keras.activations.relu(-soft_assignment)
        logratio = tf.math.log(tf.matmul(x, pve_assignments) + self.epsilon) - \
                   tf.math.log(tf.matmul(x, nve_assignments) + self.epsilon)
        return logratio

class ModelRelaxation(keras.Model):
    def __init__(self, relaxation_layer, current_estimate, intercept_init):
        super(ModelRelaxation, self).__init__()
        self.relaxation_layer = relaxation_layer
        self.output_layer = keras.layers.Dense(1,
                                               kernel_initializer=keras.initializers.constant(0.1),
                                               bias_initializer=keras.initializers.constant(intercept_init))
        current_estimate = tf.convert_to_tensor(current_estimate.reshape([-1, 1]), dtype='float32')
        self.current_estimate = tf.Variable(current_estimate, trainable=False)

    def call(self, x):
        logratio = self.relaxation_layer(x)
        eta = self.output_layer(logratio) + self.current_estimate
        return eta


class CodaCoreClassifier(CodaCore):
    """A CodaCore classification model"""

    def __init__(self, *args, **kwargs):
        super(CodaCoreClassifier, self).__init__(
            args,
            kwargs
        )
        self.objective='binary_crossentropy'


