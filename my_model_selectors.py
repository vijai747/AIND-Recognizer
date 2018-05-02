import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
# n_components : referes to the number of hidden states in the hmm
# n_iter : maximum number of iterations to perform
# random_state : random number generator
# fit method : Estimates model parameters. Requires X (samples, features) and lengths (sequences)

            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_BIC = float('inf')
        best_model = None

        for hidden_feature_count in range(self.min_n_components, self.max_n_components + 1):

            try:
                hmm_model = self.base_model(hidden_feature_count)
                # Create a Hidden Markov Model for the current count of hidden features

                logL = hmm_model.score(self.X, self.lengths)
                # Calculate the log likelihood of the model created
                parameters = hidden_feature_count ** 2 + 2 * hidden_feature_count * len(self.X[0]) - 1
                # Calculate the parameter count for model
                current_BIC = -2 * logL + parameters * math.log(len(self.X))
                # Calculate the BIC value using the given formula

                if current_BIC < best_BIC:
                    best_BIC = current_BIC
                    best_model = hmm_model

            except:
                pass

        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_DIC = float('-inf')
        best_model = None
        all_words = list(self.words.keys())

        for hidden_feature_count in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(hidden_feature_count)
                # Create a Hidden Markov Model for the current count of hidden features
                this_word_score = hmm_model.score(self.X, self.lengths)
                other_word_score = 0
                # Calculate score using trained model for current word of interest
                for other_word in all_words:
                    if other_word != self.this_word:
                        otherX, otherLengths = self.hwords[other_word]
                        other_word_score += hmm_model.score(otherX, otherLengths)
                        # Calculate cumulative score using trained model for all words excluding word of interest

                current_DIC = this_word_score - (other_word_score / (len(all_words) - 1))
                # Calculate the current DIC value using the given formula
                if current_DIC > best_DIC:
                    best_DIC = current_DIC
                    best_model = hmm_model

            except:
                pass

        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_CV = float('-inf')
        best_feature_count = self.min_n_components

        for hidden_feature_count in range(self.min_n_components, self.max_n_components + 1):

            try:
                cumulative_score = 0

                if (len(self.sequences) <= 1):
                    # print("Not enough folds - assumed min hidden features")
                    hmm_model = self.base_model(hidden_feature_count)
                    average_score = hmm_model.score(self.X, self.lengths)
                else:
                    n_splits = min(len(self.sequences),3)
                    split_method = KFold(n_splits)
                    # Use KFold method to split up sequences into train and test sets

                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        trainX, trainLength = combine_sequences(cv_train_idx, self.sequences)
                        testX, testLength = combine_sequences(cv_test_idx, self.sequences)
                        # Generate rotating training and test sets
                        hmm_model = GaussianHMM(n_components=hidden_feature_count, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(trainX, trainLength)
                        # Create Hidden Markov Model using rotating training set
                        cumulative_score += hmm_model.score(testX, testLength)
                        # Calculate cumulative score using rotating test set
                    average_score = cumulative_score / n_splits
                # Calculate average CV score for specified hidden feature count
                if average_score > best_CV:
                    best_CV = average_score
                    best_feature_count = hidden_feature_count
            except:
                pass

        best_model = self.base_model(best_feature_count)

        return best_model
