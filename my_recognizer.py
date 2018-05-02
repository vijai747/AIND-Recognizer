import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # TODO implement the recognizer
    probabilities = []
    guesses = []
    all_sequences = test_set.get_all_sequences()

    for word_id in all_sequences:
        word_id_X, word_id_Lengths = test_set.get_item_Xlengths(word_id)

        best_score = float('-inf')
        best_guess = None
        word_likelihood = {}
        # Initialize variables used to keep track of likelihoods and guesses
        for word_name in models:
            # Calculate the score of a particular sample in the test_set against every word model
            current_model = models[word_name]
            try:
                current_score = current_model.score(word_id_X, word_id_Lengths)
                word_likelihood[word_name] = current_score
                # Store the likelihood of each possible word model using the word name as the key
                if current_score > best_score:
                    best_score = current_score
                    best_guess = word_name
                    # Keep track of the best guess for each test set
            except:
                pass

        probabilities.append(word_likelihood)
        guesses.append(best_guess)
        # Store likelihoods and guesses in appropriate locations before moving to next word_id

    return probabilities, guesses
