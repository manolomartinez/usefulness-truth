"""
An attempt of doing naïve dynamical programming on a discretized version of the
interface model.

I will consider 3 samples, with 20 possible states each. Policy Evaluation is
unnecessary---I will directly code policy value. The main action is in Policy
Improvement.
"""

import numpy as np
from itertools import combinations_with_replacement, permutations


class Environment:
    def __init__(self, detectable_values, samples, labels, payoff_function):
        """
        Inputs:
        * detectable_values: an integer giving the number of possible detectable_values an individual
        sample can be in.
        * samples: an integer giving the number of samples per round
        * labels: an integer giving the number of labels the classifier can use
        * payoff_function: a function from detectable values to payoffs
        """
        self.dv = detectable_values
        self.samples = samples
        self.labels = labels
        self.payoff_function = payoff_function
        self.classinputs = self.classifier_inputs()
        self.actorinputs = self.actor_inputs()
        self.payoff_vec = np.vectorize(payoff_function)

    def classifier_inputs(self):
        """
        Return an array with each possible state the classifier encounters, given
        the number of states each sample can be in, and the total number of
        samples.
        """
        states_as_a_list = [list(total_state) for total_state in
                            combinations_with_replacement(range(self.dv),
                                                          self.samples)]
        return np.array(states_as_a_list)

    def actor_inputs(self):
        """
        Return an array with each possible "state" the actor encounters, given the
        number of labels and the total number of samples.
        """
        states_as_a_list = [list(total_state) for total_state in
                            combinations_with_replacement(range(self.labels),
                                                          self.samples)]
        return np.array(states_as_a_list)

    def policy_value(self, stateindex, policy):
        """
        Return the value of a policy, in a certain state, given a function from states to
        payoffs.

                """
        classifier, actor = policy
        state = self.classinputs[stateindex]  
        # This gives a vector with the dvs of the samples
        classification = np.array(np.take(classifier, state))
        label_ranking = np.ravel([np.where(label==actor) for label in
                                          classification])
        bests = np.where(label_ranking==np.max(label_ranking))[0]
        chosen_samples = [state[best] for best in bests]
        return np.mean(self.payoff_vec(chosen_samples))


    def expected_policy_value(self, policy):
        """
        Return the expected value of a policy for all possible states.
        """
        vec = np.vectorize(lambda stateindex: self.policy_value(stateindex,
                                                                policy))
        return np.mean(vec(np.arange(len(self.classinputs))))

    def random_policy(self):
        """
        Return a random policy. A policy is a tuple of two arrays. 
            
            * The first one is a vector of length self.dv, and in
            which each cell can take a value in range(labels).
            * The second one is an array with a random permutation of
                                  range(labels), representing the ranking of
                                  labels        
        """
        actor = np.arange(self.labels)
        np.random.shuffle(actor)
        return (np.random.randint(self.labels, size=self.dv), actor)

    def policy_improvement_one_step(self, policy):
        """
        One step of policy improvement following the algorithm in Sutton &
        Barto, p. 89 
        """
        classifier, actor = policy
        old = self.expected_policy_value(policy)
        print("optimizing classifier...")
        for index in range(self.dv):
            new_classifier = np.copy(classifier)
            for other_label in range(self.labels):
                new_classifier[index] = other_label
                new = self.expected_policy_value((new_classifier, actor)) 
                if new > old:
                    print(new)
                    print("changing response to dv {} from {} to {}".
                          format(index, classifier[index], other_label))
                    classifier = np.copy(new_classifier)
                    old = new
                    actor = self.optimize_actor(old, classifier, actor)
        return (classifier, actor)

    def optimize_actor(self, old, classifier, actor):
        print("optimizing actor...")
        for new_actor in permutations(range(self.labels)):
            new = self.expected_policy_value((classifier, new_actor)) 
            if new > old:
                print(new)
                actor = np.copy(new_actor)
                old = new
        return actor


def gaussian(x):
    return 1/np.sqrt(2*np.pi*.0225) * np.exp(-1*(x-.5)**2/.045)
