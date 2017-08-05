"""
An attempt of doing naïve dynamical programming on a discretized version of the
interface model.

I will consider 3 samples, with 20 possible states each. Policy Evaluation is
unnecessary---I will directly code policy value. The main action is in Policy
Improvement.
"""

import numpy as np
import itertools as it
import collections
import functools


# From https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)


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
        self.epsilon = 1e-5
        self.possibleclassifieractions = [self.possible_classifier_actions(
            stateindex) for stateindex in range(len(self.classinputs))]
        self.possibleactoractions = self.possible_actor_actions()
        self.possibleactions = it.product(self.possibleclassifieractions,
                                          self.possibleactoractions)

    def classifier_inputs(self):
        """
        Return an array with each possible state the classifier encounters, given
        the number of states each sample can be in, and the total number of
        samples.
        """
        states_as_a_list = [list(total_state) for total_state in
                            it.combinations_with_replacement(range(self.dv),
                                                          self.samples)]
        return np.array(states_as_a_list)

    def actor_inputs(self):
        """
        Return an array with each possible "state" the actor encounters, given the
        number of labels and the total number of samples.
        """
        states_as_a_list = [list(total_state) for total_state in
                            it.combinations_with_replacement(range(self.labels),
                                                          self.samples)]
        return np.array(states_as_a_list)

    def policy_value(self, stateindex, policy):
        """
        Return the value of a policy, in a certain state, given a function from states to
        payoffs.
                """
        actor, classifier = policy
        state = self.classinputs[stateindex]  
        # This gives a vector with the dvs of the samples
        classification = np.array(np.take(classifier, state))
        label_ranking = np.ravel([np.where(label==actor) for label in
                                          classification])
        bests = np.where(label_ranking==np.max(label_ranking))[0]
        chosen_samples = [state[best] for best in bests]
        return np.mean(self.payoff_vec(chosen_samples))

    @memoized
    def policy_value2(self, stateindex, policy):
        """
        Return the value of a policy, in a certain state, given a function from states to
        payoffs. This alternative form is suited to policy_improvement2.
        """
        actor, classification = policy
        state = self.classinputs[stateindex]  
        # This gives a vector with the dvs of the samples
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
        return (actor, np.random.randint(self.labels, size=self.dv))

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
        for new_actor in it.permutations(range(self.labels)):
            new = self.expected_policy_value((classifier, new_actor)) 
            if new > old:
                print(new)
                actor = np.copy(new_actor)
                old = new
        return actor

    def possible_classifier_actions(self, stateindex):
        """
        Return an iterator with all actions available to the classifier in the
        state corresponding to that stateindex.
        """
        state = self.classinputs[stateindex]  
        unique_dvs = np.unique(state)
        base_class_iterator = it.product(np.arange(self.labels),
                                         repeat=len(unique_dvs))
        return map(lambda similar: from_uniques_to_tuples(unique_dvs, state,
                                                          similar),
                   base_class_iterator)

    def possible_actor_actions(self):
        """
        Return an iterator with all actions available to the actor.
        """
        return it.permutations(range(self.labels))

    def possible_actions(self, stateindex):
        """
        Return an iterator with all actions available in a certain state.
        """
        return it.product(self.possible_actor_actions(),
                          self.possible_classifier_actions(stateindex))

    def update_policy(self, policy, stateindex, action):
        actor, classifier = np.copy(policy)
        new_actor, classification = action
        state = self.classinputs[stateindex]  
        for dv, label in zip(state, classification):
            classifier[dv] = label
        return new_actor, classifier


    def policy_improvement2(self, policy):
        """
        A more faithful rendering of Sutton & Barto's algorithm.
        """
        actor, classifier = policy
        expected = self.expected_policy_value(policy)
        for s in range(len(self.classinputs)):
            print(s)
            print(expected)
            policy, expected = self.improve_one_state(s, np.copy(policy),
                                                      expected)
            print(policy)
            print(expected)
        return policy

    def improve_one_state(self, stateindex, policy, expected):
        print("before doing anything {}".format(policy))
        actor, classifier = policy
        newpolicies = [self.update_policy((np.copy(actor),
                                           np.copy(classifier)), stateindex,
                                          action) for action in
                       self.possible_actions(stateindex)]
        expected_payoffs = [self.expected_policy_value(apolicy) for apolicy in
                            newpolicies]
        max_expected = np.max(expected_payoffs) 
        if max_expected - expected > self.epsilon:
            # print("from {} to {}".format(old_payoff, max_payoff))
            print("we are here {} {}".format(max_expected, expected))
            bestindex = np.argmax(expected_payoffs)
            new_policy = newpolicies[bestindex]
            expected = max_expected
        else:
            new_policy = policy
            print("OLD policy: {}".format(policy))
        return new_policy, expected


def gaussian(x):
    return 1/np.sqrt(2*np.pi*.0225) * np.exp(-1*(x-.5)**2/.045)


def from_uniques_to_tuples(uniques, original, similar):
    """
    Take an original tuple together with an array with the unique members of
    that tuple and a similar array with as many unique elements as <uniques>,
    and return a tuple that is like <original>, but with each element
    substituted by its analogue in <similar>
    """
    new_tuple = np.empty_like(original)
    for i, elem in enumerate(uniques):
        new_tuple[original==elem] = similar[i]
    return new_tuple

def tabulate(function, start=0):
    """
    Return function(0), function(1), ...
    """
    return map(function, count(start))

def compare_elements(vector):
    """
    Compare all element pairs and produce a vector of booleans
    """
    return [first == second for first, second in it.combinations(vector, 2)]
