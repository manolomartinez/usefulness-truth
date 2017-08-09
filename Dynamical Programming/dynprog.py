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
import multiprocess


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


class EnvironmentTwoAgents:
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

    def expected_policy_value(self, policy):
        """
        Return the expected value of a policy for all possible states.
        """
        vec = np.vectorize(lambda stateindex: self.policy_value(stateindex,
                                                                policy))
        return np.mean(vec(np.arange(len(self.classinputs))))

    def delta_pvm(self, oldmatrix, oldpolicy, newpolicy):
        """
        Calculate the value of newpolicy for all states, given the values
        (oldmatrix) for a highly overlapping policy oldpolicy
        """
        oldactor, oldclassifier = oldpolicy
        newactor, newclassifier = newpolicy
        # First, if the actor has changed we simply calculate everything from
        # scratch
        if oldactor != newactor:
            return self.policy_value_matrix(newpolicy)
        else:
            booleans = oldclassifier == newclassifier
            havechanged = np.ravel(np.where(booleans==False))

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
        return it.permutations(range(self.samples))

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

    def policy_improvement(self, policy):
        """
        Based on Sutton & Barto's dynamical-programming algorithm, p. 89.
        """
        actor, classifier = policy
        expected = self.expected_policy_value(policy)
        print("Initial policy value: {}".format(expected))
        last_changed = 0
        changed = True
        for s in it.cycle(range(len(self.classinputs))):
            print("State: {}".format(s))
            policy, expected, changed = self.improve_one_state(
                s, np.copy(policy), expected)
            if changed:
                last_changed = s
            elif last_changed == s + 1:
                break
        return policy

    def improve_one_state(self, stateindex, policy, expected):
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
            print("Expected policy value changed from {} to {}".format(
                expected, max_expected))
            bestindex = np.argmax(expected_payoffs)
            new_policy = newpolicies[bestindex]
            print("New policy: {}".format(new_policy))
            expected = max_expected
            changed = True
        else:
            new_policy = policy
            print("No change in policy")
            changed = False
        return new_policy, expected, changed

    def expected_payoffs_multiprocess(self, newpolicies):
        """
        Calculate expected payoffs for many policies in parallel       
        """
        pool = multiprocess.Pool(None)
        expected_payoffs = pool.imap_unordered(self.expected_policy_value,
                                               newpolicies)        
        data = np.array([payoff for payoff in expected_payoffs])
        pool.close()
        pool.join()
        return data


class EnvironmentThreeAgents:
    def __init__(self, detectable_values1, detectable_values2, samples,
                 labels1, labels2, payoff_function):
        """
        Inputs:
        * detectable_values: integers giving the number of possible
        detectable_values (of two different kinds) an individual
        sample can be in.
        * samples: an integer giving the number of samples per round
        * labels: an integer giving the number of labels the classifier can use
        (there are two classifiers, so, two set of labels)
        * payoff_function: a function from the two detectable values to payoffs
        """
        self.dv1 = detectable_values1
        self.dv2 = detectable_values2
        self.samples = samples
        self.labels1 = labels1
        self.labels2 = labels2
        self.whichone = {1:(detectable_values1, labels1),
                         2:(detectable_values2, labels2)}
        self.payoff_function = payoff_function
        self.classinputs1 = list(self.classifier_inputs(1))
        self.classinputs2 = list(self.classifier_inputs(2))
        self.stateindices = list(it.product(self.classinputs1,
                                            self.classinputs2))
        self.actorinputs = list(self.actor_inputs())
        self.payoff_vec = np.vectorize(payoff_function)
        self.epsilon = 1e-5

    def classifier_inputs(self, identity):
        """
        Return an array with each possible state the classifier encounters, given
        the number of states each sample can be in, and the total number of
        samples.
        """
        dv, _ = self.whichone[identity]
        return it.combinations_with_replacement(range(dv), self.samples)

    def actor_inputs(self):
        """
        Return an array with each possible "state" the actor encounters, given the
        number of labels, of the two sorts, and the total number of samples.
        """
        label_combinations = it.product(range(self.labels1),
                                        range(self.labels2))
        return it.combinations_with_replacement(label_combinations,
                                                self.samples)

    def possible_classifier_actions(self, identity, stateindex):
        """
        Return an iterator with all actions available to the classifier in the
        state corresponding to that stateindex.
        """
        dvs, labels = self.whichone[identity]
        classinputs = {1:self.classinputs1, 2:self.classinputs2}[identity]
        stateindex = {1:stateindex[0], 2:stateindex[1]}[identity]
        state = classinputs[stateindex]  
        unique_dvs = np.unique(state)
        base_class_iterator = it.product(np.arange(labels),
                                         repeat=len(unique_dvs))
        return map(lambda similar: from_uniques_to_tuples(unique_dvs, state,
                                                          similar),
                   base_class_iterator)

    def possible_actor_actions(self):
        """
        Return an array with all actions available to the actor.
        """
        iterable = it.permutations(it.product(range(self.samples), repeat=2))
        return np.array(list(iterable), dtype=(np.int, np.int))
                                                          

    def possible_actions(self, stateindex):
        """
        Return an iterator with all actions available in a certain state.
        """
        return it.product(self.possible_actor_actions(),
                          self.possible_classifier_actions(1, stateindex),
                          self.possible_classifier_actions(2, stateindex))


    def policy_value(self, stateindex, policy):
        """
        Return the value of a policy, in a certain state, given a function from states to
        payoffs.
                """
        actor, classifier1, classifier2 = policy
        stateindex1, stateindex2 = stateindex
        state1 = self.classinputs1[stateindex1]  
        state2 = self.classinputs2[stateindex2]  
        # This gives a vector with the dvs of the samples
        classification1 = np.take(classifier1, state1)
        classification2 = np.take(classifier2, state2)
        classification = list(zip(classification1, classification2))
        label_ranking = np.ravel([np.where(np.all(labels==actor, axis=1)) for labels in
                                          classification])
        bests = np.ravel(np.where(label_ranking==np.max(label_ranking))[0])
        chosen_samples = [(state1[best], state2[best]) for best in bests]
        return np.mean([self.payoff_function(*sample) for sample in
                        chosen_samples])

    def expected_policy_value(self, policy):
        """
        Return the expected value of a policy for all possible states.
        """
        func = lambda stateindex: self.policy_value(stateindex, policy)
        return np.mean([func(stateindex) for stateindex in
                        it.product(range(len(self.classinputs1)),
                                   range(len(self.classinputs2)))])

    def delta_pvm(self, oldmatrix, oldpolicy, newpolicy):
        """
        Calculate the value of newpolicy for all states, given the values
        (oldmatrix) for a highly overlapping policy oldpolicy
        """
        oldactor, oldclassifier = oldpolicy
        newactor, newclassifier = newpolicy
        # First, if the actor has changed we simply calculate everything from
        # scratch
        if oldactor != newactor:
            return self.policy_value_matrix(newpolicy)
        else:
            booleans = oldclassifier == newclassifier
            havechanged = np.ravel(np.where(booleans==False))

    def random_policy(self):
        """
        Return a random policy. A policy is a tuple of two arrays. 
            
            * The first one is a vector of length self.dv, and in
            which each cell can take a value in range(labels).
            * The second one is an array with a random permutation of
                                  range(labels), representing the ranking of
                                  labels        
        """
        actor = np.array(list(it.product(range(self.samples), repeat=2)),
                         dtype=(np.int, np.int))
        np.random.shuffle(actor)
        return (actor, np.random.randint(self.labels2, size=self.dv2),
                                         np.random.randint(self.labels2,
                                                           size=self.dv2))

    def update_policy(self, policy, stateindex, action):
        stateindex1, stateindex2 = stateindex
        actor, classifier1, classifier2 = np.copy(policy)
        new_actor, classification1, classification2 = action
        state1 = self.classinputs1[stateindex1]  
        for dv, label in zip(state1, classification1):
            classifier1[dv] = label
        state2 = self.classinputs2[stateindex2]  
        for dv, label in zip(state2, classification2):
            classifier2[dv] = label
        return new_actor, classifier1, classifier2

    def policy_improvement(self, policy):
        """
        Based on Sutton & Barto's dynamical-programming algorithm, p. 89.
        """
        actor, classifier1, classifer2 = policy
        expected = self.expected_policy_value(policy)
        print("Initial policy value: {}".format(expected))
        last_changed = 0
        changed = True
        all_states = list(it.product(range(len(self.classinputs1)),
                                     range(len(self.classinputs2))))
        for i, s in zip(it.count(), it.cycle(all_states)):
            print("State: {}".format(s))
            policy, expected, changed = self.improve_one_state(s,
                                                               np.copy(policy),
                                                               expected)
            if changed:
                last_changed = s
            else: 
                if i < len(all_states) - 1:
                    condition = last_changed == all_states[i+1]
                else:
                    condition = last_changed == all_states[0]
                if condition:
                    break
        return policy

    def improve_one_state(self, stateindex, policy, expected):
        actor, classifier1, classifier2 = policy
        newpolicies = [self.update_policy((np.copy(actor),
                                           np.copy(classifier1),
                                           np.copy(classifier2)), stateindex,
                                          action) for action in
                       self.possible_actions(stateindex)]
        expected_payoffs = [self.expected_policy_value(apolicy) for apolicy in
                            newpolicies]
        max_expected = np.max(expected_payoffs) 
        if max_expected - expected > self.epsilon:
            # print("from {} to {}".format(old_payoff, max_payoff))
            print("Expected policy value changed from {} to {}".format(
                expected, max_expected))
            bestindex = np.argmax(expected_payoffs)
            new_policy = newpolicies[bestindex]
            print("New policy: {}".format(new_policy))
            expected = max_expected
            changed = True
        else:
            new_policy = policy
            print("No change in policy")
            changed = False
        return new_policy, expected, changed

    def expected_payoffs_multiprocess(self, newpolicies):
        """
        Calculate expected payoffs for many policies in parallel       
        """
        pool = multiprocess.Pool(None)
        expected_payoffs = pool.imap_unordered(self.expected_policy_value,
                                               newpolicies)        
        data = np.array([payoff for payoff in expected_payoffs])
        pool.close()
        pool.join()
        return data


def gaussian(x):
    return 1/np.sqrt(2*np.pi*.0225) * np.exp(-1*(x/10-.5)**2/.045)


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
