# Copyright 2019 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import product
import sys
import unittest
if sys.version_info.major < 3:
    from mock import patch
else:
    from unittest.mock import patch

from dimod import ExactSolver, BinaryQuadraticModel
import dwavebinarycsp
from dwavebinarycsp.exceptions import ImpossibleBQM
from tabu import TabuSampler

from job_shop_scheduler import JobShopScheduler, get_jss_bqm, is_auxiliary_variable


def fill_with_zeros(expected_solution_dict, job_dict, max_time):
    """Fills the 'missing' expected_solution_dict keys with a value of 0.
    args:
        expected_solution_dict: a dictionary.  {taskName: taskValue, ..}
        job_dict: a dictionary. {jobName: [(machineVal, taskTimeSpan), ..], ..}
        max_time: integer. Max time for the schedule
    """
    for job, tasks in job_dict.items():
        for pos in range(len(tasks)):
            prefix = str(job) + "_" + str(pos)

            for t in range(max_time):
                key = prefix + "," + str(t)

                if key not in expected_solution_dict:
                    expected_solution_dict[key] = 0


def get_energy(solution_dict, bqm):
    min_energy = float('inf')
    aux_variables = [v for v in bqm.variables if is_auxiliary_variable(v)]

    # Try all possible values of auxiliary variables
    for aux_values in product([0, 1], repeat=len(aux_variables)):
        for variable, value in zip(aux_variables, aux_values):
            solution_dict[variable] = value

        temp_energy = bqm.energy(solution_dict)

        if temp_energy < min_energy:
            min_energy = temp_energy

    return min_energy


class TestIndividualJSSConstraints(unittest.TestCase):
    def test_oneStartConstraint(self):
        jobs = {"car": [("key", 2), ("gas", 1)],
                "stove": [("gas", 4)]}
        jss = JobShopScheduler(jobs, 3)
        jss._add_one_start_constraint()

        # Tasks only start once
        one_start_solution = {"car_0,0": 1, "car_0,1": 0, "car_0,2": 0,
                              "car_1,0": 0, "car_1,1": 1, "car_1,2": 0,
                              "stove_0,0": 1, "stove_0,1": 0, "stove_0,2": 0}

        # Task car_1 starts twice; it starts on times 0 and 1
        multi_start_solution = {"car_0,0": 1, "car_0,1": 0, "car_0,2": 0,
                                "car_1,0": 0, "car_1,1": 1, "car_1,2": 0,
                                "stove_0,0": 1, "stove_0,1": 1, "stove_0,2": 0}

        self.assertTrue(jss.csp.check(one_start_solution))
        self.assertFalse(jss.csp.check(multi_start_solution))

    def test_precedenceConstraint(self):
        jobs = {0: [("m1", 2), ("m2", 1)]}
        max_time = 3
        jss = JobShopScheduler(jobs, max_time)
        jss._add_precedence_constraint()

        # Task 0_0 starts after task 0_1
        backward_solution = {"0_0,2": 1, "0_1,0": 1}
        fill_with_zeros(backward_solution, jobs, max_time)

        # Tasks start at the same time
        same_start_solution = {"0_0,1": 1, "0_1,1": 1}
        fill_with_zeros(same_start_solution, jobs, max_time)

        # Task 0_1 starts before 0_0 has completed
        overlap_solution = {"0_0,1": 1, "0_1,2": 1}
        fill_with_zeros(overlap_solution, jobs, max_time)

        # Tasks follows correct order and respects task duration
        ordered_solution = {"0_0,0": 1, "0_1,2": 1}
        fill_with_zeros(ordered_solution, jobs, max_time)

        self.assertFalse(jss.csp.check(backward_solution))
        self.assertFalse(jss.csp.check(same_start_solution))
        self.assertFalse(jss.csp.check(overlap_solution))
        self.assertTrue(jss.csp.check(ordered_solution))

    def test_shareMachineConstraint(self):
        jobs = {"movie": [("pay", 1), ("watch", 3)],
                "tv": [("watch", 1)],
                "netflix": [("watch", 3)]}
        max_time = 7
        jss = JobShopScheduler(jobs, max_time)
        jss._add_share_machine_constraint()

        # All jobs 'watch' at the same time
        same_start_solution = {"movie_0,0": 1, "movie_1,1": 1,
                               "tv_0,1": 1,
                               "netflix_0,1": 1}
        fill_with_zeros(same_start_solution, jobs, max_time)

        # 'movie' does not obey precedence, but respects machine sharing
        bad_order_share_solution = {"movie_0,4": 1, "movie_1,0": 1,
                                    "tv_0,3": 1,
                                    "netflix_0,4": 1}
        fill_with_zeros(bad_order_share_solution, jobs, max_time)

        self.assertFalse(jss.csp.check(same_start_solution))
        self.assertTrue(jss.csp.check(bad_order_share_solution))

    def test_absurdTimesAreRemoved(self):
        pass


class TestCombinedJSSConstraints(unittest.TestCase):
    # TODO: test job with no tasks
    # TODO: test no jobs
    # TODO: test non-integer durations
    # TODO: insufficient max_time
    def test_denseSchedule(self):
        jobs = {"a": [(1, 2), (2, 2), (3, 2)],
                "b": [(3, 3), (2, 1), (1, 1)],
                "c": [(2, 2), (1, 3), (2, 1)]}
        max_time = 6
        jss = JobShopScheduler(jobs, max_time)
        jss.get_bqm()   # Run job shop scheduling constraints

        # Solution that obeys all constraints
        good_solution = {"a_0,0": 1, "a_1,2": 1, "a_2,4": 1,
                         "b_0,0": 1, "b_1,4": 1, "b_2,5": 1,
                         "c_0,0": 1, "c_1,2": 1, "c_2,5": 1}
        fill_with_zeros(good_solution, jobs, max_time)

        # Tasks a_1 and b_1 overlap in time on machine 2
        overlap_solution = {"a_0,0": 1, "a_1,2": 1, "a_2,4": 1,
                            "b_0,0": 1, "b_1,3": 1, "b_2,5": 1,
                            "c_0,0": 1, "c_1,2": 1, "c_2,5": 1}
        fill_with_zeros(overlap_solution, jobs, max_time)

        self.assertTrue(jss.csp.check(good_solution))
        self.assertFalse(jss.csp.check(overlap_solution))

    def test_relaxedSchedule(self):
        jobs = {"breakfast": [("cook", 2), ("eat", 1)],
                "music": [("play", 2)]}
        max_time = 5
        jss = JobShopScheduler(jobs, max_time)
        jss.get_bqm()   # Run job shop scheduling constraints

        # Solution obeys all constraints
        good_solution = {"breakfast_0,0": 1, "breakfast_1,4": 1,
                         "music_0,3": 1}
        fill_with_zeros(good_solution, jobs, max_time)

        # 'breakfast' tasks are out of order
        bad_order_solution = {"breakfast_0,1": 1, "breakfast_1,0": 1,
                              "music_0,0": 1}
        fill_with_zeros(bad_order_solution, jobs, max_time)

        self.assertTrue(jss.csp.check(good_solution))
        self.assertFalse(jss.csp.check(bad_order_solution))


class TestJSSExactSolverResponse(unittest.TestCase):
    def compare(self, response, expected):
        """Comparing response to expected results
        """
        # Comparing variables found in sample and expected
        expected_keys = set(expected.keys())
        sample_keys = set(response.keys())
        common_keys = expected_keys & sample_keys   # TODO: NO AUX
        different_keys = sample_keys - expected_keys  # sample_keys is a superset

        # Check that common variables match
        for key in common_keys:
            if is_auxiliary_variable(key):
                continue

            self.assertEqual(response[key], expected[key], "Failed on key {}".format(key))

        # Check that non-existent 'sample' variables are 0
        for key in different_keys:
            if is_auxiliary_variable(key):
                continue

            self.assertEqual(response[key], 0, "Failed on key {}".format(key))

    def test_tinySchedule(self):
        jobs = {"a": [(1, 1), (2, 1)],
                "b": [(2, 1)]}
        max_time = 2

        # Get exact sample from Job Shop Scheduler BQM
        jss = JobShopScheduler(jobs, max_time)
        bqm = jss.get_bqm()
        response = ExactSolver().sample(bqm)
        response_sample = next(response.samples())

        # Verify that response_sample obeys constraints
        self.assertTrue(jss.csp.check(response_sample))

        # Create expected solution
        expected = {"a_0,0": 1, "a_1,1": 1, "b_0,0": 1}

        # Compare variable values
        self.compare(response_sample, expected)

    def test_largerSchedule(self):
        jobs = {'small1': [(1, 1), (1, 1)],
                'small2': [(2, 2)],
                'longJob': [(0, 1), (1, 1), (2, 1)]}
        max_time = 4

        # Get exact sample from Job Shop Scheduler BQM
        jss = JobShopScheduler(jobs, max_time)
        bqm = jss.get_bqm()
        response = ExactSolver().sample(bqm)
        response_sample = next(response.samples())

        # Verify that response_sample obeys constraints
        self.assertTrue(jss.csp.check(response_sample))

        # Create expected solution
        expected = {"small1_0,0": 1,
                    "small1_1,2": 1,
                    "small2_0,0": 1,
                    "longJob_0,0": 1, "longJob_1,1": 1, "longJob_2,2": 1}

        # Compare variable values
        self.compare(response_sample, expected)

    def test_simple_schedule_more_machines(self):
        jobs = {"j0": [(0, 1)],
                "j1": [(1, 1)],
                "j2": [(2, 1)]}
        max_time = 3

        # Get JSS BQM
        scheduler = JobShopScheduler(jobs, max_time)
        bqm = scheduler.get_bqm()

        # Expected solution
        expected = {"j0_0,0": 1,
                    "j1_0,0": 1,
                    "j2_0,0": 1}
        fill_with_zeros(expected, jobs, max_time)
        expected_energy = get_energy(expected, bqm)

        # Sampled solution
        # response = EmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=10000)
        # response_sample, sample_energy, _, chain_break_fraction = next(response.data())
        # print("Chain Break Fraction: ", chain_break_fraction)
        # response = SimulatedAnnealingSampler().sample(bqm, num_reads=2000, beta_range=[0.01, 10])
        response = ExactSolver().sample(bqm)
        response_sample, sample_energy, _ = next(response.data())

        # Print response
        self.assertTrue(scheduler.csp.check(response_sample))
        self.assertEqual(expected_energy, sample_energy)
        self.compare(response_sample, expected)


class TestGetJssBqm(unittest.TestCase):
    def test_get_jss_bqm(self):
        jobs = {"sandwich": [("bread", 1), ("roast_beef", 1)],
                "french_toast": [("egg", 1), ("bread", 1)]}
        max_time = 3

        bqm = get_jss_bqm(jobs, max_time)
        self.assertIsInstance(bqm, BinaryQuadraticModel)

    def test_stitch_kwargs(self):
        """Ensure stitch_kwargs is being passed through get_jss_bqm to dwavebinarycsp.stitch
        """
        jobs = {"sandwich": [("bread", 1), ("roast_beef", 1)],
                "french_toast": [("egg", 1), ("bread", 1)]}
        max_time = 3

        # Verify that reasonable stitch args result in a BQM
        good_stitch_kwargs = {"max_graph_size": 6, "min_classical_gap": 1.5}
        bqm = get_jss_bqm(jobs, max_time, good_stitch_kwargs)
        self.assertIsInstance(bqm, BinaryQuadraticModel)

        # ImpossibleBQM should be raised, as the max_graph size is too small
        bad_stitch_kwargs = {"max_graph_size": 0}
        self.assertRaises(ImpossibleBQM, get_jss_bqm, jobs, max_time, bad_stitch_kwargs)


class TestGetBqm(unittest.TestCase):
    def test_stitch_kwargs(self):
        """Ensure stitch_kwargs is being passed to dwavebinarycsp.stitch
        """
        jobs = {"sandwich": [("bread", 1), ("roast_beef", 1)],
                "french_toast": [("egg", 1), ("bread", 1)]}
        max_time = 3

        # Verify that reasonable stitch args result in a BQM
        good_stitch_kwargs = {"max_graph_size": 6, "min_classical_gap": 1.5}
        scheduler = JobShopScheduler(jobs, max_time)
        bqm = scheduler.get_bqm(good_stitch_kwargs)
        self.assertIsInstance(bqm, BinaryQuadraticModel)

        # ImpossibleBQM should be raised, as the max_graph size is too small
        bad_stitch_kwargs = {"max_graph_size": 0}
        scheduler = JobShopScheduler(jobs, max_time)
        self.assertRaises(ImpossibleBQM, scheduler.get_bqm, bad_stitch_kwargs)

    def test_time_dependent_biases(self):
        """Test that the time-dependent biases that encourage short schedules are applied
        appropriately
        """
        jobs = {"j1": [("m1", 2), ("m2", 2)],
                "j2": [("m1", 2)]}

        # Create mock object for stitch(..) output
        linear = {'j1_0,0': -2.0, 'j1_0,1': -2.0, 'j1_0,2': -2.0,
                  'j1_1,2': -2.0, 'j1_1,3': -2.0, 'j1_1,4': -2.0,
                  'j2_0,0': -6.0, 'j2_0,1': -6.0, 'j2_0,2': -6.0, 'j2_0,3': -6.0, 'j2_0,4': -6.0,
                  'aux0': -8.0}
        quadratic = {('j1_0,0', 'j1_0,1'): 4.0, ('j1_0,0', 'j1_0,2'): 4.0, ('j1_0,0', 'j2_0,0'): 4,
                     ('j1_0,0', 'j2_0,1'): 2, ('j1_0,1', 'j1_0,2'): 4.0, ('j1_0,1', 'j1_1,2'): 2,
                     ('j1_0,1', 'j2_0,1'): 4, ('j1_0,1', 'j2_0,2'): 2, ('j1_0,1', 'j2_0,0'): 2,
                     ('j1_0,2', 'j1_1,2'): 2, ('j1_0,2', 'j1_1,3'): 2, ('j1_0,2', 'j2_0,2'): 4,
                     ('j1_0,2', 'j2_0,3'): 2, ('j1_0,2', 'j2_0,1'): 2, ('j1_1,2', 'j1_1,3'): 4.0,
                     ('j1_1,2', 'j1_1,4'): 4.0, ('j1_1,3', 'j1_1,4'): 4.0, ('j2_0,3', 'j2_0,2'): 4,
                     ('j2_0,3', 'j2_0,4'): 4.0, ('j2_0,3', 'j2_0,0'): 4.0, ('j2_0,3', 'j2_0,1'): 4,
                     ('j2_0,3', 'aux0'): 4.0, ('j2_0,2', 'j2_0,4'): 4.0, ('j2_0,2', 'j2_0,0'): 4.0,
                     ('j2_0,2', 'j2_0,1'): 4.0, ('j2_0,2', 'aux0'): 4.0, ('j2_0,4', 'j2_0,0'): 4.0,
                     ('j2_0,4', 'j2_0,1'): 4.0, ('j2_0,4', 'aux0'): 4.0, ('j2_0,0', 'j2_0,1'): 4.0,
                     ('j2_0,0', 'aux0'): 4.0, ('j2_0,1', 'aux0'): 4.0}
        vartype = dwavebinarycsp.BINARY
        mock_stitched_bqm = BinaryQuadraticModel(linear, quadratic, 14.0, vartype)

        scheduler = JobShopScheduler(jobs)
        with patch.object(dwavebinarycsp, "stitch", return_value=mock_stitched_bqm):
            bqm = scheduler.get_bqm()

        # Check linear biases
        # Note: I have grouped the tests by job-task and am comparing the linear biases between
        #   adjacent times. Tasks that finish before or at the lowerbound of the optimal schedule
        #   are not penalized with an additional bias; hence all these task-times should have the
        #   same bias.
        #   For example, the 0th task of job 2 (aka 'j2_0') will have the same linear
        #   bias for times 0 through 2 because with these start times, the task would complete
        #   before the optimal schedule lowerbound. (i.e. "j2_0,0", "j2_0,1", "j2_0,2" all have the
        #   same linear biases). The remaining task-times will have increasing biases with time, in
        #   this way, the shortest schedule is encouraged.
        self.assertEqual(bqm.linear['j2_0,0'], bqm.linear['j2_0,1'])
        self.assertEqual(bqm.linear['j2_0,1'], bqm.linear['j2_0,2'])
        self.assertLess(bqm.linear['j2_0,2'], bqm.linear['j2_0,3'])
        self.assertLess(bqm.linear['j2_0,3'], bqm.linear['j2_0,4'])

        self.assertEqual(bqm.linear['j1_0,0'], bqm.linear['j1_0,1'])
        self.assertEqual(bqm.linear['j1_0,1'], bqm.linear['j1_0,2'])

        self.assertLess(bqm.linear['j1_1,2'], bqm.linear['j1_1,3'])
        self.assertLess(bqm.linear['j1_1,3'], bqm.linear['j1_1,4'])

        # Check quadratic biases
        # Specifically, quadratic biases should not be penalized when encouraging shorter schedules
        # Note: Unable to simply compare dicts as BQM's quadraticview may re-order dict tuple-keys;
        #   hence, we're comparing BQM adjacencies.
        bqm_with_unchanged_quadratic = BinaryQuadraticModel({}, quadratic, 0, vartype)
        self.assertEqual(bqm.adj, bqm_with_unchanged_quadratic.adj)


if __name__ == "__main__":
    unittest.main()
