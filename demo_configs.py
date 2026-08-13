# Copyright 2024 D-Wave Systems Inc.
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

"""This file stores input parameters for the app."""

import json

THUMBNAIL = "static/dwave_logo.svg"

APP_TITLE = "Job Shop Scheduling"
MAIN_HEADER = "Job Shop Scheduling"
DESCRIPTION = """\
Run the job shop scheduling problem for several different scenarios.
Explore the Gantt Chart for solution details.
"""

SHOW_CQM = True  # whether to show the CQM solver as an option in the app; set to False to hide it

CLASSICAL_TAB_LABEL = "Classical Results"
DWAVE_TAB_LABEL = "Hybrid Results"

# The list of scenarios that the user can choose from in the app.
# These can be found in the 'input' directory.
SCENARIOS = {
    "3x3": "instance3_3.txt",
    "5x5": "instance5_5.txt",
    "10x10": "instance10_10.txt",
    "15x15": "taillard15_15.txt",
    "20x15": "instance20_15.txt",
    "20x25": "instance20_25.txt",
    "30x30": "instance30_30.txt",
}

# solver time limits in seconds (value means default)
SOLVER_TIME = {
    "min": 5,
    "max": 300,
    "step": 5,
    "value": 5,
}

# The list of resources that the user can choose from in the app
RESOURCE_NAMES = json.load(open("./src/data/resource_names.json", "r"))["industrial"]
