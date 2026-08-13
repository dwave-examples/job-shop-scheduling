# Copyright 2026 D-Wave
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

from enum import Enum

from demo_configs import SHOW_CQM

class SolverType(Enum):
    HYBRID = 0
    MIP = 1

    @property
    def label(self):
        return {
            SolverType.HYBRID: "Quantum Hybrid" if SHOW_CQM else "Stride Hybrid Solver",
            SolverType.MIP: "Classical Solver (COIN-OR Branch-and-Cut)",
        }[self]


class HybridSolverType(Enum):
    STRIDE = 0
    CQM = 1

    @property
    def label(self):
        return {
            HybridSolverType.STRIDE: "Stride",
            HybridSolverType.CQM: "CQM",
        }[self]


class Model(Enum):
    MIP = 0
    QM = 1

    @property
    def label(self):
        return {
            Model.MIP: "Mixed Integer Model",
            Model.QM: "Mixed Integer Quadratic Model",
        }[self]
