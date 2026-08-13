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

import pathlib
import time
from typing import NamedTuple

import dash
import plotly.graph_objs as go
from dash import MATCH, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from demo_configs import CLASSICAL_TAB_LABEL, DWAVE_TAB_LABEL, RESOURCE_NAMES, SHOW_CQM
from src.demo_enums import HybridSolverType, Model, SolverType
from src.generate_charts import generate_gantt_chart, get_empty_figure, get_minimum_task_times
from src.job_shop_scheduler import run_shop_scheduler, run_stride
from src.model_data import JobShopData

BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("input").resolve()


@dash.callback(
    Output({"type": "to-collapse-class", "index": MATCH}, "className"),
    Output({"type": "collapse-trigger", "index": MATCH}, "aria-expanded"),
    inputs=[
        Input({"type": "collapse-trigger", "index": MATCH}, "n_clicks"),
        State({"type": "to-collapse-class", "index": MATCH}, "className"),
    ],
    prevent_initial_call=True,
)
def toggle_left_column(collapse_trigger: int, to_collapse_class: str) -> tuple[str, str]:
    """Toggles a 'collapsed' class that hides and shows some aspect of the UI.

    Args:
        collapse_trigger: The (total) number of times a collapse button has been clicked.
        to_collapse_class: Current class name of the thing to collapse, 'collapsed' if not
            visible, empty string if visible.

    Returns:
        A tuple containing:

        - str: The new class name of the thing to collapse.
        - str: The aria-expanded value.
    """

    classes = to_collapse_class.split(" ") if to_collapse_class else []
    if "collapsed" in classes:
        classes.remove("collapsed")
        return " ".join(classes), "true"
    return to_collapse_class + " collapsed" if to_collapse_class else "collapsed", "false"


@dash.callback(
    Output("hybrid-select", "style"),
    Input("solver-select", "value"),
)
def update_solvers_selected(selected_solvers: list[str]) -> dict:
    """Hide Stride/CQM selector when Hybrid is unselected. Not applicable when SHOW_CQM is False.

    Args:
        selected_solvers (list[str]): Currently selected solvers.

    Returns:
        dict: Style for the hybrid select wrapper.
    """
    if SHOW_CQM and f"{SolverType.HYBRID.value}" in selected_solvers:
        return {}

    return {"display": "none"}


@dash.callback(
    Output("solver-select", "disabled"),
    Output("solver-select", "value"),
    Output("hybrid-select", "disabled"),
    Output("hybrid-select", "value"),
    Output("last-selected-solvers", "data"),
    Output("last-selected-hybrid-solver", "data"),
    inputs=[
        Input("model-select", "value"),
        State("solver-select", "value"),
        State("hybrid-select", "value"),
        State("last-selected-solvers", "data"),
        State("last-selected-hybrid-solver", "data"),
    ],
    prevent_initial_call=True,
)
def update_solver_options(
    model: int, selected_solvers: list[str], hybrid_selected_solver: str, last_selected_solvers: list[str], last_selected_hybrid_solver: str
) -> tuple[bool, list[str], bool, str, list[str], str]:
    """Hides and shows classical solver option using 'hide-classic' class.

    Args:
        model_value: Currently selected model from model-select dropdown.
        selected_solvers: Currently selected solvers.
        hybrid_selected_solver: Currently selected hybrid solver.
        last_selected_solvers: Previously selected solvers.
        last_selected_hybrid_solver: Previously selected hybrid solver.

    Returns:
        A tuple containing:

        - bool: Whether the solver-select checklist should be disabled.
        - list[str]: Unselects MIP and selects Hybrid or updates to previously selected solvers.
        - bool: Whether the hybrid-select checklist should be disabled.
        - str: Which hybrid solver to select.
        - list[str]: Updates last_selected_solvers with the list of solvers that were selected
        before updating.
        - str: Updates last_selected_hybrid_solver with the hybrid solver that was selected
        before updating.
    """
    model = Model(int(model))
    print(hybrid_selected_solver)
    print(type(hybrid_selected_solver))

    if model is Model.QM:
        return True, [f"{SolverType.HYBRID.value}"], True, f"{HybridSolverType.CQM.value}", selected_solvers, hybrid_selected_solver
    return False, last_selected_solvers, False, last_selected_hybrid_solver, dash.no_update, dash.no_update


class UpdateTabLoadingStateReturn(NamedTuple):
    """Return type for the ``update_tab_loading_state`` callback function."""

    dwave_tab_label: str = DWAVE_TAB_LABEL
    mip_tab_label: str = CLASSICAL_TAB_LABEL
    dwave_tab_disabled: str = dash.no_update
    mip_tab_disabled: str = dash.no_update
    run_button_style: dict = {}
    cancel_button_style: dict = {"display": "none"}
    running_dwave: bool = False
    running_classical: bool = False
    active_tab: str = dash.no_update


@dash.callback(
    Output("dwave-tab", "children", allow_duplicate=True),
    Output("mip-tab", "children", allow_duplicate=True),
    Output("dwave-tab", "disabled", allow_duplicate=True),
    Output("mip-tab", "disabled", allow_duplicate=True),
    Output("run-button", "style", allow_duplicate=True),
    Output("cancel-button", "style", allow_duplicate=True),
    Output("running-dwave", "data", allow_duplicate=True),
    Output("running-classical", "data", allow_duplicate=True),
    Output("tabs", "value"),
    [
        Input("run-button", "n_clicks"),
        Input("cancel-button", "n_clicks"),
        State("solver-select", "value"),
        State("hybrid-select", "value"),
    ],
    prevent_initial_call=True,
)
def update_tab_loading_state(
    run_click: int, cancel_click: int, solvers: list[str], hybrid_solver: str
) -> UpdateTabLoadingStateReturn:
    """Updates the tab loading state after the run button
    or cancel button has been clicked.

    Args:
        run_click: The number of times the run button has been clicked.
        cancel_click: The number of times the cancel button has been clicked.
        solvers: The list of selected solvers.
        hybrid_solver: The selected hybrid solver.
    Returns:
        A NamedTuple, UpdateTabLoadingStateReturn, containing:

        - dwave_tab_label: The label for the D-Wave tab.
        - mip_tab_label: The label for the Classical tab.
        - dwave_tab_disabled: True if D-Wave tab should be disabled, False otherwise.
        - mip_tab_disabled: True if Classical tab should be disabled, False otherwise.
        - run_button_style: Style for the run button.
        - cancel_button_style: Style for the cancel button.
        - running_dwave: Whether CQM/Stride solver is running.
        - running_classical: Whether MIP is running.
        - active_tab: The value of the tab that should be active.
    """

    if ctx.triggered_id == "run-button" and run_click > 0:
        run_cqm = int(hybrid_solver) is HybridSolverType.CQM.value
        run_stride = int(hybrid_solver) is HybridSolverType.STRIDE.value
        run_mip = f"{SolverType.MIP.value}" in solvers
        run_dwave = run_cqm or run_stride

        return UpdateTabLoadingStateReturn(
            dwave_tab_label="Loading..." if run_dwave else dash.no_update,
            mip_tab_label="Loading..." if run_mip else dash.no_update,
            dwave_tab_disabled=True if run_dwave else dash.no_update,
            mip_tab_disabled=True if run_mip else dash.no_update,
            run_button_style={"display": "none"},
            cancel_button_style={},
            running_dwave=run_dwave,
            running_classical=run_mip,
            active_tab="input-tab",
        )

    if ctx.triggered_id == "cancel-button" and cancel_click > 0:
        return UpdateTabLoadingStateReturn()

    raise PreventUpdate


@dash.callback(
    Output("run-button", "style"),
    Output("cancel-button", "style"),
    background=True,
    inputs=[
        Input("running-dwave", "data"),
        Input("running-classical", "data"),
    ],
    prevent_initial_call=True,
)
def update_button_visibility(running_dwave: bool, running_classical: bool) -> tuple[dict, dict]:
    """Updates the visibility of the run and cancel buttons.

    Args:
        running_dwave: Whether the D-Wave solver is running.
        running_classical: Whether the Classical solver is running.

    Returns:
        A tuple containing:

        - dict: Run button style.
        - dict: Cancel button style.
    """
    if not running_classical and not running_dwave:
        return {}, {"display": "none"}

    return {"display": "none"}, {}


class RunOptimizationHybridReturn(NamedTuple):
    """Return type for the ``run_optimization_hybrid`` callback function."""

    gantt_chart: go.Figure = dash.no_update
    makespan: int = 0
    tab_classname: str = ""
    tab_label: str = DWAVE_TAB_LABEL
    tab_disabled: bool = dash.no_update
    running_dwave: bool = False


@dash.callback(
    Output("dwave-gantt-chart", "figure"),
    Output("dwave-makespan", "children"),
    Output("dwave-tab", "className"),
    Output("dwave-tab", "children"),
    Output("dwave-tab", "disabled"),
    Output("running-dwave", "data"),
    background=True,
    inputs=[
        Input("run-button", "n_clicks"),
        State("model-select", "value"),
        State("solver-select", "value"),
        State("hybrid-select", "value"),
        State("scenario-select", "value"),
        State("solver-time-limit", "value"),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_optimization_hybrid(
    run_click: int, model: int, solvers: list[str], hybrid_solver: str, scenario: str, time_limit: int
) -> RunOptimizationHybridReturn:
    """Runs optimization using the D-Wave hybrid solver.

    Args:
        run_click: The number of times the run button has been clicked.
        model: The model to use for the optimization.
        solvers: The solvers that have been selected.
        hybrid_solver: The hybrid solver that have been selected.
        scenario: The scenario to use for the optimization.
        time_limit: The time limit for the optimization.

    Returns:
        A NamedTuple, RunOptimizationHybridReturn, containing:

        - gantt_chart: Gantt chart for the D-Wave hybrid solver.
        - makespan: Makespan for the D-Wave hybrid solver.
        - tab_classname: Class name for the D-Wave tab.
        - tab_label: The label for the D-Wave tab.
        - tab_disabled: True if D-Wave tab should be disabled, False otherwise.
        - running_dwave: Whether D-Wave solver is running.
    """
    if f"{SolverType.HYBRID.value}" not in solvers:
        return RunOptimizationHybridReturn()

    model = Model(int(model))
    model_data = JobShopData()

    running_cqm = int(hybrid_solver) is HybridSolverType.CQM.value

    model_data.load_from_file(DATA_PATH.joinpath(scenario), resource_names=RESOURCE_NAMES)

    results = run_shop_scheduler(
        model_data,
        solver="cqm" if running_cqm else "stride",
        allow_quadratic_constraints=(model is Model.QM),
        solver_time_limit=time_limit,
    )

    return RunOptimizationHybridReturn(
        gantt_chart=generate_gantt_chart(results),
        makespan=results["Finish"].max(),
        tab_classname="tab-success",
        tab_disabled=False,
    )


class RunOptimizationMipReturn(NamedTuple):
    """Return type for the ``run_optimization_mip`` callback function."""

    gantt_chart: go.Figure = dash.no_update
    makespan: int = 0
    tab_classname: str = ""
    tab_label: str = CLASSICAL_TAB_LABEL
    tab_disabled: bool = dash.no_update
    running_classical: bool = False


@dash.callback(
    Output("mip-gantt-chart", "figure"),
    Output("mip-makespan", "children"),
    Output("mip-tab", "className"),
    Output("mip-tab", "children"),
    Output("mip-tab", "disabled"),
    Output("running-classical", "data"),
    background=True,
    inputs=[
        Input("run-button", "n_clicks"),
        State("solver-select", "value"),
        State("scenario-select", "value"),
        State("solver-time-limit", "value"),
    ],
    cancel=[Input("cancel-button", "n_clicks")],
    prevent_initial_call=True,
)
def run_optimization_mip(
    run_click: int, solvers: list[int], scenario: str, time_limit: int
) -> RunOptimizationMipReturn:
    """Runs optimization using the COIN-OR Branch-and-Cut solver.

    Args:
        run_click: The number of times the run button has been clicked.
        solvers: The solvers that have been selected.
        scenario: The scenario to use for the optimization.
        time_limit: The time limit for the optimization.

    Returns:
        A NamedTuple, RunOptimizationMipReturn, containing:

        - gantt_chart: Gantt chart for the Classical solver.
        - makespan: Makespan for the Classical solver.
        - tab_classname: Class name for the Classical tab.
        - tab_label: The label for the Classical tab.
        - tab_disabled: True if Classical tab should be disabled, False otherwise.
        - running_classical: Whether Classical solver is running.
    """
    if f"{SolverType.MIP.value}" not in solvers:
        return RunOptimizationMipReturn()

    start = time.perf_counter()
    model_data = JobShopData()
    filename = scenario

    model_data.load_from_file(DATA_PATH.joinpath(filename), resource_names=RESOURCE_NAMES)

    results = run_shop_scheduler(
        model_data,
        solver="mip",
        allow_quadratic_constraints=False,
        solver_time_limit=time_limit,
    )

    if results.empty:
        return RunOptimizationMipReturn(
            gantt_chart=get_empty_figure("No solution found for Classical solver"),
            tab_classname="tab-fail",
            tab_disabled=False,
        )

    return RunOptimizationMipReturn(
        gantt_chart=generate_gantt_chart(results),
        makespan=results["Finish"].max(),
        tab_classname="tab-success",
        tab_disabled=False,
    )


@dash.callback(
    Output("unscheduled-gantt-chart", "figure"),
    Input("scenario-select", "value"),
)
def generate_unscheduled_gantt_chart(scenario: str) -> go.Figure:
    """Generates a Gantt chart of the unscheduled tasks for the given scenario.

    Args:
        scenario: The name of the scenario; must be a key in SCENARIOS.

    Returns:
        A Plotly figure object with the input data.
    """
    model_data = JobShopData()
    model_data.load_from_file(DATA_PATH.joinpath(scenario), resource_names=RESOURCE_NAMES)
    df = get_minimum_task_times(model_data)
    fig = generate_gantt_chart(df)
    return fig
