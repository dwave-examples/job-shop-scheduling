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

from __future__ import annotations

from enum import EnumMeta

import dash_mantine_components as dmc
from dash import dcc, html

from demo_configs import (
    CLASSICAL_TAB_LABEL,
    DESCRIPTION,
    DWAVE_TAB_LABEL,
    MAIN_HEADER,
    SCENARIOS,
    SHOW_CQM,
    SOLVER_TIME,
    THUMBNAIL,
)
from src.demo_enums import HybridSolverType, Model, SolverType

THEME_COLOR = "#2d4376"


def dropdown(label: str, id: str, options: list) -> html.Div:
    """Dropdown element for option selection.

    Args:
        label: The title that goes above the dropdown.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
    """
    return html.Div(
        className="dropdown-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            dmc.Select(
                id=id,
                data=options,
                value=options[0]["value"],
                allowDeselect=False,
            ),
        ],
    )


def checklist(label: str, id: str, options: list, values: list, inline: bool = True) -> html.Div:
    """Checklist element for option selection.

    Args:
        label: The title that goes above the checklist.
        id: A unique selector for this element.
        options: A list of dictionaries of labels and values.
        values: A list of values that should be preselected in the checklist.
        inline: Whether the options of the checklist are displayed beside or below each other.
    """
    return html.Div(
        className="checklist-wrapper",
        children=[
            dmc.CheckboxGroup(
                id=id,
                className=f"checklist{' checklist--inline' if inline else ''}",
                label=label,
                value=values,
                children=dmc.Group(
                    [
                        dmc.Checkbox(
                            label=option["label"], value=option["value"], color=THEME_COLOR
                        )
                        for option in options
                    ],
                ),
            ),
        ],
    )


def input(label: str, id: str, configs: dict, type: str = "number") -> html.Div:
    """Input element for either text or number input.

    Args:
        label: The title that goes above the input.
        id: A unique selector for this element.
        configs: A dictionary of configurations for the input element.
        type: The type of input, either "number" or "text".
    """
    return html.Div(
        className="input-wrapper",
        children=[
            html.Label(label, htmlFor=id),
            (
                dmc.TextInput(
                    id=id,
                    **configs,
                )
                if type == "text"
                else dmc.NumberInput(
                    id=id,
                    **configs,
                )
            ),
        ],
    )


def generate_options(options: list | EnumMeta | dict) -> list[dict]:
    """Format options for dropdowns, checklists, radios, etc.

    Args:
        options: A list, EnumMeta, or dictionary of options to format.

    Returns:
        A list of dictionaries with "label" and "value" keys for each option.
    """
    if isinstance(options, EnumMeta):
        return [{"label": option.label, "value": f"{option.value}"} for option in options]

    if isinstance(options, dict):
        return [{"label": f"{key}", "value": f"{value}"} for key, value in options.items()]

    return [{"label": f"{option}", "value": f"{option}"} for option in options]


def generate_settings_form() -> html.Div:
    """Generate settings for selecting the scenario, model, and solver.

    Returns:
        A Div containing the settings for selecting the scenario, model, and solver.
    """

    scenario_options = generate_options(SCENARIOS)
    model_options = generate_options(Model)

    return html.Div(
        className="settings",
        children=[
            dropdown(
                "Scenario (jobs x resources)",
                "scenario-select",
                scenario_options,
            ),
            html.Div(
                dropdown(
                    "Model",
                    "model-select",
                    model_options,
                ), className="" if SHOW_CQM else "display-none"
            ),
            dmc.CheckboxGroup(
                id="solver-select",
                label="Solvers",
                value=[f"{SolverType.HYBRID.value}", f"{SolverType.MIP.value}"],
                children=dmc.Group(
                    [
                        dmc.Checkbox(
                            label=SolverType.HYBRID.label,
                            value=f"{SolverType.HYBRID.value}",
                            color=THEME_COLOR,
                        ),
                        dmc.RadioGroup(
                            children=dmc.Group(
                                [
                                    dmc.Radio(s.label, value=f"{s.value}", color=THEME_COLOR)
                                    for s in HybridSolverType
                                ]
                            ),
                            id="hybrid-select",
                            value=f"{HybridSolverType.STRIDE.value}",
                            size="sm",
                            deselectable=False,
                            style={"display": "none"},
                        ),
                        dmc.Checkbox(
                            label=SolverType.MIP.label,
                            value=f"{SolverType.MIP.value}",
                            color=THEME_COLOR,
                        ),
                    ],
                ),
            ),
            input(
                "Solver Time Limit (seconds)",
                "solver-time-limit",
                SOLVER_TIME,
            ),
        ],
    )


def generate_run_buttons() -> html.Div:
    """Run and cancel buttons to run the optimization."""
    return html.Div(
        id="button-group",
        children=[
            html.Button("Run Optimization", id="run-button", className="button"),
            html.Button(
                "Cancel Optimization",
                id="cancel-button",
                className="button",
                style={"display": "none"},
            ),
        ],
    )


def generate_graph(graph: str) -> html.Div:
    """Generates graph."""
    return html.Div(
        className="graph",
        children=[
            dcc.Graph(
                id=f"{graph}-gantt-chart",
                responsive=True,
                config={"displayModeBar": False},
            ),
        ],
    )


def generate_solution_tab(title: str, tab: str, index: int) -> dmc.TabsPanel:
    """Generates solution tab containing, solution graphs, sort functionality, and
    problem details dropdown.

    Returns:
        dmc.TabsPanel: A Tab containing the solution graph and problem details.
    """
    return dmc.TabsPanel(
        value=f"{tab}-tab",
        tabIndex=11 + index,
        children=[
            html.Div(
                className="solution-card",
                children=[
                    html.Div(
                        className="gantt-chart-card",
                        children=[
                            html.Div(
                                className="gantt-heading",
                                children=[
                                    html.Div(
                                        [
                                            html.H2(
                                                title,
                                                className="gantt-title",
                                            ),
                                            html.H4(
                                                ["Makespan: ", html.Span(id=f"{tab}-makespan")],
                                                className="makespan",
                                            ),
                                        ]
                                    ),
                                ],
                            ),
                            html.Div(
                                className="graph-wrapper",
                                children=[
                                    generate_graph(tab),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def create_interface():
    """Create the main application interface."""
    return html.Div(
        id="app-container",
        children=[
            html.A(  # Skip link for accessibility
                "Skip to main content",
                href="#main-content",
                id="skip-to-main",
                className="skip-link",
                tabIndex=1,
            ),
            dcc.Store("last-selected-solvers"),
            dcc.Store("last-selected-hybrid-solver"),
            dcc.Store("running-dwave"),
            dcc.Store("running-classical"),
            # Settings and results columns
            html.Main(
                className="columns-main",
                id="main-content",
                children=[
                    # Left column
                    html.Div(
                        id={"type": "to-collapse-class", "index": 0},
                        className="left-column",
                        children=[
                            html.Div(
                                className="left-column-layer-1",  # Fixed width Div to collapse
                                children=[
                                    html.Div(
                                        className="left-column-layer-2",  # Padding and content wrapper
                                        children=[
                                            html.Div(
                                                [
                                                    html.H1(MAIN_HEADER),
                                                    html.P(DESCRIPTION),
                                                ],
                                                className="title-section",
                                            ),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        html.Div(
                                                            [
                                                                generate_settings_form(),
                                                                generate_run_buttons(),
                                                            ],
                                                            className="settings-and-buttons",
                                                        ),
                                                        className="settings-and-buttons-wrapper",
                                                    ),
                                                    # Left column collapse button
                                                    html.Div(
                                                        html.Button(
                                                            id={
                                                                "type": "collapse-trigger",
                                                                "index": 0,
                                                            },
                                                            className="left-column-collapse",
                                                            title="Collapse sidebar",
                                                            children=[
                                                                html.Div(className="collapse-arrow")
                                                            ],
                                                            **{"aria-expanded": "true"},
                                                        ),
                                                    ),
                                                ],
                                                className="form-section",
                                            ),
                                        ],
                                    )
                                ],
                            ),
                        ],
                    ),
                    # Right column
                    html.Div(
                        className="right-column",
                        children=[
                            dmc.Tabs(
                                id="tabs",
                                value="input-tab",
                                color="white",
                                children=[
                                    html.Header(
                                        className="banner",
                                        children=[
                                            html.Nav(
                                                [
                                                    dmc.TabsList(
                                                        [
                                                            dmc.TabsTab("Input", value="input-tab"),
                                                            dmc.TabsTab(
                                                                DWAVE_TAB_LABEL,
                                                                value="dwave-tab",
                                                                id="dwave-tab",
                                                                disabled=True,
                                                            ),
                                                            dmc.TabsTab(
                                                                CLASSICAL_TAB_LABEL,
                                                                value="mip-tab",
                                                                id="mip-tab",
                                                                disabled=True,
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                            html.Img(src=THUMBNAIL, alt="D-Wave logo"),
                                        ],
                                    ),
                                    dmc.TabsPanel(
                                        value="input-tab",
                                        tabIndex="12",
                                        children=[
                                            html.Div(
                                                className="gantt-chart-card",
                                                children=[
                                                    html.Div(
                                                        className="gantt-heading",
                                                        children=[
                                                            html.H2(
                                                                "Unscheduled Jobs and Resources",
                                                                className="gantt-title",
                                                            ),
                                                        ],
                                                    ),
                                                    dcc.Loading(
                                                        parent_className="graph-wrapper",
                                                        type="circle",
                                                        delay_show=300,
                                                        color=THEME_COLOR,
                                                        children=[
                                                            dcc.Graph(
                                                                id="unscheduled-gantt-chart",
                                                                responsive=True,
                                                                config={"displayModeBar": False},
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    generate_solution_tab("Quantum Hybrid Solver", "dwave", 1),
                                    generate_solution_tab("Classical Solver (COIN-OR)", "mip", 2),
                                ],
                            )
                        ],
                    ),
                ],
            ),
        ],
    )
