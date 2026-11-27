
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from backend.scheduler import (
    generate_schedule,
    parse_timestamp,
    PRIORITY_WEIGHTS
)


# ==========================================
# GENERATE SCHEDULE
# ==========================================

result = generate_schedule(
    hours=6,
    max_objects=10,
    max_tasks=10,
    mode="demo",
    include_fusion=True
)

schedule = result["schedule"]

candidates = result["candidates"]

planning_start = parse_timestamp(
    result["planning_start"]
)

planning_hours = result[
    "planning_hours"
]

COLORS = {
    "CRITICAL": "#EF4444",
    "HIGH": "#F59E0B",
    "MEDIUM": "#EAB308",
    "LOW": "#22B8CF"
}

SENSOR_COLORS = {
    "radar": "#22B8CF",
    "lidar": "#F59E0B",
    "optical": "#A78BFA"
}


# ==========================================
# DASHBOARD STRUCTURE
# ==========================================

fig = make_subplots(

    rows=2,

    cols=2,

    subplot_titles=(

        "Autonomous Observation Timeline",

        "Satellite Priority Ranking",

        "Priority Contribution Analysis",

        "Sensor Resource Allocation"
    ),

    horizontal_spacing=0.15,

    vertical_spacing=0.18
)


# ==========================================
# PANEL 1: OBSERVATION GANTT CHART
# ==========================================

sensor_positions = {
    "radar": 0,
    "lidar": 1,
    "optical": 2
}

for task in schedule:

    start = parse_timestamp(
        task["start_time"]
    )

    end = parse_timestamp(
        task["end_time"]
    )

    start_minutes = (
        start - planning_start
    ).total_seconds() / 60

    end_minutes = (
        end - planning_start
    ).total_seconds() / 60

    y = sensor_positions[
        task["sensor"]
    ]

    color = COLORS[
        task["priority_level"]
    ]

    hover = (
        f"Satellite: "
        f"{task['satellite_name']}"
        "<br>"
        f"Sensor: {task['sensor']}"
        "<br>"
        f"Priority: "
        f"{task['priority_score']}"
        "<br>"
        f"Start: {task['start_time']}"
        "<br>"
        f"End: {task['end_time']}"
        "<br>"
        f"{task['explanation']}"
    )

    fig.add_trace(

        go.Bar(

            x=[
                end_minutes
                - start_minutes
            ],

            base=[start_minutes],

            y=[y],

            orientation="h",

            width=0.55,

            marker=dict(
                color=color
            ),

            text=[
                task["satellite_name"]
            ],

            textposition="inside",

            hovertext=[hover],

            hoverinfo="text",

            showlegend=False
        ),

        row=1,

        col=1
    )

    # Marker makes short observation
    # tasks visible at a six-hour scale.

    fig.add_trace(

        go.Scatter(

            x=[
                (
                    start_minutes
                    + end_minutes
                ) / 2
            ],

            y=[y],

            mode="markers",

            marker=dict(
                color=color,
                size=10,
                symbol="diamond"
            ),

            hovertext=[hover],

            hoverinfo="text",

            showlegend=False
        ),

        row=1,

        col=1
    )

if not schedule:

    fig.add_annotation(

        x=planning_hours * 30,

        y=1,

        text="No eligible observation slots",

        showarrow=False,

        row=1,

        col=1
    )

fig.update_yaxes(

    tickvals=[0, 1, 2],

    ticktext=[
        "Radar",
        "LiDAR",
        "Optical"
    ],

    range=[-0.6, 2.6],

    row=1,

    col=1
)

fig.update_xaxes(

    title_text=(
        "Minutes from planning start"
    ),

    range=[
        0,
        planning_hours * 60
    ],

    row=1,

    col=1
)


# ==========================================
# PANEL 2: PRIORITY RANKING
# ==========================================

names = [

    item["satellite_name"]

    for item in candidates
]

scores = [

    item["priority_score"]

    for item in candidates
]

colors = [

    COLORS[
        item["priority_level"]
    ]

    for item in candidates
]

fig.add_trace(

    go.Bar(

        x=scores,

        y=names,

        orientation="h",

        marker=dict(
            color=colors
        ),

        text=[
            f"{score:.1f}"
            for score in scores
        ],

        textposition="outside",

        hovertext=[
            item["explanation"]
            for item in candidates
        ],

        showlegend=False
    ),

    row=1,

    col=2
)

fig.update_yaxes(

    autorange="reversed",

    row=1,

    col=2
)

fig.update_xaxes(

    title_text="Priority score",

    range=[0, 110],

    row=1,

    col=2
)


# ==========================================
# PANEL 3: PRIORITY CONTRIBUTIONS
# ==========================================

factor_colors = {
    "conjunction": "#EF4444",
    "anomaly": "#F59E0B",
    "uncertainty": "#A78BFA",
    "visibility": "#22C55E",
    "urgency": "#22B8CF"
}

for factor in PRIORITY_WEIGHTS:

    values = [

        item["contributions"][factor]

        for item in candidates
    ]

    fig.add_trace(

        go.Bar(

            x=names,

            y=values,

            name=factor.title(),

            marker=dict(
                color=factor_colors[
                    factor
                ]
            )
        ),

        row=2,

        col=1
    )

fig.update_xaxes(

    tickangle=-35,

    row=2,

    col=1
)

fig.update_yaxes(

    title_text="Priority contribution",

    range=[0, 100],

    row=2,

    col=1
)


# ==========================================
# PANEL 4: SENSOR ALLOCATION
# ==========================================

sensors = list(
    result[
        "sensor_utilisation"
    ].keys()
)

allocated_minutes = [

    result[
        "sensor_utilisation"
    ][sensor][
        "allocated_seconds"
    ] / 60

    for sensor in sensors
]

task_counts = [

    result[
        "sensor_utilisation"
    ][sensor]["tasks"]

    for sensor in sensors
]

fig.add_trace(

    go.Bar(

        x=[
            sensor.upper()
            for sensor in sensors
        ],

        y=allocated_minutes,

        marker=dict(

            color=[
                SENSOR_COLORS[sensor]
                for sensor in sensors
            ]
        ),

        text=[
            f"{count} tasks"
            for count in task_counts
        ],

        textposition="outside",

        showlegend=False
    ),

    row=2,

    col=2
)

fig.update_yaxes(

    title_text="Allocated minutes",

    rangemode="tozero",

    row=2,

    col=2
)


# ==========================================
# DASHBOARD LAYOUT
# ==========================================

fig.update_layout(

    title=(

        "COSMOTRACK | AUTONOMOUS "
        "OBSERVATION SCHEDULER"

        "<br><sup>"

        f"Candidates: "
        f"{result['total_candidates']}"

        f" | Scheduled: "
        f"{result['total_scheduled']}"

        f" | Unscheduled: "
        f"{result['total_unscheduled']}"

        f" | Violations: "
        f"{result['validation']['total_violations']}"

        " | SIMULATED SENSOR TASKING"

        "</sup>"
    ),

    template="plotly_dark",

    height=950,

    barmode="stack",

    legend=dict(
        orientation="h",
        y=-0.25
    ),

    margin=dict(
        t=120,
        b=180,
        l=100,
        r=50
    )
)


# ==========================================
# TERMINAL OUTPUT
# ==========================================

print("=" * 65)

print(
    "COSMOTRACK SCHEDULER PERFORMANCE"
)

print("=" * 65)

for key in [

    "total_candidates",

    "total_scheduled",

    "total_unscheduled",

    "schedule_success_percent",

    "urgent_coverage_percent"
]:

    print(
        f"{key}: {result[key]}"
    )

print(
    "Schedule valid:",
    result["validation"]["valid"]
)

print(
    "Constraint violations:",
    result["validation"][
        "total_violations"
    ]
)

print("\nEXPLAINABLE OBSERVATION TASKS")

for task in schedule:

    print("-" * 65)

    print(
        "Satellite:",
        task["satellite_name"]
    )

    print(
        "Sensor:",
        task["sensor"].upper()
    )

    print(
        "Priority:",
        task["priority_score"]
    )

    print(
        "Start:",
        task["start_time"]
    )

    print(
        "End:",
        task["end_time"]
    )

    print(
        "Explanation:",
        task["explanation"]
    )

print("\nUNSCHEDULED TARGETS")

for target in result["unscheduled"]:

    print(
        target["satellite_name"],
        "|",
        target["reason"]
    )


# ==========================================
# SAVE INTERACTIVE DASHBOARD
# ==========================================

fig.write_html(

    "scheduler_dashboard.html",

    include_plotlyjs=True
)

fig.show()
