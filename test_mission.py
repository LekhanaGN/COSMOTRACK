
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from backend.mission import (
    get_mission_overview
)


result = get_mission_overview(

    hours=6,

    max_objects=10,

    mode="demo",

    include_fusion=True
)


catalogue = result["catalogue"] or {}

fusion = result["fusion"] or {}

anomalies = result["anomalies"] or {}

evaluation = anomalies.get(
    "evaluation"
) or {}

scheduler = result["scheduler"] or {}

alerts = result["alert_feed"]

severity_counts = alerts[
    "severity_counts"
]


# ==========================================
# DASHBOARD STRUCTURE
# ==========================================

fig = make_subplots(

    rows=3,

    cols=2,

    specs=[

        [
            {"type": "indicator"},
            {"type": "indicator"}
        ],

        [
            {"type": "domain"},
            {"type": "xy"}
        ],

        [
            {"type": "xy"},
            {"type": "xy"}
        ]
    ],

    subplot_titles=(

        "Tracked Catalogue Objects",

        "EKF Position Error Improvement",

        "Orbital Catalogue Distribution",

        "Central Alert Severity",

        "Hybrid AI Evaluation",

        "Autonomous Observation Scheduling"
    ),

    vertical_spacing=0.12
)


# ==========================================
# TOTAL OBJECTS
# ==========================================

fig.add_trace(

    go.Indicator(

        mode="number",

        value=catalogue.get(
            "total_objects",
            0
        ),

        number={
            "font": {
                "color": "cyan",
                "size": 55
            }
        }
    ),

    row=1,
    col=1
)


# ==========================================
# EKF IMPROVEMENT
# ==========================================

if fusion.get("improvement_percent") is not None:

    fig.add_trace(

        go.Indicator(

            mode="number",

            value=fusion[
                "improvement_percent"
            ],

            number={
                "suffix": "%",
                "valueformat": ".2f",
                "font": {
                    "color": "lime",
                    "size": 55
                }
            }
        ),

        row=1,
        col=2
    )

else:

    fig.add_annotation(

        x=0.78,
        y=0.85,

        xref="paper",
        yref="paper",

        text="Fusion unavailable",

        showarrow=False
    )


# ==========================================
# CATALOGUE DISTRIBUTION
# ==========================================

fig.add_trace(

    go.Pie(

        labels=[
            "Satellites",
            "Debris",
            "Stations"
        ],

        values=[

            catalogue.get(
                "active_satellites",
                0
            ),

            catalogue.get(
                "debris_objects",
                0
            ),

            catalogue.get(
                "space_stations",
                0
            )
        ],

        hole=0.55,

        marker={
            "colors": [
                "cyan",
                "orange",
                "violet"
            ]
        },

        textinfo="label+value",

        showlegend=False
    ),

    row=2,
    col=1
)


# ==========================================
# ALERT SEVERITY
# ==========================================

levels = [
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW"
]

fig.add_trace(

    go.Bar(

        x=levels,

        y=[
            severity_counts.get(level, 0)
            for level in levels
        ],

        marker={
            "color": [
                "red",
                "orange",
                "yellow",
                "cyan"
            ]
        },

        showlegend=False
    ),

    row=2,
    col=2
)


# ==========================================
# HYBRID AI PERFORMANCE
# ==========================================

metric_keys = [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1"
]

ml_metrics = evaluation.get(
    "ml_only_metrics",
    {}
)

fig.add_trace(

    go.Bar(

        x=metric_names,

        y=[
            ml_metrics[key] * 100
            if key in ml_metrics
            else None

            for key in metric_keys
        ],

        name="Isolation Forest",

        marker_color="violet"
    ),

    row=3,
    col=1
)

fig.add_trace(

    go.Bar(

        x=metric_names,

        y=[
            evaluation[key] * 100
            if key in evaluation
            else None

            for key in metric_keys
        ],

        name="Hybrid detection",

        marker_color="cyan"
    ),

    row=3,
    col=1
)

fig.update_yaxes(

    title_text="Performance (%)",

    range=[0, 110],

    row=3,
    col=1
)


# ==========================================
# AUTONOMOUS SCHEDULING
# ==========================================

fig.add_trace(

    go.Bar(

        x=[
            "Scheduled",
            "Unscheduled"
        ],

        y=[

            scheduler.get(
                "total_scheduled"
            ),

            scheduler.get(
                "total_unscheduled"
            )
        ],

        marker_color=[
            "lime",
            "orange"
        ],

        showlegend=False
    ),

    row=3,
    col=2
)


# ==========================================
# FINAL LAYOUT
# ==========================================

fig.update_layout(

    title=(

        "COSMOTRACK | UNIFIED MISSION CONTROL"

        "<br><sup>"

        f"System: {result['system_status']}"

        f" | Mode: {result['mode'].upper()}"

        f" | Alerts: {alerts['total_alerts']}"

        " | SIMULATED SENSOR OPERATIONS"

        "</sup>"
    ),

    template="plotly_dark",

    height=1000,

    barmode="group",

    legend=dict(
        orientation="h",
        y=-0.08
    ),

    margin=dict(
        t=120,
        b=100
    )
)


# ==========================================
# TERMINAL SUMMARY
# ==========================================

print("=" * 65)

print("COSMOTRACK UNIFIED MISSION CONTROL")

print("=" * 65)

print(
    "System status:",
    result["system_status"]
)

print(
    "Tracked objects:",
    catalogue.get("total_objects")
)

print(
    "Total alerts:",
    alerts["total_alerts"]
)

print(
    "Scheduled observations:",
    scheduler.get("total_scheduled")
)

print("\nMODULE HEALTH")

for module, status in result[
    "module_status"
].items():

    print(
        module,
        "|",
        status["status"]
    )

print("\nCENTRAL ALERT FEED")

for alert in alerts["alerts"][:10]:

    print(
        alert["severity"],
        "|",
        alert["category"],
        "|",
        alert["title"]
    )

if result["warnings"]:

    print("\nWARNINGS")

    for warning in result["warnings"]:
        print(warning)


fig.write_html(

    "mission_dashboard.html",

    include_plotlyjs=True
)

fig.show()
