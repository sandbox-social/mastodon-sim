import argparse
import base64
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import dash
import dash_cytoscape as cyto
import networkx as nx
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots

cyto.load_extra_layouts()
from io import StringIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
print("project root: " + str(PROJECT_ROOT))
from sim.analysis_utils.output_proc_utils import post_process_output

# ---------------------Example-specific code
PROBE_LABEL = "VotePref"
custom_names = ["Bill Fredrickson", "Bradley Carter"]
line_colors = {
    "Bill Fredrickson": "#1f77b4",
    "Bradley Carter": "#ff7f0e",
    "did not vote": "#000000",
}
node_colors = {"Bill Fredrickson": "#1f77b4", "Bradley Carter": "#ff7f0e", "Other": "#808080"}


def probe_plot_preprocessing(probe_data):
    candidate1_votes_over_time = []
    candidate2_votes_over_time = []
    non_votes_over_time = []
    for ep in sorted(probe_data.keys()):
        ep_votes = probe_data[ep]
        total_ep_votes = len(ep_votes)
        candidate1_votes = sum(
            1 for vote in ep_votes.values() if vote == custom_names[0].split(" ")[0]
        )
        candidate2_votes = sum(
            1 for vote in ep_votes.values() if vote == custom_names[1].split(" ")[0]
        )
        candidate1_votes_over_time.append(
            (candidate1_votes / total_ep_votes) * 100 if total_ep_votes > 0 else 0
        )
        candidate2_votes_over_time.append(
            (candidate2_votes / total_ep_votes) * 100 if total_ep_votes > 0 else 0
        )
        non_votes_over_time.append(
            ((total_ep_votes - candidate2_votes - candidate1_votes) / total_ep_votes) * 100
            if total_ep_votes > 0
            else 0
        )

    graphs_data = [
        {
            "label": "for " + custom_names[0].split(" ")[0],
            "data": candidate1_votes_over_time,
            "color": line_colors[custom_names[0]],
        },
        {
            "label": "for " + custom_names[1].split(" ")[0],
            "data": candidate2_votes_over_time,
            "color": line_colors[custom_names[1]],
        },
        {
            "label": "did not vote",
            "data": non_votes_over_time,
            "color": line_colors["did not vote"],
        },
    ]

    title_label = "Vote Distribution Over Time"
    yaxis_label = "Vote Percentage"
    return graphs_data, title_label, yaxis_label


# -------------------------------------------------------------


# data loading
def stream_filtered_jsonl(
    content_string: str, selected_name: str, selected_episode: int
) -> Generator[dict[Any, Any], None, None]:
    """
    Stream and filter JSONL content line by line, only yielding matching records.

    Args:
        content_string: Base64 encoded content string
        selected_name: Name to filter by
        selected_episode: Episode index to filter by

    Yields
    ------
        Dict: Parsed JSON objects that match the filter criteria
    """
    # Handle the data URI format if present
    if "," in content_string:
        _, content_string = content_string.split(",", 1)

    if not content_string:
        print("content string is empty")
    # Create a text stream from the decoded content
    decoded = base64.b64decode(content_string).decode("utf-8")
    stream = StringIO(decoded)

    # Process and filter the stream line by line
    for line in stream:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        try:
            record = json.loads(line)
            # Only yield records that match our filters
            if (selected_name is None or record.get("agent_name") == selected_name) and (
                selected_episode is None or record.get("episode_idx") == selected_episode
            ):
                yield record
        except json.JSONDecodeError as e:
            print(f"Error processing JSONL line: {e!s}")
            continue


def convert_linebreaks(string):
    lst = string.split("\n")
    item = html.Br()
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


def create_display(entry):
    first_line = entry["prompt"].split("\n")[0]
    return html.Details(
        [
            html.Summary(f"Prompt: {first_line}..."),
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong("Full Prompt: "),
                            html.Span(convert_linebreaks(entry["prompt"])),
                        ],
                        style={"backgroundColor": "#e8f0fe", "padding": "5px", "margin": "5px"},
                    ),
                    html.Div(
                        [html.Strong("Output: "), html.Span(convert_linebreaks(entry["output"]))],
                        style={"backgroundColor": "#e2f7e1", "padding": "5px", "margin": "5px"},
                    ),
                    *[
                        html.Div([html.Strong(f"{k}: "), html.Span(convert_linebreaks(str(v)))])
                        for k, v in entry.items()
                        if k not in {"prompt", "output"}
                    ],
                ]
            ),
        ],
        style={"backgroundColor": "#f4f4f4", "padding": "10px", "margin": "5px"},
    )


def compute_positions(graph):
    pos = nx.kamada_kawai_layout(graph, scale=750)
    scaled_pos = {}
    for node, (x, y) in pos.items():
        scaled_pos[node] = {"x": x, "y": y}

    return scaled_pos


# Serialization function to convert complex data structures into JSON-serializable format
def serialize_data(
    follow_graph, interactions_by_episode, active_users_by_episode, toots, probe_data
):
    return {
        "nodes": list(follow_graph.nodes),
        "edges": list(follow_graph.edges),
        "interactions_by_episode": interactions_by_episode,
        "active_users_by_episode": {k: list(v) for k, v in active_users_by_episode.items()},
        "toots": toots,
        "probe_data": probe_data,
    }


# Deserialization function to convert JSON-serializable data back into original structures
def deserialize_data(serialized):
    follow_graph = nx.DiGraph()
    follow_graph.add_nodes_from(serialized["nodes"])
    follow_graph.add_edges_from(serialized["edges"])

    # Convert episode keys back to integers
    for k, v in serialized["probe_data"].items():
        print(k)
    interactions_by_episode = {int(k): v for k, v in serialized["interactions_by_episode"].items()}
    active_users_by_episode = {
        int(k): set(v) for k, v in serialized["active_users_by_episode"].items()
    }
    toots = serialized["toots"]
    probe_data = {k: v for k, v in serialized["probe_data"].items()}
    return follow_graph, interactions_by_episode, active_users_by_episode, toots, probe_data


def get_target_user(row):
    if row.label == "post":
        target_user = row.source_user
    elif row.label == "like_toot" or row.label == "boost_toot":
        target_user = row.data["target_user"]
    elif row.label == "reply":
        target_user = row.data["reply_to"]["target_user"]
    return target_user


def get_int_dict(int_df):
    past = dict(
        zip(
            ["post", "like_toot", "boost_toot", "reply"],
            ["posted", "liked", "boosted", "replied"],
            strict=False,
        )
    )
    int_df["int_data"] = int_df.apply(
        lambda x: {
            "action": past[x.label],
            "episode": x.episode,
            "source": x.source_user,
            "target": get_target_user(x),
            "toot_id": str(x.data["toot_id"]),
        },
        axis=1,
    )
    int_df.int_data = int_df.apply(
        lambda x: x.int_data | {"parent_toot_id": str(x.data["reply_to"]["toot_id"])}
        if x.label == "reply"
        else x.int_data,
        axis=1,
    )
    return int_df.groupby("episode")["int_data"].apply(list).to_dict()


def get_toot_dict(int_df):
    past = dict(
        zip(
            ["post", "like_toot", "boost_toot", "reply"],
            ["posted", "liked", "boosted", "replied"],
            strict=False,
        )
    )
    text_df = int_df.loc[(int_df.label == "post") | (int_df.label == "reply"), :].reset_index(
        drop=True
    )

    # handle Nones as toot_ids by appending an index
    no_toot_id = text_df.data.apply(lambda x: x["toot_id"] is None)
    text_df["no_toot_id_idx"] = -1
    text_df.loc[no_toot_id, "no_toot_id_idx"] = range(no_toot_id.sum())
    text_df.loc[no_toot_id, "data"] = text_df.loc[no_toot_id, :].apply(
        lambda x: x.data | {"toot_id": "None" + str(x.no_toot_id_idx)}, axis=1
    )

    text_df["toot_id"] = text_df.data.apply(lambda x: x["toot_id"])
    text_df = text_df.set_index("toot_id")
    text_df["text_data"] = text_df.apply(
        lambda x: {"user": x.source_user, "action": past[x.label], "content": x.data["post_text"]},
        axis=1,
    )
    text_df.text_data = text_df.apply(
        lambda x: x.text_data | {"parent_toot_id": x.data["reply_to"]["toot_id"]}
        if x.label == "reply"
        else x.text_data,
        axis=1,
    )

    return text_df.text_data.to_dict()


def load_data(input_var):
    if len(input_var) < 500:
        df = pd.read_json(input_var, lines=True)
    else:
        df = pd.read_json(StringIO(input_var), lines=True)

    names = list(df.source_user.unique())
    name_dict = dict(zip([n.split()[0] for n in names], names, strict=False))
    print(name_dict)

    # replace all first name occurence with fullnames in target field
    def replace_full(data):
        if "target_user" in data:
            if len(data["target_user"].split()) == 1:
                data.update(target_user=name_dict[data["target_user"]])
        return data

    df.loc[:, "data"] = df.loc[:, "data"].apply(lambda x: replace_full(x))

    # make sure all tootids are strings
    def get_toot_id(data):
        if "toot_id" in data:
            data["toot_id"] = str(data["toot_id"])
        return data

    df["data"] = df.data.apply(get_toot_id)

    probe_df, int_df, edge_df = post_process_output(df)

    # probe_data
    probe_data = (
        probe_df.loc[probe_df.label == PROBE_LABEL, ["source_user", "response", "episode"]]
        .groupby("episode")
        .apply(lambda x: dict(zip(x.source_user, x.response, strict=False)))
        .to_dict()
    )

    # final follow network
    follow_graph = nx.from_pandas_edgelist(
        edge_df, "source_user", "target_user", create_using=nx.DiGraph()
    )  # invalid in presence of unfollows (in which case use episodewise_graphbuild)

    # active users with episode keys
    active_users_by_episode = int_df.groupby("episode")["source_user"].apply(set).to_dict()

    # interaction data
    int_dict = get_int_dict(int_df.copy())

    # toot_data
    toot_dict = get_toot_dict(int_df.copy())

    return follow_graph, int_dict, active_users_by_episode, toot_dict, probe_data


# Main entry point
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run the Dash app with specific data files.")
    parser.add_argument(
        "--output_file",
        type=str,
        nargs="?",
        default=None,
        help="The path to the output log file.",
    )

    args = parser.parse_args()

    # Initialize variables
    if args.output_file:
        # Load the data using the files passed as arguments
        (follow_graph, interactions_by_episode, active_users_by_episode, toots, probe_data) = (
            load_data(args.output_file)
        )

        # Compute positions
        all_positions = compute_positions(follow_graph)

        layout = {"name": "preset", "positions": all_positions}

        # Serialize the initial data
        serialized_initial_data = serialize_data(
            follow_graph, interactions_by_episode, active_users_by_episode, toots, probe_data
        )
    else:
        # No initial data provided
        serialized_initial_data = None

    app = dash.Dash(__name__)

    # Create the layout with conditional sections
    app.layout = html.Div(
        [
            html.H1(
                id="dashboard-title", children="Social Sandbox Dashboard"
            ),  # Add this line before the dashboard div
            # Store component to hold serialized data
            dcc.Store(id="data-store", data=serialized_initial_data),
            # Upload Screen
            html.Div(
                id="upload-screen",
                children=[
                    # Upload Output Log
                    html.Div(
                        [
                            html.Label(
                                "Upload an output file:",
                                style={
                                    "font-size": "18px",
                                    "font-weight": "bold",
                                    "margin-bottom": "10px",
                                    "color": "#555555",
                                    "text-align": "center",
                                },
                            ),
                            dcc.Upload(
                                id="upload-app-logger",
                                children=html.Div(
                                    [
                                        "Drag and Drop or ",
                                        html.A(
                                            "Select Files",
                                            style={
                                                "color": "#1a73e8",
                                                "text-decoration": "underline",
                                            },
                                        ),
                                    ]
                                ),
                                style={
                                    "width": "100%",
                                    "max-width": "400px",
                                    "height": "80px",
                                    "lineHeight": "80px",
                                    "borderWidth": "2px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "10px",
                                    "textAlign": "center",
                                    "background-color": "#f9f9f9",
                                    "cursor": "pointer",
                                    "margin": "0 auto",  # Center the upload box
                                    "transition": "border 0.3s ease-in-out",
                                },
                                multiple=False,
                            ),
                        ],
                        style={"width": "100%", "max-width": "500px", "margin-bottom": "30px"},
                    ),
                    # Submit Button
                    html.Button(
                        "Submit",
                        id="submit-button",
                        n_clicks=0,
                        style={
                            "width": "200px",
                            "height": "50px",
                            "font-size": "18px",
                            "background-color": "#4CAF50",  # Green background
                            "color": "white",
                            "border": "none",
                            "border-radius": "8px",
                            "cursor": "pointer",
                            "box-shadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
                            "transition": "background-color 0.3s ease, transform 0.2s ease",
                            "margin-bottom": "20px",
                            "align-self": "center",  # Center the button
                        },
                    ),
                    # Error Message
                    html.Div(
                        id="upload-error-message",
                        style={
                            "color": "red",
                            "textAlign": "center",
                            "margin-top": "20px",
                            "font-size": "16px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "height": "100vh",
                    "background-color": "#f0f2f5",  # Light gray background for better contrast
                    "padding": "20px",
                },
            ),
            # Dashboard
            html.Div(
                id="dashboard",  # Added 'dashboard' id here
                children=[
                    # Upload components and Submit button added at the bottom
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Upload an output file:"),
                                    dcc.Upload(
                                        id="upload-app-logger-dashboard",
                                        children=html.Div([html.A("Select Files")]),
                                        style={
                                            "width": "50px",
                                            "height": "24px",
                                            "lineHeight": "15px",
                                            "borderWidth": "2px",
                                            "borderStyle": "dashed",
                                            "borderRadius": "2px",
                                            "textAlign": "center",
                                            "background-color": "#f0f0f0",
                                            "cursor": "pointer",
                                            "padding": "7px",
                                        },
                                        multiple=False,
                                    ),
                                ],
                                className="upload-component",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Upload Files",
                                        id="upload-button-dashboard",
                                        n_clicks=0,
                                        style={
                                            "width": "70px",
                                            "height": "40px",
                                            "lineHeight": "15px",
                                            "padding": "7px",
                                            "margin-top": "20px",
                                        },
                                    ),
                                ],
                                className="upload-button-container",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justify-content": "space-around",
                            # 'flex-wrap': 'wrap',
                            "gap": "20px",
                            "margin-top": "20px",
                            "margin-bottom": "20px",
                        },
                        id="dashboard-upload-section",
                    ),
                    # Line graphs container
                    html.Div(
                        [
                            # probe data line graph
                            dcc.Graph(
                                id="probe-data-line",
                                config={"displayModeBar": False},
                                style={
                                    "height": "220px",
                                    "width": "32%",
                                    "display": "inline-block",
                                },
                            ),
                            # Interactions count line graph
                            dcc.Graph(
                                id="interactions-line-graph",
                                config={"displayModeBar": False},
                                style={
                                    "height": "170px",
                                    "width": "32%",
                                    "display": "inline-block",
                                },
                            ),
                            # Heatmap graph
                            dcc.Graph(
                                id="heatmap-graph",
                                config={"displayModeBar": False},
                                style={
                                    "height": "250px",
                                    "width": "32%",
                                    "display": "inline-block",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justify-content": "space-between",
                            "margin-top": "15px",
                            "margin-bottom": "20px",
                        },
                    ),
                    # Main content: Cytoscape graph and interactions window
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Dropdown(
                                        id="name-selector",
                                        options=[],  # To be populated by callback
                                        value=None,
                                        placeholder="Select Name",
                                        clearable=True,
                                        style={
                                            "padding": "10px",
                                            "font-size": "16px",
                                            "font-weight": "bold",
                                            "width": "200px",
                                            "z-index": "1000",  # Ensure it stays on top of the Cytoscape graph
                                        },
                                    ),
                                    # dcc.Dropdown(
                                    #     id="mode-dropdown",
                                    #     options=[
                                    #         {"label": "Universal View", "value": "normal"},
                                    #         {"label": "Active View", "value": "focused"},
                                    #     ],
                                    #     value="normal",
                                    #     clearable=False,
                                    #     style={
                                    #         "padding": "10px",
                                    #         "font-size": "16px",
                                    #         "font-weight": "bold",
                                    #         "width": "200px",
                                    #         "z-index": "1000",  # Ensure it stays on top of the Cytoscape graph
                                    #     },
                                    # ),
                                ],
                                style={
                                    "position": "absolute",
                                    "top": "10px",  # Aligns at the top of the graph
                                    "left": "10px",  # Aligns on the left
                                    "display": "flex",
                                    "gap": "10px",  # Space between the two dropdowns
                                    "z-index": "1000",  # Ensure it's above the Cytoscape graph
                                },
                            ),
                            # Episode number display (top-right)
                            html.Div(
                                id="current-episode",
                                style={
                                    "position": "absolute",
                                    "top": "10px",
                                    "right": "10px",
                                    "padding": "10px",
                                    "font-size": "20px",
                                    "font-weight": "bold",
                                    "background-color": "#ffcc99",  # Optional: add a background color
                                    "z-index": "1000",  # Ensure it stays on top of the Cytoscape graph
                                },
                                children="",
                            ),
                            # Flex container for Cytoscape and Interactions Window
                            html.Div(
                                [
                                    cyto.Cytoscape(
                                        id="cytoscape-graph",
                                        elements=[],  # To be populated by callback
                                        layout={
                                            "name": "preset",
                                            "positions": {},
                                        },  # To be updated by callback
                                        style={
                                            "width": "100%",  # Initial width set to 100%
                                            "height": "500px",
                                            "background-color": "#e1e1e1",
                                            "transition": "width 0.5s",  # Smooth width transition
                                        },
                                        stylesheet=[
                                            {
                                                "selector": ".default_node",
                                                "style": {
                                                    "background-color": "#fffca0",
                                                    "label": "data(label)",
                                                    "color": "#000000",
                                                    "font-size": "20px",
                                                    "text-halign": "center",
                                                    "text-valign": "center",
                                                    "width": "70px",
                                                    "height": "70px",
                                                    "border-width": 6,
                                                    "border-color": "#000000",
                                                },
                                            },
                                            {
                                                "selector": ".follow_edge",
                                                "style": {
                                                    "curve-style": "bezier",
                                                    "target-arrow-shape": "triangle",
                                                    "opacity": 0.8,
                                                    "width": 2,
                                                    "line-color": "#FFFFFF",
                                                },
                                            },
                                            {
                                                "selector": ".interaction_edge",
                                                "style": {
                                                    "curve-style": "bezier",
                                                    "target-arrow-shape": "triangle",
                                                    "opacity": 0.8,
                                                    "width": 4,
                                                    "line-color": "#000000",
                                                    "visibility": "hidden",
                                                },
                                            },
                                            {
                                                "selector": ".interaction_edge:hover",
                                                "style": {
                                                    "label": "data(label)",
                                                    "font-size": "14px",
                                                    "color": "#000000",
                                                },
                                            },
                                            # Edge labels
                                            {
                                                "selector": "edge",
                                                "style": {
                                                    "label": "data(label)",
                                                    "text-rotation": "autorotate",
                                                    "text-margin-y": "-10px",
                                                    "font-size": "10px",
                                                    "color": "#000000",
                                                    "text-background-color": "#FFFFFF",
                                                    "text-background-opacity": 0.8,
                                                    "text-background-padding": "3px",
                                                },
                                            },
                                            # Specific styles for custom nodes
                                            {
                                                "selector": f'[id="{custom_names[0]}"]',
                                                "style": {
                                                    "background-color": "blue",
                                                    "border-color": "#000000",
                                                },
                                            },
                                            {
                                                "selector": f'[id="{custom_names[1]}"]',
                                                "style": {
                                                    "background-color": "orange",
                                                    "border-color": "#000000",
                                                },
                                            },
                                            # Highlighted Nodes (Added)
                                            {
                                                "selector": ".highlighted",
                                                "style": {
                                                    "background-color": "#98FF98",  # Mint color
                                                    "border-color": "#FF69B4",  # Hot pink border for visibility
                                                    "border-width": 4,
                                                },
                                            },
                                        ],
                                    ),
                                    # Interactions Window
                                    html.Div(
                                        [
                                            html.H3("Platform Interactions"),
                                            html.Div(
                                                id="interactions-window",
                                                style={"overflowY": "auto", "height": "580px"},
                                            ),
                                        ],
                                        style={
                                            "width": "0%",  # Initial width set to 0%
                                            "height": "600px",
                                            "padding": "10px",
                                            "border-left": "1px solid #ccc",
                                            "background-color": "#f9f9f9",
                                            "transition": "width 0.5s",  # Smooth width transition
                                            "overflow": "hidden",
                                        },
                                        id="interactions-container",
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "row",
                                    "height": "600px",
                                    "transition": "all 0.5s ease",  # Smooth transition for all properties
                                },
                            ),
                        ],
                        style={
                            "position": "relative",
                            "height": "600px",
                            "margin-top": "10px",
                            "margin-bottom": "20px",
                            "width": "70%",
                            "margin": "auto",
                        },
                    ),
                    html.Label("Select Episode:"),
                    # Episode slider
                    dcc.Slider(
                        id="episode-slider",
                        min=0,  # To be updated by callback
                        max=0,  # To be updated by callback
                        value=0,  # To be updated by callback
                        marks={},  # To be updated by callback
                        step=None,
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    # After dashboard-upload-section, add:
                    html.Hr(style={"margin": "20px 0"}),  # Separator
                    # JSONL Viewer Section
                    html.Div(
                        [
                            dcc.Upload(
                                id="upload-jsonl",
                                children=html.Button(
                                    "Upload prompts & responses file",
                                    style={
                                        "padding": "10px 20px",
                                        "fontSize": "16px",
                                        "backgroundColor": "#4CAF50",
                                        "color": "white",
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "cursor": "pointer",
                                    },
                                ),
                                multiple=False,
                                style={"margin": "20px 0", "textAlign": "center"},
                                max_size=-1,
                            ),
                            html.H2(
                                "Within-agent processing for selected agent and episode",
                                style={"textAlign": "center"},
                            ),
                            html.Div(id="jsonl-output"),
                            dcc.Store(id="jsonl-store"),
                        ],
                        id="jsonl-viewer-section",
                        style={"padding": "20px"},
                    ),
                ],
                style={"display": "none"},  # Initially hidden; shown when data is available
            ),
            # Hidden div for error messages (specific to dashboard uploads)
            html.Div(id="error-message", style={"color": "red", "textAlign": "center"}),
        ]
    )

    @app.callback(
        Output("jsonl-output", "children"),
        [
            Input("upload-jsonl", "contents"),
            Input("name-selector", "value"),
            Input("episode-slider", "value"),
        ],
        prevent_initial_call=True,
    )
    def process_jsonl_data(contents, selected_name, selected_episode):
        """Process JSONL data with streaming and filtering."""
        if contents is None:
            print("contents is None")
            return None
        if not contents:
            print("Contents is empty")

        print("streaming")
        try:
            # Stream and filter the data, collecting only matching records
            return [
                create_display(record)
                for record in stream_filtered_jsonl(contents, selected_name, selected_episode)
            ]

        except Exception as e:
            print(f"Error processing JSONL: {e!s}")
            return None

    @app.callback(
        [
            Output("upload-screen", "style"),
            Output("dashboard", "style"),
            Output("dashboard-upload-section", "style"),
            Output("name-selector", "options"),
        ],
        [Input("data-store", "data")],
    )
    def toggle_layout(data_store):
        if data_store and "nodes" in data_store and len(data_store["nodes"]) > 0:
            # Data is available; show dashboard and hide upload screen
            return (
                {"display": "none"},
                {"display": "block"},
                {"display": "flex"},
                [{"label": name, "value": name} for name in sorted(data_store["nodes"])],
            )
        # No data; show upload screen and hide dashboard
        return {"display": "flex"}, {"display": "none"}, {"display": "none"}, []

    # Combined Callback for Initial and Dashboard Uploads
    @app.callback(
        [
            Output("dashboard-title", "children"),
            Output("data-store", "data"),
            Output("upload-error-message", "children"),
            Output("error-message", "children"),
        ],
        [
            Input("submit-button", "n_clicks"),
            Input("upload-button-dashboard", "n_clicks"),
        ],
        [
            State("upload-app-logger", "contents"),
            State("upload-app-logger", "filename"),
            State("upload-app-logger-dashboard", "contents"),
            State("upload-app-logger-dashboard", "filename"),
            State("data-store", "data"),
        ],
    )
    def update_data(
        n_clicks_initial,
        n_clicks_dashboard,
        app_logger_contents_initial,
        app_logger_filename_initial,
        app_logger_contents_dashboard,
        app_logger_filename_dashboard,
        current_data,
    ):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        dashboard_title_with_filename = "Social Sandbox Dashboard: " + app_logger_filename_initial
        try:
            if triggered_id == "submit-button":
                if app_logger_contents_initial is not None:
                    # Process app_logger
                    content_type, content_string = app_logger_contents_initial.split(",")
                    decoded = base64.b64decode(content_string)
                    app_logger_string = decoded.decode("utf-8")
                    (
                        follow_graph_new,
                        interactions_by_episode_new,
                        active_users_by_episode_new,
                        toots_new,
                        probe_data_new,
                    ) = load_data(app_logger_string)

                    # Serialize the new data
                    serialized_new_data = serialize_data(
                        follow_graph_new,
                        interactions_by_episode_new,
                        active_users_by_episode_new,
                        toots_new,
                        probe_data_new,
                    )
                    # *** Add these two lines: parse and store the raw data ***
                    import io

                    raw_df = pd.read_json(io.StringIO(app_logger_string), lines=True)
                    serialized_new_data["raw_data"] = raw_df.to_dict(orient="records")
                    # *** End additional lines ***

                    return dashboard_title_with_filename, serialized_new_data, "", ""

                raise ValueError("Output Log file required.")

            if triggered_id == "upload-button-dashboard":
                if app_logger_contents_dashboard is not None:
                    # Process app_logger
                    content_type, content_string = app_logger_contents_dashboard.split(",")
                    decoded = base64.b64decode(content_string)
                    app_logger_string = decoded.decode("utf-8")
                    (
                        follow_graph_new,
                        interactions_by_episode_new,
                        active_users_by_episode_new,
                        toots_new,
                        probe_data_new,
                    ) = load_data(app_logger_string)

                    # Serialize the new data
                    serialized_new_data = serialize_data(
                        follow_graph_new,
                        interactions_by_episode_new,
                        active_users_by_episode_new,
                        toots_new,
                        probe_data_new,
                    )
                    # *** Add these lines to store the raw file data ***
                    import io

                    raw_df = pd.read_json(io.StringIO(app_logger_string), lines=True)
                    serialized_new_data["raw_data"] = raw_df.to_dict(orient="records")
                    # *** End additional lines ***

                    dashboard_title_with_filename = (
                        "Social Sandbox Dashboard: " + app_logger_filename_dashboard
                    )
                    return dashboard_title_with_filename, serialized_new_data, "", ""
                raise ValueError("Output Log files required for dashboard upload.")

            raise dash.exceptions.PreventUpdate

        except Exception as e:
            if triggered_id == "submit-button":
                return dash.no_update, f"Error uploading initial data: {e!s}", ""
            if triggered_id == "upload-button-dashboard":
                return dash.no_update, "", f"Error uploading dashboard data: {e!s}"
            return dash.no_update, "", ""

    @app.callback(Output("heatmap-graph", "figure"), Input("data-store", "data"))
    def update_heatmap(data_store):
        # If no data-store or no raw data is present, return a figure indicating so
        if not data_store or "raw_data" not in data_store:
            return go.Figure(data=[], layout=go.Layout(title="No data uploaded"))

        # Build a DataFrame from the raw uploaded records
        raw_data = data_store["raw_data"]
        df = pd.DataFrame(raw_data)

        # If the "data" column is stored as a string, parse it into a dict.
        if not df.empty and isinstance(df.iloc[0]["data"], str):
            df["data"] = df["data"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

        # Filter for records with event_type "action" that have a non-null suggested_action.
        dft = df[
            (df["event_type"] == "action")
            & (df["data"].apply(lambda x: x.get("suggested_action") is not None))
        ].copy()

        if dft.empty:
            return go.Figure(
                data=[], layout=go.Layout(title="No action records with a suggested_action found")
            )

        # If the suggested action equals "toot", convert it to "post"
        dft["suggested_action"] = dft["data"].apply(
            lambda x: "post" if x.get("suggested_action") == "toot" else x.get("suggested_action")
        )

        # Build a contingency table:
        # Rows: suggested_action, Columns: actual action taken (from the "label" column)
        contingency = pd.crosstab(dft["label"], dft["suggested_action"])

        # Create the heatmap figure using Plotly
        heatmap_fig = go.Figure(
            data=go.Heatmap(
                z=contingency.values,
                x=list(contingency.columns),
                y=list(contingency.index),
                colorscale="YlOrRd",
                colorbar=dict(title="Count"),
            )
        )

        heatmap_fig.update_layout(
            title={"text": "Action alignment distribution", "font": {"size": 14}},
            xaxis_title="Suggested Action",
            yaxis_title="Chosen Action",
            margin=dict(l=40, r=40, t=40, b=40),
            height=270,
        )

        return heatmap_fig

    # Callback to update the dashboard based on data-store
    @app.callback(
        [
            Output("cytoscape-graph", "elements"),
            Output("cytoscape-graph", "layout"),
            Output("cytoscape-graph", "stylesheet"),
            Output("probe-data-line", "figure"),
            Output("interactions-line-graph", "figure"),
            Output("current-episode", "children"),
            Output("interactions-window", "children"),  # Added Output
            Output("interactions-container", "style"),  # Added Output to control width
            Output("cytoscape-graph", "style"),  # Added Output to control width
            Output("episode-slider", "min"),
            Output("episode-slider", "max"),
            Output("episode-slider", "value"),
            Output("episode-slider", "marks"),
            Output("name-selector", "value"),
        ],
        [
            Input("episode-slider", "value"),
            # Input("mode-dropdown", "value"),
            Input("name-selector", "value"),  # Added Input
            Input("data-store", "data"),  # Added Input to trigger update on data change
        ],
    )
    def update_graph(selected_episode, selected_name, data_store):  # selected_mode,
        if not data_store:
            # If no data is present, return defaults
            return (
                [],  # elements
                {"name": "preset", "positions": {}},  # layout
                [],  # stylesheet
                {},  # probe data line
                {},  # interactions-line-graph
                "Episode: N/A",  # current-episode
                [],  # interactions-window
                {
                    "width": "0%",  # Collapsed width
                    "height": "600px",
                    "padding": "10px",
                    "border-left": "1px solid #ccc",
                    "background-color": "#f9f9f9",
                    "transition": "width 0.5s",  # Smooth width transition
                    "overflow": "hidden",
                },  # interactions-container
                {
                    "width": "100%",  # Full width
                    "height": "600px",
                    "background-color": "#e1e1e1",
                    "transition": "width 0.5s",  # Smooth width transition
                },  # cytoscape-style
                0,  # slider min
                0,  # slider max
                0,  # slider value
                {},  # slider marks
                None,  # name-selector value
            )

        # Deserialize the data_store.
        (follow_graph, interactions_by_episode, active_users_by_episode, toots, probe_data) = (
            deserialize_data(data_store)
        )

        # Compute positions based on the current follow_graph
        all_positions = compute_positions(follow_graph)
        layout = {"name": "preset", "positions": all_positions}

        # Build Cytoscape elements
        elements = [
            {
                "data": {"id": node, "label": node.split(" ")[0]},
                "classes": "default_node",
            }
            for node in follow_graph.nodes
        ] + [
            {
                "data": {
                    "source": src,
                    "target": tgt,
                },
                "classes": "follow_edge",
            }
            for src, tgt in follow_graph.edges
        ]

        # Add all interaction edges classified by the episode they belong to
        for episode, interactions in interactions_by_episode.items():
            for interaction in interactions:
                source = interaction["source"]
                target = interaction["target"]

                # Check if both source and target exist in the graph before creating the edge
                if source in follow_graph.nodes and target in follow_graph.nodes:
                    elements.append(
                        {
                            "data": {
                                "source": source,
                                "target": target,
                                "label": f"{interaction['action']}",
                            },
                            "classes": f"interaction_edge episode_{episode}",  # Classify edge by episode
                        }
                    )

        # Initialize the stylesheet
        stylesheet = [
            {
                "selector": ".default_node",
                "style": {
                    "background-color": "#fffca0",
                    "label": "data(label)",
                    "color": "#000000",
                    "font-size": "20px",
                    "text-halign": "center",
                    "text-valign": "center",
                    "width": "70px",
                    "height": "70px",
                    "border-width": 6,
                    "border-color": "#000000",
                },
            },
            {
                "selector": ".follow_edge",
                "style": {
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "target-arrow-color": "#FFFFFF",
                    "opacity": 0.8,
                    "width": 2,
                    "line-color": "#FFFFFF",
                },
            },
            {
                "selector": ".interaction_edge",
                "style": {
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "target-arrow-color": "#000000",
                    "opacity": 0.8,
                    "width": 4,
                    "line-color": "#000000",
                    "visibility": "hidden",
                },
            },
            {
                "selector": "edge",
                "style": {
                    "label": "data(label)",
                    "text-rotation": "autorotate",
                    "text-margin-y": "-10px",
                    "font-size": "10px",
                    "color": "#000000",
                    "text-background-color": "#FFFFFF",
                    "text-background-opacity": 0.8,
                    "text-background-padding": "3px",
                },
            },
            {
                "selector": ".interaction_edge:hover",
                "style": {
                    "label": "data(label)",
                    "font-size": "14px",
                    "color": "#000000",
                },
            },
            # Specific styles for custom nodes
            {
                "selector": f'[id="{custom_names[0]}"]',
                "style": {
                    "background-color": "blue",
                    "border-color": "#000000",
                },
            },
            {
                "selector": f'[id="{custom_names[1]}"]',
                "style": {
                    "background-color": "orange",
                    "border-color": "#000000",
                },
            },
            # Highlighted Nodes
            {
                "selector": ".highlighted",
                "style": {
                    "background-color": "#98FF98",  # Mint color
                    "border-color": "#FF69B4",  # Hot pink border for visibility
                    "border-width": 4,
                },
            },
        ]

        # Determine the sizing of the interactions window and Cytoscape graph
        interactions_content = []

        if selected_name:
            # Get interactions where source is selected_name in selected_episode
            interactions = [
                interaction
                for interaction in interactions_by_episode.get(selected_episode, [])
                if interaction["source"] == selected_name
            ]

            if interactions:
                for interaction in interactions:
                    action = interaction["action"]
                    if action in ["liked", "boosted"]:
                        toot_id = interaction["toot_id"]
                        content = toots.get(toot_id, {}).get("content", "No content available.")
                        user = toots.get(toot_id, {}).get("user", "No user available.")
                        interactions_content.append(
                            html.Div(
                                [
                                    html.H4(
                                        f"{action.capitalize()} a toot (ID: {toot_id}) by {user}"
                                    ),
                                    html.P(content),
                                ],
                                style={
                                    "border": "1px solid #ccc",
                                    "padding": "10px",
                                    "margin-bottom": "10px",
                                },
                            )
                        )
                    elif action == "replied":
                        parent_toot_id = interaction.get("parent_toot_id")
                        reply_toot_id = interaction.get("toot_id")
                        parent_content = toots.get(parent_toot_id, {}).get(
                            "content", "No content available."
                        )
                        reply_content = toots.get(reply_toot_id, {}).get(
                            "content", "No content available."
                        )
                        user = toots.get(parent_toot_id, {}).get("user", "No user available.")
                        interactions_content.append(
                            html.Div(
                                [
                                    html.H4(f"Replied to toot (ID: {parent_toot_id}) by {user}"),
                                    html.P(parent_content),
                                    html.H5(f"Reply (ID: {reply_toot_id}):"),
                                    html.P(reply_content),
                                ],
                                style={
                                    "border": "1px solid #ccc",
                                    "padding": "10px",
                                    "margin-bottom": "10px",
                                },
                            )
                        )
                    elif action == "posted":
                        toot_id = interaction["toot_id"]
                        content = toots.get(toot_id, {}).get("content", "No content available.")
                        interactions_content.append(
                            html.Div(
                                [
                                    html.H4(f"Posted a toot (ID: {toot_id})"),
                                    html.P(content),
                                ],
                                style={
                                    "border": "1px solid #ccc",
                                    "padding": "10px",
                                    "margin-bottom": "10px",
                                },
                            )
                        )
            else:
                interactions_content.append(
                    html.P("No interactions found for this agent in the selected episode.")
                )
        else:
            interactions_content.append(html.P("Select an agent to view their interactions."))

        if selected_name:
            interactions_style = {
                "width": "30%",  # Expanded width
                "height": "600px",
                "padding": "10px",
                "border-left": "1px solid #ccc",
                "background-color": "#f9f9f9",
                "transition": "width 0.5s",  # Smooth width transition
                "overflow": "auto",
            }
            cytoscape_style = {
                "width": "70%",  # Reduced width
                "height": "600px",
                "background-color": "#e1e1e1",
                "transition": "width 0.5s",  # Smooth width transition
            }
        else:
            interactions_style = {
                "width": "0%",  # Collapsed width
                "height": "600px",
                "padding": "10px",
                "border-left": "1px solid #ccc",
                "background-color": "#f9f9f9",
                "transition": "width 0.5s",  # Smooth width transition
                "overflow": "hidden",
            }
            cytoscape_style = {
                "width": "100%",  # Full width
                "height": "600px",
                "background-color": "#e1e1e1",
                "transition": "width 0.5s",  # Smooth width transition
            }

        # Highlight selected node and the nodes they follow
        if selected_name:
            # Find the nodes that the selected node follows (outgoing edges)
            follows = list(follow_graph.successors(selected_name))
            # Define the selector for the selected node and its followees
            if follows:
                highlight_selector = f'[id="{selected_name}"], ' + ", ".join(
                    [f'[id="{follow}"]' for follow in follows]
                )
            else:
                highlight_selector = f'[id="{selected_name}"]'

            # Apply the 'highlighted' class to the selected node and its followees
            stylesheet.append(
                {
                    "selector": highlight_selector,
                    "style": {
                        "background-color": "#98FF98",  # Mint color
                        "border-color": "#FF69B4",  # Hot pink border for visibility
                        "border-width": 4,
                    },
                }
            )

        # Show interaction edges for the selected episode
        for episode in interactions_by_episode.keys():
            visibility = "visible" if episode == selected_episode else "hidden"
            stylesheet.append(
                {
                    "selector": f".episode_{episode}",
                    "style": {"visibility": visibility},
                }
            )

        # Update node border colors based on probe_data
        episode_probe_data = probe_data.get(selected_episode, {})
        total_probe_data = len(episode_probe_data)

        for node in follow_graph.nodes:
            if node in episode_probe_data:
                probe_datum = episode_probe_data[node]
                node_label = node if node in custom_names else "Other"
                stylesheet.append(
                    {
                        "selector": f'[id="{node}"]',
                        "style": {"border-color": node_colors[node_label]},
                    }
                )

        # Create the line graph showing probe_data over time
        probe_data_episodes = sorted(probe_data.keys())

        probe_graphs_data, title_label, yaxis_label = probe_plot_preprocessing(probe_data)

        probe_data_line_fig = go.Figure()
        for graph_data in probe_graphs_data:
            probe_data_line_fig.add_trace(
                go.Scatter(
                    x=probe_data_episodes,
                    y=graph_data["data"],
                    mode="lines+markers",
                    name=graph_data["label"],
                    line=dict(color=graph_data["color"]),
                )
            )

        max_episode = max(list(interactions_by_episode.keys()))
        probe_data_line_fig.update_layout(
            title={"text": title_label, "font": {"size": 14}},
            xaxis={
                "title": {"text": "Episode", "font": {"size": 10}},
                "tickfont": {"size": 8},
                "range": [-1, max_episode + 1],
                "dtick": 1,
            },
            yaxis={
                "title": {"text": yaxis_label, "font": {"size": 10}},
                "tickfont": {"size": 8},
                "range": [0, 100],
            },
            height=200,
            margin=dict(l=40, r=40, t=20, b=10),
            legend=dict(orientation="h", x=0.5, y=-0.25, xanchor="center"),
        )

        # Create the line graph showing interactions over time
        interaction_types = ["liked", "boosted", "replied", "posted"]
        interactions_over_time = {interaction: [] for interaction in interaction_types}

        total_users = len(follow_graph.nodes) - 1

        active_user_fractions = []
        # int_episodes = list(active_users_by_episode.keys())
        int_episodes = sorted(interactions_by_episode.keys())
        for ep in int_episodes:
            num_active_users = len(active_users_by_episode[ep]) - 1  # dont count news agent

            counts = {interaction: 0 for interaction in interaction_types}

            # Count interactions
            for interaction in interactions_by_episode.get(ep, []):
                action = interaction["action"]
                if action in counts:
                    counts[action] += 1

            # Append counts to the respective lists
            for interaction in interaction_types:
                if interaction == "posted":
                    interactions_over_time[interaction].append(
                        (counts[interaction] - 1) / num_active_users if num_active_users > 0 else 0
                    )
                else:
                    interactions_over_time[interaction].append(
                        (counts[interaction]) / num_active_users if num_active_users > 0 else 0
                    )

            # Calculate active user fraction
            active_user_fraction = num_active_users / total_users if total_users > 0 else 0
            active_user_fractions.append(active_user_fraction)

        interactions_line_fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add normalized interaction traces to the primary y-axis
        interactions_line_fig.add_trace(
            go.Scatter(
                x=int_episodes,
                y=interactions_over_time["liked"],
                mode="lines+markers",
                name="Likes",
                line=dict(color="#2ca02c"),  # Green
                marker=dict(symbol="circle", size=6),
            ),
            secondary_y=False,
        )
        interactions_line_fig.add_trace(
            go.Scatter(
                x=int_episodes,
                y=interactions_over_time["boosted"],
                mode="lines+markers",
                name="Boosts",
                line=dict(color="#ff7f0e"),  # Orange
                marker=dict(symbol="square", size=6),
            ),
            secondary_y=False,
        )
        interactions_line_fig.add_trace(
            go.Scatter(
                x=int_episodes,
                y=interactions_over_time["replied"],
                mode="lines+markers",
                name="Replies",
                line=dict(color="#9467bd"),  # Purple
                marker=dict(symbol="diamond", size=6),
            ),
            secondary_y=False,
        )
        interactions_line_fig.add_trace(
            go.Scatter(
                x=int_episodes,
                y=interactions_over_time["posted"],
                mode="lines+markers",
                name="Posts",
                line=dict(color="#1f77b4"),  # Blue
                marker=dict(symbol="triangle-up", size=6),
            ),
            secondary_y=False,
        )

        # Add active users fraction trace to the secondary y-axis
        interactions_line_fig.add_trace(
            go.Scatter(
                x=int_episodes,
                y=active_user_fractions,
                mode="lines",
                name="Active User Fraction",
                line=dict(color="gray"),
            ),
            secondary_y=True,
        )

        # Define the y-axis range
        y_axis_range = [0, 1.5]

        # Update both y-axes
        interactions_line_fig.update_yaxes(
            title_text="Action Rate of Active Users",
            range=y_axis_range,
            secondary_y=False,
            showgrid=True,
            gridcolor="lightgray",
        )

        interactions_line_fig.update_yaxes(
            title_text="Active User Fraction",
            range=[0, 1],
            secondary_y=True,
            showgrid=False,  # Typically, grid lines are only on the primary y-axis
            gridcolor="lightgray",
        )

        interactions_line_fig.update_layout(
            title={"text": "Interactions Over Time", "font": {"size": 14}},
            xaxis={
                "title": {"text": "Episode", "font": {"size": 10}},
                "tickfont": {"size": 8},
                "range": [-1, max_episode + 1],
                "dtick": 1,
            },
            yaxis={
                "title": {"text": "Interactions/ Num. Agents", "font": {"size": 10}},
                "tickfont": {"size": 8},
            },
            height=200,
            margin=dict(l=40, r=40, t=20, b=10),
            showlegend=True,
            legend=dict(orientation="h", x=0.5, y=-0.25, xanchor="center"),
        )

        # Adjust the x-axis range to include all episodes
        interactions_line_fig.update_xaxes(range=[-1, max(int_episodes) + 1])

        # Update the name-selector dropdown options
        unique_names = sorted(follow_graph.nodes)
        name_options = [{"label": name, "value": name} for name in unique_names]

        # Set episode slider properties
        slider_min = min(int_episodes)
        slider_max = max(int_episodes)
        slider_value = selected_episode if selected_episode in int_episodes else slider_min
        slider_marks = {str(ep): f"{ep}" for ep in sorted(int_episodes)}

        # Return all outputs, including the interactions window content and styles
        return (
            elements,  # Updated elements
            layout,
            stylesheet,
            # fig,
            probe_data_line_fig,
            interactions_line_fig,
            f"Episode: {selected_episode}",  # Updated episode display
            interactions_content,
            interactions_style,
            cytoscape_style,
            slider_min,
            slider_max,
            slider_value,
            slider_marks,
            selected_name,
        )

    # Run the Dash app
    app.run_server(debug=True)
