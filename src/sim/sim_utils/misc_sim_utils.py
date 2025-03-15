import json
import sys
import threading

from concordia.utils import html as html_lib
from IPython import display

file_lock = threading.Lock()
import datetime
from collections.abc import Callable

import numpy as np
from concordia.associative_memory import (
    associative_memory,
    formative_memories,
)
from concordia.clocks import game_clock
from concordia.language_model import language_model


# def write_item(out_item, output_filename):
#     with file_lock:
#         with open(output_filename, "a") as f:
#             print(json.dumps(out_item), file=f)  # adds the new line character
def write_item(out_item, output_filename):
    try:
        with file_lock:
            with open(output_filename, "a") as f:
                json_str = json.dumps(out_item)  # Separate this step for debugging
                print(json_str, file=f)
                print(
                    f"Successfully wrote item with type: {out_item.get('event_type')}"
                )  # Debug print
    except Exception as e:
        print(f"Error in write_item: {e}")
        print(
            f"Problem item: {type(out_item)}, keys: {out_item.keys() if isinstance(out_item, dict) else 'not a dict'}"
        )


class EventLogger:
    def __init__(self, event_type, output_filename):
        self.episode_idx = None
        self.output_filename = output_filename
        self.type = event_type
        self.dummy = None

    def log(self, log_data):
        if isinstance(log_data, list):
            for log_item in log_data:
                log_item["episode"] = self.episode_idx
                log_item["event_type"] = self.type
                write_item(log_item, self.output_filename)
        else:
            log_data["episode"] = self.episode_idx
            log_data["event_type"] = self.type
            if self.type == "action":
                log_data["data"]["suggested_action"] = self.dummy

            write_item(log_data, self.output_filename)


def read_token_data(file_path):
    try:
        with open(file_path) as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        return {"prompt_tokens": 0, "completion_tokens": 0}


def post_analysis(env, model, agents, roles, store_data, output_rootname):
    memories = {}
    for agent in agents:
        if roles[agent_agent_name]:
            memories[agent._agent_name] = store_data[agent._agent_name]

    all_gm_memories = env.memory.retrieve_recent(k=10000, add_time=True)

    detailed_story = "\n".join(all_gm_memories)
    print("len(detailed_story): ", len(detailed_story))
    # print(detailed_story)

    episode_summary = model.sample_text(
        f"Sequence of events:\n{detailed_story}"
        "\nNarratively summarize the above temporally ordered "
        "sequence of events. Write it as a news report. Summary:\n",
        max_tokens=3500,
        terminators=(),
    )
    print(episode_summary)

    # Summarise the perspective of each agent
    agent_logs = []
    agent_log_names = []
    for agent in agents:
        name = agent._agent_name
        detailed_story = "\n".join(
            memories[agent._agent_name].retrieve_recent(k=1000, add_time=True)
        )
        summary = ""
        summary = model.sample_text(
            f"Sequence of events that happened to {name}:\n{detailed_story}"
            "\nWrite a short story that summarises these events.\n",
            max_tokens=3500,
            terminators=(),
        )

        all_agent_mem = memories[agent._agent_name].retrieve_recent(k=1000, add_time=True)
        all_agent_mem = ["Summary:", summary, "Memories:", *all_agent_mem]
        agent_html = html_lib.PythonObjectToHTMLConverter(all_agent_mem).convert()
        agent_logs.append(agent_html)
        agent_log_names.append(f"{name}")

    # ## Build and display HTML log of the experiment
    gm_mem_html = html_lib.PythonObjectToHTMLConverter(all_gm_memories).convert()

    tabbed_html = html_lib.combine_html_pages(
        [gm_mem_html, *agent_logs],
        ["GM", *agent_log_names],
        summary=episode_summary,
        title="Mastodon experiment",
    )

    tabbed_html = html_lib.finalise_html(tabbed_html)
    with open(output_rootname + "_summary.html", "w", encoding="utf-8") as f:
        f.write(tabbed_html)

    display.HTML(tabbed_html)


def rebuild_from_saved_checkpoint(
    load_from_checkpoint_path: str,
    agents,
    roles,
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    embedder: Callable[[str], np.ndarray],
    update_time_interval: datetime.timedelta | None = None,
    memory_importance: Callable[[str], float] | None = None,
):
    # new_agents = []
    # for it, agent_input, agent in enumerate(zip(agent_data, agents, strict=False)):
    #     agent_json_path = agent_json_paths[it]
    #     module_path = (
    #         "sim_setting." + agent_input["role_dict"]["module_path"]
    #         if agent_input["role_dict"]["name"] != "exogeneous"
    #         else "agent_utils.exogenous_agent"
    #     )
    #     json_data = agent_json_path
    #     new_agent = module_path.rebuild_from_json(
    #         json_data,
    #         config,
    #         model,
    #         memory,
    #         clock,
    #         embedder,
    #     )
    #     new_agents.append(new_agent)
    # clock = None
    # model = None

    # return agents, clock, model
    return None


class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()  # Optional: ensures output is written immediately

    def flush(self):
        self.terminal.flush()
        self.file.flush()
