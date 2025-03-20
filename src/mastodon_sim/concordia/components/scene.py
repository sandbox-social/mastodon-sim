# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A GameMaster that simulates a player's interaction with their phone."""

import re
import threading

# _PHONE_CALL_TO_ACTION = textwrap.dedent("""\
#   What action is {name} currently performing or has just performed
#   with their smartphone to best achieve their goal?
#   Consider their plan, but deviate if necessary.
#   Give a specific activity using one app. For example:
#   {name} uses/used the Chat app to send "hi, what's up?" to George.
#   """)
from html import unescape
from typing import Literal

import termcolor
from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import blank_memories
from concordia.clocks import game_clock
from concordia.document import interactive_document
from concordia.environment import game_master as game_master_lib
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains
from concordia.typing import agent, component
from concordia.typing.entity import OutputType

from mastodon_sim import mastodon_ops
from mastodon_sim.concordia.components import apps, logging
from mastodon_sim.concordia.components.apps import COLOR_TYPE
from sim.sim_utils.misc_sim_utils import ConfigStore

file_lock = threading.Lock()

DEFAULT_CALL_TO_SPEECH = (
    "Given the above, what is {name} likely to say next? Respond in"
    ' the format `{name} -- "..."` For example, '
    'Cristina -- "Hello! Mighty fine weather today, right?", '
    'Ichabod -- "I wonder if the alfalfa is ready to harvest", or '
    'Townsfolk -- "Good morning".\n'
)


def build(
    player: entity_agent_with_logging.EntityAgentWithLogging,
    phone: apps.Phone,
    clock: game_clock.MultiIntervalClock,
    model: language_model.LanguageModel,
    memory_factory: blank_memories.MemoryFactory,
) -> game_master_lib.GameMaster:
    """Build a GameMaster that simulates a player's interaction with their phone.

    Args:
      player: The player who is interacting with the phone.
      phone: The player's phone.
      clock: A clock.
      model: A language model.
      memory_factory: A memory factory for creating the GM's memory.

    Returns
    -------
    concordia.environment.game_master.GameMaster
      A GameMaster that simulates a player's interaction with their phone
    """
    memory = memory_factory.make_blank_memory()
    phone_component = _PhoneComponent(model, player, phone)
    cfg = ConfigStore.get_config()
    phone_spec = agent.ActionSpec(
        call_to_action=cfg.soc_sys.call_to_action, output_type=OutputType.FREE, tag="phone"
    )
    return game_master_lib.GameMaster(
        model=model,
        memory=memory,
        clock=clock,
        name="PhoneGameMaster",
        players=(player,),
        components=(phone_component,),
        action_spec=phone_spec,
        update_thought_chain=(thought_chains.identity,),
        player_observes_event=False,
    )


class _PhoneComponent(component.Component):
    """Parses the player's actions and invokes them on phone apps."""

    def __init__(  # noqa: PLR0913
        self,
        model: language_model.LanguageModel,
        player: entity_agent_with_logging.EntityAgentWithLogging,
        phone: apps.Phone,
        log_color: Literal[
            "black",
            "grey",
            "red",
            "green",
            "yellow",
            "blue",
            "magenta",
            "cyan",
            "light_grey",
            "dark_grey",
            "light_red",
            "light_green",
            "light_yellow",
            "light_blue",
            "light_magenta",
            "light_cyan",
            "white",
        ]
        | None = "yellow",
        verbose: bool = True,
        semi_verbose: bool = False,
    ):
        self._model = model
        self._player = player
        self._phone = phone
        self._logger = logging.Logger(log_color, verbose, semi_verbose)
        self._state: list = []

    def name(self) -> str:
        return "PhoneComponent"

    def _print(
        self,
        entry: str,
        emoji: str = "",
        color: COLOR_TYPE = None,
    ) -> None:
        formatted_entry = f"{emoji} {entry}" if emoji else entry
        print(termcolor.colored(formatted_entry, color or self._log_color))

    def terminate_episode(self) -> bool:
        chain_of_thought = interactive_document.InteractiveDocument(self._model)
        chain_of_thought.statement(
            f"Assess {self._player.name}'s interactions with phone so far:\n"
            + "\n- ".join(self._state)
        )
        did_conclude = chain_of_thought.yes_no_question(
            "Is any of the following statements likely true:"
            "\n - Performing this set of actions would fill up 30 minutes of the user's time."
            "\n - By performing this set of actions, the user would achieve their goal for this period."
        )
        return did_conclude

    def update_after_event(self, event_statement: str):
        # print(f"Player state:\n{self._player.state()}")
        # TODO: May want to add player state to the transcript

        print("Inside phone_update_after_event")
        print("event statement: " + event_statement)
        # print("Self state:" + "\n- ".join(self._state))
        assert isinstance(self._phone.apps[0], apps.MastodonSocialNetworkApp)
        app = self._phone.apps[0]

        if self._state == []:
            self._state.append(f"- {self._player.name} retrieved their timeline")
            p_username = app.public_get_username(self._player.name.split()[0])
            timeline = mastodon_ops.get_own_timeline(p_username, limit=10)

            def _clean_html(html_string):
                clean_text = re.sub("<[^<]+?>", "", unescape(html_string))
                return re.sub(r"\s+", " ", clean_text).strip()

            output_now = ""
            for post in timeline:
                media_desc = ""
                if post["media_attachments"]:
                    # media_lm = gpt_model.GptLanguageModel(model="gpt-4o-mini")
                    media_contents = []
                    for attachment in post["media_attachments"]:
                        media_contents.append(attachment["url"])
                    toot_headline = _clean_html(post["content"])
                    call_to_speech = DEFAULT_CALL_TO_SPEECH.format(
                        name=self._player.name,
                    )
                    call_to_action = (
                        f"{media_contents!s} Context: Sussinctly describe this image in the form of an impression that it made on {self._player.name.split()[0]} when they viewed it alongside the following text of the toot they just read on the Mastodon app: "
                        + "'"
                        + toot_headline
                        + "'"
                    )
                    media_desc = self._player.act(
                        action_spec=agent.ActionSpec(
                            call_to_action=call_to_action,
                            output_type=OutputType.FREE,
                            tag="media",
                        )
                    )
                    # media_desc = media_lm.sample_text(prompt = call_to_action)
                    media_desc = (
                        media_desc.strip(self._player.name.split()[0])
                        .strip()
                        .strip(self._player.name.split()[1])
                        .strip()
                        .strip("--")
                        .strip()
                        .strip('"')
                    )
                    media_desc = "Impression of attached image: \n" + media_desc
                    print(media_desc)
                output_now += f"User: {post['account']['display_name']} (@{post['account']['username']})\nContent: {_clean_html(post['content'])}\n{media_desc}\nToot ID: {post['id']}"

            self._player.observe(f"[Action done on phone]: Retrieved timeline: \n{output_now}")
            return [f"[Action done on phone]: Retrieved timeline: \n{output_now}"]

        chain_of_thought = interactive_document.InteractiveDocument(self._model)
        chain_of_thought.statement(event_statement)
        # check_post = chain_of_thought.yes_no_question(
        #     "Does the action in the above transcript involve the user posting a toot, replying to a toot, or boosting a toot?"
        # )q
        # print(check_post)
        # if check_post:
        check_dup = chain_of_thought.yes_no_question(
            f"Would {self._player.name} see this action as essentially acheiving the same thing as an action in the following list describing previously taken actions? (Answer No if the list is empty.): \nPrevious actions:\n"
            + "\n- ".join(self._state)
        )
        # print(check_dup)
        if check_dup:
            self._player.observe(
                f"[Action done on phone]: The following phone action was not conducted because it has already been taken - {event_statement}"
            )
            return [
                f"[Action done on phone]: The following phone action was not conducted because it has already been taken - {event_statement}"
            ]

        self._state.append(event_statement.strip())
        action_names = [a.name for a in app.actions()]
        chain_of_thought.statement(app.description())
        action_index = chain_of_thought.multiple_choice_question(
            " ".join(
                [
                    "In the above transcript, what action did the user perform?",
                    "Pick the one that is the most specific and the given information is sufficient to perform it.",
                    "If the transcript mentions multiple actions, pick one that contribute content, like making a post or reply.",
                    "Remember that the get_own_timeline shows all posts from people the user follows and should be chosen when the user mentions viewing their timeline.",
                    "Example: If the user mentions checking out other artists, but doesn't mention who, do not conduct that action.",
                ]
            ),
            answers=action_names,
        )
        # print(action_index)
        toot_id_required = chain_of_thought.yes_no_question(
            "In the above transcript, does the user's action require a numeric toot id? (numeric toot id is required for replying, liking, boosting etc.)"
        )
        # print(toot_id_required)
        if toot_id_required:
            # find most recent statement
            print("Processing Toot IDx")
            p_username = app.public_get_username(self._player.name.split()[0])
            print(p_username)
            timeline = mastodon_ops.get_own_timeline(p_username, limit=10)
            print("Got ID from Mastodon")

            def _clean_html(html_string):
                clean_text = re.sub("<[^<]+?>", "", unescape(html_string))
                return re.sub(r"\s+", " ", clean_text).strip()

            output = []
            for post in timeline:
                output.append(
                    f"\nUser: {post['account']['display_name']} (@{post['account']['username']})\nContent: {_clean_html(post['content'])}\nToot ID: {post['id']}"
                )
            if len(output) > 0:
                toot_index = chain_of_thought.multiple_choice_question(
                    "In the above transcript, which of the following toots is the user liking or responding to? Identify this using the name (if mentioned) and the context of the action and the toots from the timeline below",
                    answers=output,
                )
                responding_id = output[toot_index].split("Toot ID: ")[1]
                chain_of_thought.statement(
                    f"The exact Toot ID called for in the above action is: {responding_id}\n"
                )
                print(responding_id)
        print("Continuing action!")
        action = app.actions()[action_index]

        # # pull out suggested action from action_suggester logging channge to store in output
        app.action_logger.dummy = self._player._log["ActionSuggester"]["Selected action"]

        try:
            argument_text = chain_of_thought.open_question(
                action.instructions(),
                terminators=[],
                # Roughly, Mastodon has 500 char limit -> 125 tokens + 100 tokens for other params
                max_tokens=500 // 4 + 100,
                # print_prompt=True,  # TODO: add this arg to the open_question method
            )
            self._print(
                f"Attempting to invoke action '{action.name}'"
                f" with the argument_text:\n{argument_text}",
                color="yellow",
            )

            result = app.invoke_action(action, argument_text)

            # TODO: verify if this makes sense
            if isinstance(result, str):
                try:
                    if "Please conduct a different action" in result:
                        self._player.observe(
                            f"[Action done on phone] : Duplicate Action Attempted!! {result}"
                        )
                        self._print("Duplicate phone action result observed.", color="yellow")
                    else:
                        self._player.observe(f"[Action done on phone] : {result}")
                        self._print("Phone action result observed.", color="yellow")
                except Exception as e:
                    self._print(f"Error while observing result: {e}", color="red")

            return [result]
        except apps.ActionArgumentError:
            return []
        except Exception as e:
            self._print("Error while invoking action: " + str(e), color="red")
            return []
