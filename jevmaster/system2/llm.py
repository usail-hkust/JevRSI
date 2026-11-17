"""LLM-backed System-2: judge the uncertain steps, then develop one component."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from jevmaster.prompts.system2 import (
    build_develop_prompt, build_judge_prompt, build_world_prompt, fix_prompt,
    judge_kinds, parse_judge, parse_skill_markdown,
)
from jevmaster.skills.skill import Skill
from jevmaster.system1.llm_s1 import (LLM_S1_SYSTEM_PROMPT,
                                     build_llm_s1_user_message,
                                     parse_llm_s1_reply)
from jevmaster.system2.base import SkillLookup, System2
from jevmaster.system2.clients.openai_client import OpenAIChatClient
from jevmaster.types import Observation, ReviewJob, ReviewResult

logger = logging.getLogger(__name__)


def run_review(chat, job: ReviewJob, shown_blocks: Dict[str, str],
               existing_blocks: Dict[str, str], lookup: SkillLookup,
               blocked_blocks: Optional[Dict[str, str]] = None,
               check: Optional[Callable[[Skill], Optional[str]]] = None,
               max_fixes: int = 1, develop_chat=None,
               features: Optional[dict] = None) -> ReviewResult:
    """Two turns of one chat: judge -> develop (only when the judge says so).

    An admission failure is sent back up to ``max_fixes`` times; each fix turn
    continues the same chat, so System-2 sees its earlier version and the
    problem with it.  A world-model job (``job.kind_hint``) has no judge turn:
    the framework already knows the model must be written or revised.
    """
    result = ReviewResult()
    if job.kind_hint == "world_model":
        found = lookup(job.target) if job.target else None
        decision = "REVISE" if found is not None else "NEW"
        result.decision, result.kind = decision, "world_model"
        result.target = job.target if found is not None else None
        result.reason = (f"{len(job.misses)} wrong prediction(s)" if found is not None
                         else "no world model for this task")
        target_block = found.as_review_block() if isinstance(found, Skill) else None
        system, prompt = build_world_prompt(job, target_block, features=features)
        messages = [{"role": "system", "content": system}]
        reply = None
        kind = "world_model"
    else:
        system, user = build_judge_prompt(job, shown_blocks, existing_blocks, blocked_blocks,
                                          features=features)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        reply = chat(messages)
        result.chats.append({"turn": "judge", "messages": list(messages), "reply": reply})
        result.prompt_tokens += int(getattr(reply, "prompt_tokens", 0) or 0)
        result.completion_tokens += int(getattr(reply, "completion_tokens", 0) or 0)

        decision, target, kind, reason = parse_judge(getattr(reply, "text", "") or "")
        found = lookup(target) if decision == "REVISE" and target else None
        if decision == "REVISE" and found is None:
            decision, target = "NEW", None          # stale name: write it as new
        if found is not None:
            kind = found.kind if isinstance(found, Skill) else kind
        if decision != "NONE" and kind not in judge_kinds(features):
            # A kind the judge cannot offer (e.g. a world model): nothing to develop here.
            decision, target, reason = "NONE", None, f"{reason} [{kind} is not offered]"
        result.decision, result.target, result.kind, result.reason = decision, target, kind, reason
        if decision == "NONE":
            return result

        target_block = None
        if found is not None:
            target_block = (found.as_review_block(
                show_gain=bool((features or {}).get("lifecycle")))
                if isinstance(found, Skill) else str(found))
        prompt = build_develop_prompt(kind, job, target_block, features=features)
    develop_chat = develop_chat or chat
    parse_retried = False
    item: Optional[Skill] = None
    for turn in ["develop"] + ["fix"] * max(0, int(max_fixes)):
        if reply is not None:
            messages.append({"role": "assistant", "content": getattr(reply, "text", "") or ""})
        messages.append({"role": "user", "content": prompt})
        reply = develop_chat(messages)
        logged = list(messages) if len(messages) <= 2 else messages[-1:]   # keep a judge-less system prompt
        result.chats.append({"turn": turn, "messages": logged, "reply": reply})
        result.prompt_tokens += int(getattr(reply, "prompt_tokens", 0) or 0)
        result.completion_tokens += int(getattr(reply, "completion_tokens", 0) or 0)
        markdown = parse_skill_markdown(getattr(reply, "text", "") or "")
        if markdown is None:
            if parse_retried or not (getattr(reply, "text", "") or "").strip():
                break
            parse_retried = True
            prompt = ("Reply with the complete component in the requested markdown "
                      "format (frontmatter and body); do not add another decision line.")
            continue
        result.skill_markdown = markdown
        try:
            item = Skill.from_markdown(markdown)
        except ValueError as exc:
            prompt = fix_prompt(f"invalid markdown: {exc}")
            continue
        item.kind = kind
        problem = check(item) if check is not None else None
        if problem is None:
            break
        prompt = fix_prompt(problem)
    return result


class LLMSystem2(System2):
    backend = "llm"

    def __init__(self, client: OpenAIChatClient,
                 develop_client: Optional[OpenAIChatClient] = None):
        self.client = client
        self.develop_client = develop_client

    def review(self, job: ReviewJob, shown_blocks: Dict[str, str],
               existing_blocks: Dict[str, str], lookup: SkillLookup,
               blocked_blocks: Optional[Dict[str, str]] = None,
               check=None, max_fixes: int = 1, features: Optional[dict] = None) -> ReviewResult:
        develop = self.develop_client.chat_messages if self.develop_client else None
        return run_review(self.client.chat_messages, job, shown_blocks, existing_blocks,
                          lookup, blocked_blocks=blocked_blocks, check=check,
                          max_fixes=max_fixes, develop_chat=develop, features=features)

