import os
import pprint
import textwrap
from pathlib import Path
from typing import List, Tuple

from transformers import AutoTokenizer


class ChatProcessor:
    """
    Build a chat prompt with the model's *own* Hugging‑Face chat template.

    Parameters
    ----------
    tokenizer_version : str | None
        Name of a sub‑folder inside ./tokenizer_config holding the HF
        `tokenizer_config.json`, `special_tokens_map.json`, merges, vocab, …
        Pass `None` if you *don't* want any processing.
    """

    def __init__(self, tokenizer_version: str | None):
        self.tokenizer_version = tokenizer_version
        self.loaded_prompts: dict[str, str] = {}

        cfg_dir = (
            Path(__file__).resolve().parent / "tokenizer_config" / tokenizer_version
        )
        if not cfg_dir.exists():
            raise FileNotFoundError(
                f"Tokenizer files for “{tokenizer_version}” "
                f"not found in {cfg_dir}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg_dir, trust_remote_code=True
        )

        # Sanity check: template must exist
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                f"Tokenizer for {tokenizer_version} has no chat_template; "
                "copy the full HF tokenizer directory."
            )

    # ------------------------------------------------------------------ helpers

    def _get_cached_prompt(self, filename: str) -> str:
        """Cache small .txt files referenced from system messages."""
        if filename not in self.loaded_prompts:
            here = Path(__file__).resolve().parent
            with open(here / "samples" / filename, "r", encoding="utf‑8") as f:
                self.loaded_prompts[filename] = f.read()
        return self.loaded_prompts[filename]

    # ---------------------------------------------------------------- public API

    def process_chat(
        self, chat: List[Tuple[str, str]]
    ) -> Tuple[str, str | None]:
        """
        Convert a list like [('system', …), ('user', …), ('assistant', …)]
        into `(prompt, assistant_reply)`.

        * If self.tokenizer_version is None → returns chat unchanged.
        * If there is no assistant entry yet, `assistant_reply` will be "".
        """

        messages = []
        assistant_reply = ""

        for role, content in chat:
            if role == "assistant":
                assistant_reply = content
                continue

            if role == "system" and content.endswith(".txt") and not any(
                c in content for c in r'\/:*?"<>|'
            ):
                content = self._get_cached_prompt(content)

            messages.append({"role": role, "content": content})

        # Build the prompt using the model's *own* template.

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # leaves the assistant “stub” open
            )
        except:
            print(chat)
            # print(messages)
            raise ValueError()


        return prompt, assistant_reply
    

# ===============================================
# Simple helper for setting up SFT data
# ===============================================
class LlamaFactoryChatProcessor:
    """
    Simple wrapper to cache system prompts for use in SFT.
    """

    def __init__(self):
        self.loaded_prompts: dict[str, str] = {}

    def _get_cached_prompt(self, filename: str) -> str:
        if filename not in self.loaded_prompts:
            here = Path(__file__).resolve().parent
            with open(here / "samples" / filename, "r", encoding="utf-8") as f:
                self.loaded_prompts[filename] = f.read()
        return self.loaded_prompts[filename]

    def process_chat(self, chat: list[tuple[str, str]]) -> tuple[str, str, str]:
        sys = usr = ast = None
        for role, res in chat:
            if role == "system":
                if res.endswith(".txt") and not any(c in res for c in r'\/:*?"<>|'):
                    sys = self._get_cached_prompt(res)
                else:
                    sys = res
            elif role == "user":
                usr = res
            elif role == "assistant":
                ast = res
            else:
                raise ValueError(f"Undefined role encountered: {role}")
        
        if sys is None or usr is None or ast is None:
            print(f"FAILURE:\nSys:\n{sys}\n\nUsr:\n{usr}\n\nAst:\n{ast}")
            raise ValueError(f"System / User / Assistant data is none / non-existant.")
        
        return sys, usr, ast
    

# ===============================================
# Simple tokenizer wrapper to return the number of tokens 
# in a piece of text
# ===============================================
class TokenizerCounter:
    """
    Load the HF tokenizer from ./tokenizer_config/<tokenizer_version> and
    return exact token counts for raw text.

    Usage:
        tc = TokenizerCounter("qwen25")
        n = tc.count("some assistant text")                 # no special tokens
        n_with_specials = tc.count("text", add_special_tokens=True)
    """
    def __init__(self, tokenizer_version: str):
        cfg_dir = Path(__file__).resolve().parent / "tokenizer_config" / tokenizer_version
        if not cfg_dir.exists():
            raise FileNotFoundError(
                f"Tokenizer files for “{tokenizer_version}” not found in {cfg_dir}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg_dir, trust_remote_code=True)

    def count(self, text: str | None, add_special_tokens: bool = False) -> int:
        if not text:
            return 0
        # Use encode for a fast length; identical to len(tokenizer(text).input_ids)
        return len(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))