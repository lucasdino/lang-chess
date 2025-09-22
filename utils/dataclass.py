import os
import json
import random

from .prompts.chat_to_prompt import ChatProcessor


# ---------------------------------------------------------------------
# Single-file JSONL loader (used for single files)
# ---------------------------------------------------------------------
class JSONLDataClass:
    def __init__(self, data_dir, filename, task_map, model_version):
        """Load a single jsonl file of raw chat logs."""
        self.filename = filename
        self.trimmed_filename = os.path.splitext(filename)[0]
        self.data_dir = data_dir
        self.filepath = os.path.join(data_dir, filename)
        self.task_type = next(v for k, v in task_map.items() if filename.startswith(k))
        self.chat_processor = ChatProcessor(model_version)

        self.data = self._load_data(self.filepath)

    def _load_data(self, filepath, shuffle: bool = True):
        with open(filepath, "r") as f:
            raw_data = [json.loads(line.strip()) for line in f if line.strip()]

        data = []
        # Only “chat” format is supported here.
        for datum in raw_data:
            prompt, response = self.chat_processor.process_chat(datum["chat"])
            data.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "info": datum["info"],
                }
            )

        print(f"Loaded {filepath} with {len(data)} entries.")
        if shuffle:
            random.shuffle(data)
        return data


# ---------------------------------------------------------------------
# Folder-level loader (Used for model responses)
# ---------------------------------------------------------------------
class JSONFolderDataClass:
    """
    Load an entire *folder* of .json or .jsonl files that each contain
    {'model_response': ..., 'info': ...} items.

    A `sys_prompt` must be provided so the loader can wrap each response
    into a system/user/assistant triple before handing it to ChatProcessor.
    """

    def __init__(self, data_dir: str, folder_name: str, model_version: str, sys_prompt: str):
        self.folder_name = folder_name
        self.trimmed_foldername = folder_name.rstrip("/\\")
        self.data_dir = data_dir
        self.sys_prompt = sys_prompt
        self.folder_path = os.path.join(data_dir, folder_name)
        self.chat_processor = ChatProcessor(model_version)

        self.data = self._load_folder(self.folder_path, sys_prompt=sys_prompt)

    # -----------------------------------------------------------------
    def _load_folder(self, folder_path, sys_prompt: str, shuffle: bool = True):
        filepaths = sorted(
            [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".json") or f.endswith(".jsonl")
            ]
        )
        all_data = []
        for fp in filepaths:
            with open(fp, "r") as f:
                try:
                    if fp.endswith(".jsonl"):
                        raw_data = [json.loads(line.strip()) for line in f if line.strip()]
                    else:
                        raw_data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse {fp}: {e}")
                    raise

            for datum in raw_data:
                user_prompt = (
                    f"Below is the provided model response:\n\n{datum['model_response']}"
                )
                chat = [
                    ["system", sys_prompt],
                    ["user", user_prompt],
                    ["assistant", ""],
                ]
                prompt, response = self.chat_processor.process_chat(chat)
                all_data.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "info": datum["info"],
                    }
                )

        print(
            f"Loaded {len(filepaths)} files from {folder_path} → {len(all_data)} processed items."
        )
        if shuffle:
            random.shuffle(all_data)
        return all_data