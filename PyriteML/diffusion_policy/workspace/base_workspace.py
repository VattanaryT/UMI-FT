import copy
import pathlib
import secrets
import string
import threading
from typing import Optional

import dill
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def init_wandb_trackers(self, accelerator, cfg):
        """Start accelerate's wandb tracker without depending on wandb's
        local resume='auto' bookkeeping file.

        Passing `resume=True` straight through to `wandb.init()` (as the old
        per-workspace `run()` code did) makes wandb treat it as
        `resume="auto"`, which tries to read/write a local
        `<dir>/wandb/wandb-resume.json` file to remember the run id across
        process restarts *in the same working directory*. That assumption
        breaks in this Slurm+apptainer setup: the default wandb dir is the
        process cwd, which is the read-only container image tree, and even
        after pointing WANDB_DIR elsewhere, each job/container invocation
        doesn't share a pre-existing writable `wandb/` folder for it to
        write into -- both cases crash inside `_save_autoresume_run_id`.

        Instead we manage a stable run id ourselves, stored as a plain text
        file next to the checkpoints (a directory that is writable,
        job-specific, and persists across resumes), and always pass
        `resume="allow"` with that explicit id. `resume="allow"` never
        touches the local auto-resume file (only `resume="auto"` does): it
        just resumes the run with that id on the wandb server if it already
        exists, or starts a new one otherwise.
        """
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        project = wandb_cfg.pop("project")
        wandb_cfg.pop("log_every", None)  # train-loop knob, not a wandb.init arg
        wants_resume = bool(wandb_cfg.pop("resume", False))

        run_id_path = pathlib.Path(self.output_dir) / "wandb_run_id.txt"
        run_id = wandb_cfg.get("id")
        if run_id is None and wants_resume and run_id_path.is_file():
            run_id = run_id_path.read_text().strip() or None
        if run_id is None:
            # `wandb.util` is no longer imported by `import wandb` in newer
            # wandb releases, so don't rely on it; wandb ids are just short
            # random lowercase alphanumeric strings.
            run_id = "".join(
                secrets.choice(string.ascii_lowercase + string.digits)
                for _ in range(8)
            )

        run_id_path.parent.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(run_id)

        wandb_cfg["id"] = run_id
        wandb_cfg["resume"] = "allow"
        # Keep wandb's own local run files under the writable, persistent
        # checkpoint output dir instead of defaulting to cwd (read-only
        # container image) or requiring WANDB_DIR to already exist.
        wandb_cfg.setdefault("dir", self.output_dir)

        accelerator.init_trackers(
            project_name=project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg},
        )

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=True,
    ):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(
        self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs
    ):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, **kwargs)
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(
        cls, path, exclude_keys=None, include_keys=None, **kwargs
    ):
        payload = torch.load(open(path, "rb"), pickle_module=dill)
        instance = cls(payload["cfg"])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs,
        )
        return instance

    def save_snapshot(self, tag="latest"):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
