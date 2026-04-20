from __future__ import annotations

import shutil
import subprocess

from scenario_a.model_adapters.base import BaseModelAdapter


class ExaoneAdapter(BaseModelAdapter):
    def generate(
        self,
        *,
        system_instruction: str,
        question: str,
        context_text: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        if self.config.runtime != "ollama":
            return super().generate(
                system_instruction=system_instruction,
                question=question,
                context_text=context_text,
                history=history,
            )

        if not self.config.ollama_model_name:
            raise RuntimeError("EXAONE ollama runtime requires ollama_model_name.")
        if shutil.which("ollama") is None:
            raise RuntimeError("ollama executable is not available in PATH.")

        prompt = self.build_plain_prompt(
            system_instruction=system_instruction,
            question=question,
            context_text=context_text,
            history=history or [],
        )
        completed = subprocess.run(
            ["ollama", "run", self.config.ollama_model_name],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "ollama run failed")
        return completed.stdout.strip()
