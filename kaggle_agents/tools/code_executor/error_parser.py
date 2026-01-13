"""
Error parsing and filtering for code execution.

Contains methods for parsing and categorizing errors from execution output.
"""

from __future__ import annotations

import re


class ErrorParserMixin:
    """Mixin providing error parsing and filtering methods."""

    def _filter_optuna_logs(self, output: str) -> str:
        """
        Filter out Optuna informational logs that are not errors.

        Optuna logs like "[I 2025-11-24 ...] Trial 0 finished with value: ..."
        are informational and should not be treated as errors.

        Args:
            output: stderr or stdout content

        Returns:
            Filtered output with Optuna info logs removed
        """
        # Pattern for Optuna info logs: [I YYYY-MM-DD HH:MM:SS,...]
        optuna_info_pattern = r"\[I \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+\][^\n]*\n?"
        filtered = re.sub(optuna_info_pattern, "", output)

        # Also filter Optuna study creation messages
        study_pattern = r"A new study created in memory with name:[^\n]*\n?"
        filtered = re.sub(study_pattern, "", filtered)

        # Filter Optuna trial completion messages
        trial_pattern = r"Trial \d+ finished with value:[^\n]*\n?"
        filtered = re.sub(trial_pattern, "", filtered)

        # Filter Optuna sampler messages
        sampler_pattern = r"\[I[^\]]*\].*?(?:Sampler|TPE|CMA|Grid)[^\n]*\n?"
        filtered = re.sub(sampler_pattern, "", filtered)

        return filtered.strip()

    def _filter_tqdm_logs(self, output: str) -> str:
        """
        Filter out tqdm progress bar output that is commonly written to stderr.

        Many ML scripts use tqdm, which writes progress updates to stderr by default.
        Those updates are not errors and should not cause the execution to be marked
        as failed.
        """
        if not output:
            return ""

        kept_lines: list[str] = []
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Keep anything that looks like an actual exception/error.
            if any(tok in stripped for tok in ("Traceback", "Error", "Exception")):
                kept_lines.append(line)
                continue

            # Drop common tqdm progress patterns, e.g.:
            # "Fold0 Train Epoch1:  12%|█▏        | 33/275 [00:41<04:49,  1.20s/it, loss=1.62]"
            # "Validation:   0%|          | 0/138 [00:00<?, ?it/s]"
            has_bar = "%|" in stripped and "|" in stripped
            has_rate = ("it/s" in stripped) or ("s/it" in stripped)
            has_speed = bool(re.search(r"\b\d+(\.\d+)?\s*[kMGT]?B/s\b", stripped, re.IGNORECASE))
            has_counts = bool(re.search(r"\b\d+/\d+\b", stripped))
            if has_bar and (has_rate or has_speed or has_counts):
                continue
            if (has_rate or has_speed) and has_counts:
                continue

            kept_lines.append(line)

        return "\n".join(kept_lines).strip()

    def _filter_framework_logs(self, output: str) -> str:
        """
        Filter out common non-fatal ML framework stderr noise.

        These messages are typically warnings or informational logs that
        shouldn't cause execution failure (e.g., cuFFT factory registration).
        """
        if not output:
            return ""

        drop_patterns = [
            # CUDA factory registration warnings
            r"Unable to register cuFFT factory",
            r"Unable to register cuDNN factory",
            r"Unable to register cuBLAS factory",
            # TensorFlow INFO/WARNING logs (written to stderr by default)
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+: [IWE] tensorflow/",
            r"^[IWE]\d+ .*tensorflow/",
            r"^[IWE]\d+ .*cuda_dnn\.cc",
            r"^[IWE]\d+ .*cuda_blas\.cc",
            r"^[IWE]\d+ .*cuda_fft\.cc",
            r"^[IWE]\d+ .*device_compiler",
            r"^[IWE]\d+ .*service\.cc",
            r"^[IWE]\d+ .*gpu_device\.cc",
            r"This TensorFlow binary is optimized",
            r"oneDNN custom operations are on",
            r"computation placer already registered",
            r"Registers are spilled to local memory",
            r"TF_FORCE_GPU_ALLOW_GROWTH",
            r"XLA service.*initialized for platform",
            r"StreamExecutor device",
            r"Compiled cluster using XLA",
            r"ptxas warning",
            r"disabling MLIR crash reproducer",
            r"Loaded cuDNN version",
            r"Created device /job:localhost",
        ]

        kept_lines: list[str] = []
        for line in output.splitlines():
            if any(re.search(pattern, line) for pattern in drop_patterns):
                continue
            kept_lines.append(line)

        return "\n".join(kept_lines).strip()

    def _parse_errors(self, stderr: str, stdout: str) -> list[str]:
        """
        Parse and categorize errors from output.

        Args:
            stderr: Standard error output
            stdout: Standard output

        Returns:
            List of error messages
        """
        errors = []

        # Filter out non-error stderr noise before parsing.
        # Optuna and tqdm commonly log to stderr even on successful runs.
        stderr_filtered = self._filter_optuna_logs(stderr) if stderr else ""
        stderr_filtered = self._filter_tqdm_logs(stderr_filtered)
        stderr_filtered = self._filter_framework_logs(stderr_filtered)
        if stderr_filtered:
            stderr_lines = []
            skip_next_context = False
            for line in stderr_filtered.splitlines():
                clean_line = line
                if clean_line.startswith("⚠️"):
                    clean_line = clean_line[len("⚠️") :]

                if skip_next_context:
                    if clean_line.strip() and clean_line[:1].isspace():
                        skip_next_context = False
                        continue
                    skip_next_context = False

                if "Warning" in clean_line or "WARNING" in clean_line:
                    skip_next_context = True
                    continue

                stderr_lines.append(line)
            stderr_filtered = "\n".join(stderr_lines).strip()

        # Check stderr for Python exceptions
        if stderr_filtered:
            # Extract traceback info
            if "Traceback" in stderr_filtered:
                lines = stderr_filtered.split("\n")
                error_line = ""
                for line in reversed(lines):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if re.match(r"^[A-Za-z_]+(Error|Exception):", stripped):
                        error_line = stripped
                        break
                if not error_line:
                    for line in reversed(lines):
                        stripped = line.strip()
                        if stripped:
                            error_line = stripped
                            break
                if error_line:
                    errors.append(error_line)

            # Common error patterns
            error_patterns = [
                (r"ModuleNotFoundError: No module named '(\w+)'", "Missing module: {}"),
                (r"FileNotFoundError: .* '(.*)'", "File not found: {}"),
                (r"KeyError: '(.*)'", "Missing key: {}"),
                (r"ValueError: (.*)", "Value error: {}"),
                (r"TypeError: (.*)", "Type error: {}"),
                (r"MemoryError", "Out of memory"),
            ]

            for pattern, template in error_patterns:
                match = re.search(pattern, stderr_filtered)
                if match:
                    if "{}" in template:
                        errors.append(template.format(match.group(1)))
                    else:
                        errors.append(template)

            # If no specific error found, add generic stderr (but not if only Optuna logs)
            if not errors and stderr_filtered.strip():
                # Double-check it's not just leftover Optuna formatting
                if not re.match(r"^\s*\[.*?\]\s*$", stderr_filtered.strip()):
                    error_hint = re.search(
                        r"(Traceback|Error|Exception|Segmentation fault|SIGSEGV|Killed|Out of memory|OOM|CUDA error)",
                        stderr_filtered,
                        re.IGNORECASE,
                    )
                    if error_hint:
                        errors.append(f"Error: {stderr_filtered.strip()[:200]}")

        # Do not treat warnings as errors.

        return errors
