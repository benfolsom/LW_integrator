"""Small checkpoint launcher sharing the nonlinear pair CLI, not another solver."""

import os
from pathlib import Path
import subprocess
import sys

from .nonlinear_pair import REACTION_MODES
from .pair_history_options import HISTORY_METHODS


def build_command(
    checkpoint,
    output,
    steps,
    step_ns,
    mode="preserve",
    window="",
    derivatives="preserve",
    integration_method="preserve",
    controller="fixed",
    internal_settings="",
    history_method="auto",
    dipole_budget="",
    history_spacing="",
    dipole_relative="",
    runner="pair",
    input_kind="checkpoint",
    active_count="",
    selection_spacing="",
    constraint_budget="",
    startup_fit="preserve",
):
    """One command builder for the GUI and headless interface tests."""
    if controller not in ("fixed", "adaptive_resume"):
        raise ValueError("Unsupported time controller")
    if runner not in ("pair", "particles") or input_kind not in (
        "checkpoint",
        "initial_conditions",
    ):
        raise ValueError("Unsupported runner or input kind")
    mixed = any(
        str(v).strip() for v in (active_count, selection_spacing, constraint_budget)
    )
    if runner == "pair" and (input_kind != "checkpoint" or mixed):
        raise ValueError(
            "Particle/bunch input and mixed settings require the many-particle runner"
        )
    if controller == "adaptive_resume" and (
        runner != "pair" or startup_fit != "preserve"
    ):
        raise ValueError(
            "Adaptive recording resume is pair-only and keeps checkpoint settings"
        )
    if mixed and (not str(active_count).strip() or not str(selection_spacing).strip()):
        raise ValueError(
            "Mixed stepping requires active count and selection spacing; leave all blank on resume"
        )
    if controller == "adaptive_resume":
        if (
            mode != "preserve"
            or derivatives != "preserve"
            or integration_method != "preserve"
            or str(window).strip()
            or str(step_ns).strip()
            or str(internal_settings).strip()
            or history_method not in ("auto", "preserve")
            or str(dipole_budget).strip()
            or str(history_spacing).strip()
            or str(dipole_relative).strip()
        ):
            raise ValueError(
                "Adaptive resume keeps checkpoint settings: select preserve and leave timestep/window blank"
            )
        return [
            sys.executable,
            "-m",
            "lw_integrator.nonlinear_pair_adaptive",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--intervals",
            str(steps),
        ]
    command = [
        sys.executable,
        "-m",
        (
            "lw_integrator.nonlinear_pair"
            if runner == "pair"
            else "lw_integrator.nonlinear_particles"
        ),
        "--checkpoint" if input_kind == "checkpoint" else "--initial-conditions",
        str(checkpoint),
        "--output",
        str(output),
        "--steps",
        str(steps),
        "--step-ns",
        str(step_ns),
    ]
    if mode != "preserve":
        if mode not in REACTION_MODES:
            raise ValueError("Unsupported reaction mode")
        command += ["--radiation-reaction", mode]
    if str(window).strip():
        command += ["--reaction-window-ns", str(window)]
    if derivatives != "preserve":
        if derivatives not in ("analytic", "centered", "backward"):
            raise ValueError("Unsupported reaction derivatives")
        command += ["--reaction-derivatives", derivatives]
    if integration_method != "preserve":
        if integration_method not in ("rk4", "dop853"):
            raise ValueError("Unsupported integration method")
        command += ["--integration-method", integration_method]
    if str(internal_settings).strip():
        command += ["--internal-error-settings", str(internal_settings)]
    if history_method != "auto":
        if history_method not in ("preserve", *HISTORY_METHODS):
            raise ValueError("Unsupported source history method")
        command += ["--history-method", history_method]
    if str(dipole_budget).strip():
        command += ["--dipole-drift-budget", str(dipole_budget)]
    if str(history_spacing).strip():
        command += ["--history-sample-spacing-ns", str(history_spacing)]
    if str(dipole_relative).strip():
        command += ["--dipole-drift-relative", str(dipole_relative)]
    if mixed:
        command += [
            "--pseudogrid-rk3-active-count",
            str(active_count),
            "--pseudogrid-selection-spacing-ns",
            str(selection_spacing),
        ]
        if str(constraint_budget).strip():
            command += [
                "--pseudogrid-constraint-budget-relative",
                str(constraint_budget),
            ]
    if startup_fit != "preserve":
        if startup_fit not in ("constrained", "interpolate"):
            raise ValueError("Unsupported startup fitting policy")
        command += ["--startup-dipole-fit", startup_fit]
    return command


def launch(command, output):
    """Detach from Zed/GUI lifetime; refuse to replace existing output or log."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output checkpoint: {output}")
    log = output.with_name(output.name + ".log")
    with log.open("x") as stream:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return process, log


def open_pair_window(parent):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    window = tk.Toplevel(parent)
    window.title("Full-spin particles — pair and many-particle runner")
    window.geometry("1000x760")
    canvas = tk.Canvas(window, highlightthickness=0)
    scrollbar = ttk.Scrollbar(window, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    frame = ttk.Frame(canvas, padding=12)
    content = canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind(
        "<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.bind(
        "<Configure>", lambda event: canvas.itemconfigure(content, width=event.width)
    )
    frame.columnconfigure(1, weight=1)
    ttk.Label(
        frame,
        text=(
            "Working full-spin reaction: full_dipole_coupled.\n"
            "Fresh coupled preparation selects analytical derivatives; no fitting window is needed.\n"
            "Choose pair or many-particle mode. Many-particle mode also accepts initial-condition JSON.\n"
            "Bunch inputs represent physical particles, not weighted macroparticles.\n"
            "Adaptive resume requires an adaptive checkpoint prepared by the adaptive CLI.\n"
            "Internal error settings keep the requested source-recording timestep; leave blank to preserve them.\n"
            "Self-consistency iteration, gamma reconciliation, and chrono controls in the main window\n"
            "belong to the older solver and do not affect this runner."
        ),
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))
    values = {
        key: tk.StringVar(master=window, value=value)
        for key, value in (
            ("checkpoint", ""),
            ("output", ""),
            ("steps", "1"),
            ("step_ns", ""),
            ("mode", "preserve"),
            ("window", ""),
            ("derivatives", "preserve"),
            ("integration_method", "preserve"),
            ("controller", "fixed"),
            ("internal_settings", ""),
            ("history_method", "auto"),
            ("dipole_budget", ""),
            ("history_spacing", ""),
            ("dipole_relative", ""),
            ("runner", "pair"),
            ("input_kind", "checkpoint"),
            ("active_count", ""),
            ("selection_spacing", ""),
            ("constraint_budget", ""),
            ("startup_fit", "preserve"),
        )
    }
    labels = (
        "Input checkpoint / initial-condition JSON",
        "New output checkpoint",
        "Additional steps / adaptive intervals",
        "Lab timestep (ns; blank for adaptive resume)",
        "Reaction mode",
        "Fitting window (ns; unused for analytic)",
        "Reaction derivatives",
        "Integration method",
        "Time controller",
        "Internal error settings JSON (optional)",
        "Source history (auto for fresh; preserves resume)",
        "Absolute dipole budget (optional override; native units)",
        "History sample spacing (ns; 11 samples cover 10 spacings)",
        "Relative dipole allowance (fresh default 1e-10; or absolute)",
        "Runner (pair or many particles)",
        "Input kind",
        "Active RK4 particle count (blank keeps full solver / saved policy)",
        "RK4/RK3 selection interval (ns; required with active count)",
        "Relative mass/spin constraint budget (optional)",
        "Startup fit (fresh input only)",
    )
    for row, ((key, variable), label) in enumerate(zip(values.items(), labels), 1):
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
        widget = (
            ttk.Combobox(
                frame,
                textvariable=variable,
                values={
                    "mode": ("preserve", *REACTION_MODES),
                    "derivatives": ("preserve", "analytic", "centered", "backward"),
                    "integration_method": ("preserve", "rk4", "dop853"),
                    "controller": ("fixed", "adaptive_resume"),
                    "history_method": ("auto", "preserve", *HISTORY_METHODS),
                    "runner": ("pair", "particles"),
                    "input_kind": ("checkpoint", "initial_conditions"),
                    "startup_fit": ("preserve", "constrained", "interpolate"),
                }[key],
                state="readonly",
            )
            if key
            in (
                "mode",
                "derivatives",
                "integration_method",
                "controller",
                "history_method",
                "runner",
                "input_kind",
                "startup_fit",
            )
            else ttk.Entry(frame, textvariable=variable, width=45)
        )
        widget.grid(row=row, column=1, sticky="ew")
        if key in ("checkpoint", "output", "internal_settings"):

            def choose(key=key):
                chooser = (
                    filedialog.askopenfilename
                    if key != "output"
                    else filedialog.asksaveasfilename
                )
                path = chooser(
                    parent=window,
                    filetypes=[("JSON checkpoint", "*.json")],
                    defaultextension=".json",
                )
                if path:
                    values[key].set(path)

            ttk.Button(frame, text="Browse", command=choose).grid(
                row=row, column=2, padx=4
            )
    status = tk.StringVar(
        master=window,
        value="No job launched. Changing an evolved checkpoint's reaction settings is rejected.",
    )
    ttk.Label(frame, textvariable=status, wraplength=680).grid(
        row=len(labels) + 2, column=0, columnspan=3, sticky="w", pady=10
    )
    log_text = tk.Text(frame, height=10, width=85, state="disabled")
    log_text.grid(row=len(labels) + 3, column=0, columnspan=3, sticky="nsew")

    def start():
        try:
            source = Path(values["checkpoint"].get()).expanduser().resolve()
            target = Path(values["output"].get()).expanduser().resolve()
            if (
                not source.is_file()
                or not values["output"].get().strip()
                or source == target
            ):
                raise ValueError(
                    "Select an existing input checkpoint and a distinct new output file"
                )
            command = build_command(
                source,
                target,
                values["steps"].get(),
                values["step_ns"].get(),
                values["mode"].get(),
                values["window"].get(),
                values["derivatives"].get(),
                values["integration_method"].get(),
                values["controller"].get(),
                values["internal_settings"].get(),
                values["history_method"].get(),
                values["dipole_budget"].get(),
                values["history_spacing"].get(),
                values["dipole_relative"].get(),
                runner=values["runner"].get(),
                input_kind=values["input_kind"].get(),
                active_count=values["active_count"].get(),
                selection_spacing=values["selection_spacing"].get(),
                constraint_budget=values["constraint_budget"].get(),
                startup_fit=values["startup_fit"].get(),
            )
            process, log = launch(command, target)
        except (ValueError, OSError) as error:
            messagebox.showerror(
                "Cannot launch full-spin run", str(error), parent=window
            )
            return
        button.configure(state="disabled")
        status.set(
            f"PID {process.pid}. Saves each accepted step. Closing this window leaves the job running. Log: {log}"
        )

        def poll():
            if not window.winfo_exists():
                return
            try:
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 6000))
                    text = stream.read().decode("utf-8", errors="replace")
                log_text.configure(state="normal")
                log_text.delete("1.0", "end")
                log_text.insert("end", text)
                log_text.configure(state="disabled")
                log_text.see("end")
            except OSError:
                pass
            code = process.poll()
            if code is None:
                window.after(1000, poll)
            else:
                status.set(
                    f"Job ended with exit status {code}. Check the log and last saved checkpoint: {log}"
                )
                button.configure(state="normal")

        window.after(1000, poll)

    button = ttk.Button(frame, text="Run full-spin simulation", command=start)
    button.grid(row=len(labels) + 1, column=0, columnspan=3, pady=8)
    return window


if __name__ == "__main__":
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    dialog = open_pair_window(root)
    dialog.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
