"""Layout support helpers for the main GUI."""

from __future__ import annotations

from typing import Any, Optional, Set

import tkinter as tk
from tkinter import scrolledtext, ttk

CONTENT_PANEL_MIN_WIDTH = 800  # pixels; keeps tab content usable
CONFIG_PANEL_MIN_WIDTH = 450  # pixels; keeps right-side controls readable


class _ScrollableNotebookPage:
    def __init__(self, notebook: ttk.Notebook, title: str, padding: int = 12) -> None:
        self.container = ttk.Frame(notebook)
        notebook.add(self.container, text=title)
        self.container.columnconfigure(0, weight=1)
        self.container.rowconfigure(0, weight=1)
        self._bound_widgets: Set[int] = set()

        self.canvas = tk.Canvas(self.container, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(
            self.container, orient="vertical", command=self.canvas.yview
        )
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.h_scrollbar = ttk.Scrollbar(
            self.container, orient="horizontal", command=self.canvas.xview
        )
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(
            yscrollcommand=self.scrollbar.set, xscrollcommand=self.h_scrollbar.set
        )

        self.frame = ttk.Frame(self.canvas, padding=padding)
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.frame, anchor="nw"
        )

        self.frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self._mousewheel_bound = False
        self._bind_mousewheel(self.container)
        self._bind_mousewheel(self.canvas)
        self._bind_mousewheel(self.frame)

    def _on_frame_configure(self, _event: Any) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: Any) -> None:
        canvas_width = event.width
        bbox = self.canvas.bbox("all")
        if not bbox:
            return

        content_width = bbox[2] - bbox[0]
        if canvas_width >= content_width:
            self.canvas.itemconfigure(self._window_id, width=canvas_width)
        else:
            self.canvas.itemconfigure(self._window_id, width=content_width)

    def _bind_mousewheel(self, widget: tk.Widget) -> None:
        widget_id = widget.winfo_id()
        if widget_id in self._bound_widgets:
            return
        self._bound_widgets.add(widget_id)
        widget.bind("<Enter>", lambda _event: self._activate_mousewheel(), add=True)
        widget.bind(
            "<Leave>", lambda _event: self._maybe_deactivate_mousewheel(), add=True
        )

    def refresh_mousewheel_bindings(self) -> None:
        self._register_descendants(self.frame)

    def _register_descendants(self, widget: tk.Widget) -> None:
        self._bind_mousewheel(widget)
        for child in widget.winfo_children():
            self._register_descendants(child)

    def _activate_mousewheel(self) -> None:
        if self._mousewheel_bound:
            return
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        self._mousewheel_bound = True

    def _maybe_deactivate_mousewheel(self) -> None:
        if not self._mousewheel_bound:
            return
        widget = self.container.winfo_containing(
            self.container.winfo_pointerx(), self.container.winfo_pointery()
        )
        if not self._is_descendant(widget):
            self._deactivate_mousewheel()

    def _deactivate_mousewheel(self) -> None:
        if not self._mousewheel_bound:
            return
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
        self._mousewheel_bound = False

    def _is_descendant(self, widget: Optional[tk.Widget]) -> bool:
        while widget is not None:
            if widget == self.container:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _on_mousewheel(self, event: Any) -> None:
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
        self.canvas.yview_scroll(delta, "units")


class IntegratorGUILayoutMixin:
    """Own the persistent config panel and layout utility methods."""

    def _enforce_panel_minimums(self, event=None):
        """Enforce minimum panel sizes when sash is moved."""
        if not hasattr(self, "_main_horizontal_paned"):
            return

        try:
            sash_pos = self._main_horizontal_paned.sash_coord(0)[0]
            total_width = self._main_horizontal_paned.winfo_width()
            min_left = CONTENT_PANEL_MIN_WIDTH
            max_left = total_width - CONFIG_PANEL_MIN_WIDTH
            if sash_pos < min_left:
                self._main_horizontal_paned.sash_place(0, min_left, 0)
            elif sash_pos > max_left:
                self._main_horizontal_paned.sash_place(0, max_left, 0)
        except Exception:
            pass

    def _create_scrollable_tab(
        self, notebook: ttk.Notebook, title: str, *, padding: int = 12
    ) -> ttk.Frame:
        page = _ScrollableNotebookPage(notebook, title, padding=padding)
        self._scroll_pages.append(page)
        return page.frame

    def _build_config_panel(self, parent):
        """Build persistent config/control panel on right side."""
        panel = ttk.LabelFrame(parent, text="Configuration & Control", padding=10)
        panel.pack(fill="both", expand=True, padx=5, pady=5)

        scroll_container = ttk.Frame(panel)
        scroll_container.pack(fill="both", expand=True, side="top", pady=(0, 5))

        canvas = tk.Canvas(
            scroll_container,
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            scroll_container, orient="vertical", command=canvas.yview
        )
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_resize(event):
            canvas.itemconfig(window_id, width=event.width)

        canvas.bind("<Configure>", _on_canvas_resize)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        run_config_frame = ttk.LabelFrame(
            scrollable_frame, text="Single Run Configuration", padding=4
        )
        run_config_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(run_config_frame, text="Config dir:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Entry(run_config_frame, textvariable=self.config_dir_var, width=20).grid(
            row=0, column=1, sticky="ew", pady=2, padx=(5, 2)
        )
        ttk.Button(
            run_config_frame, text="...", command=self._select_config_dir, width=3
        ).grid(row=0, column=2, sticky="w", pady=2)

        ttk.Label(run_config_frame, text="Output dir:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Entry(run_config_frame, textvariable=self.output_dir_var, width=20).grid(
            row=1, column=1, sticky="ew", pady=2, padx=(5, 2)
        )
        ttk.Button(
            run_config_frame, text="...", command=self._select_output_dir, width=3
        ).grid(row=1, column=2, sticky="w", pady=2)

        run_config_frame.columnconfigure(1, weight=1)

        ttk.Label(run_config_frame, text="Config name:").grid(
            row=2, column=0, sticky="w", pady=(10, 2)
        )
        ttk.Entry(run_config_frame, textvariable=self.config_name_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(10, 2)
        )

        ttk.Label(run_config_frame, text="Current:").grid(
            row=3, column=0, sticky="w", pady=2
        )
        self.current_config_label = ttk.Label(
            run_config_frame,
            text="<unsaved>",
            foreground="gray",
            font=("TkDefaultFont", 9, "italic"),
        )
        self.current_config_label.grid(
            row=3, column=1, columnspan=2, sticky="w", pady=2
        )

        ttk.Checkbutton(
            run_config_frame,
            text="Write resumable checkpoints",
            variable=self.checkpoint_enabled_var,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(10, 2))
        ttk.Label(run_config_frame, text="Checkpoint dir:").grid(
            row=5, column=0, sticky="w", pady=2
        )
        ttk.Entry(
            run_config_frame,
            textvariable=self.checkpoint_directory_var,
            width=20,
        ).grid(row=5, column=1, sticky="ew", pady=2, padx=(5, 2))
        ttk.Button(
            run_config_frame,
            text="...",
            command=self._select_checkpoint_directory,
            width=3,
        ).grid(row=5, column=2, sticky="w", pady=2)
        ttk.Label(run_config_frame, text="Resume from:").grid(
            row=6, column=0, sticky="w", pady=2
        )
        ttk.Entry(
            run_config_frame,
            textvariable=self.checkpoint_resume_from_var,
            width=20,
        ).grid(row=6, column=1, sticky="ew", pady=2, padx=(5, 2))
        ttk.Button(
            run_config_frame,
            text="...",
            command=self._select_checkpoint_resume_directory,
            width=3,
        ).grid(row=6, column=2, sticky="w", pady=2)
        ttk.Label(run_config_frame, text="Every steps / seconds:").grid(
            row=7, column=0, sticky="w", pady=2
        )
        checkpoint_interval_frame = ttk.Frame(run_config_frame)
        checkpoint_interval_frame.grid(
            row=7, column=1, columnspan=2, sticky="ew", pady=2, padx=(5, 0)
        )
        ttk.Entry(
            checkpoint_interval_frame,
            textvariable=self.checkpoint_interval_steps_var,
            width=8,
        ).pack(side="left")
        ttk.Entry(
            checkpoint_interval_frame,
            textvariable=self.checkpoint_interval_seconds_var,
            width=8,
        ).pack(side="left", padx=(5, 0))

        ttk.Label(run_config_frame, text="Saved configs:").grid(
            row=8, column=0, columnspan=3, sticky="w", pady=(10, 2)
        )

        run_list_frame = ttk.Frame(run_config_frame)
        run_list_frame.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=2)
        run_list_frame.rowconfigure(0, weight=1)
        run_list_frame.columnconfigure(0, weight=1)

        self.config_list = tk.Listbox(run_list_frame, height=9)
        self.config_list.grid(row=0, column=0, sticky="nsew")
        self.config_list.bind(
            "<<ListboxSelect>>", lambda _event: self._on_config_selected()
        )
        self.config_list.bind("<Double-1>", lambda _event: self._load_config())

        run_scrollbar = ttk.Scrollbar(
            run_list_frame, orient="vertical", command=self.config_list.yview
        )
        run_scrollbar.grid(row=0, column=1, sticky="ns")
        self.config_list.configure(yscrollcommand=run_scrollbar.set)

        run_config_frame.rowconfigure(9, weight=1)

        run_btn_frame = ttk.Frame(run_config_frame)
        run_btn_frame.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(5, 0))

        ttk.Button(run_btn_frame, text="Load", command=self._load_config, width=8).pack(
            side="left", padx=2
        )
        ttk.Button(run_btn_frame, text="Save", command=self._save_config, width=8).pack(
            side="left", padx=2
        )
        ttk.Button(
            run_btn_frame, text="Refresh", command=self._refresh_config_list, width=8
        ).pack(side="left", padx=2)

        sweep_config_frame = ttk.LabelFrame(
            scrollable_frame, text="Sweep Configuration", padding=4
        )
        sweep_config_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(sweep_config_frame, text="Config dir:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.sweep_config_dir_var = tk.StringVar(value=self._last_sweep_config_dir)
        ttk.Entry(
            sweep_config_frame, textvariable=self.sweep_config_dir_var, width=20
        ).grid(row=0, column=1, sticky="ew", pady=2, padx=(5, 2))
        ttk.Button(
            sweep_config_frame,
            text="...",
            command=self._select_sweep_config_dir,
            width=3,
        ).grid(row=0, column=2, sticky="w", pady=2)

        ttk.Label(sweep_config_frame, text="Output dir:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.sweep_output_dir_var = tk.StringVar(value=self._last_sweep_output_dir)
        ttk.Entry(
            sweep_config_frame, textvariable=self.sweep_output_dir_var, width=20
        ).grid(row=1, column=1, sticky="ew", pady=2, padx=(5, 2))
        ttk.Button(
            sweep_config_frame,
            text="...",
            command=self._select_sweep_output_dir,
            width=3,
        ).grid(row=1, column=2, sticky="w", pady=2)

        sweep_config_frame.columnconfigure(1, weight=1)

        ttk.Label(sweep_config_frame, text="Config name:").grid(
            row=2, column=0, sticky="w", pady=(10, 2)
        )
        ttk.Entry(sweep_config_frame, textvariable=self.sweep_config_name_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(10, 2)
        )

        ttk.Label(sweep_config_frame, text="Current:").grid(
            row=3, column=0, sticky="w", pady=(5, 2)
        )
        self.current_sweep_config_label = ttk.Label(
            sweep_config_frame,
            text="<none>",
            foreground="gray",
            font=("TkDefaultFont", 9, "italic"),
        )
        self.current_sweep_config_label.grid(
            row=3, column=1, columnspan=2, sticky="w", pady=(5, 2)
        )

        ttk.Label(sweep_config_frame, text="Saved configs:").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(10, 2)
        )

        sweep_list_frame = ttk.Frame(sweep_config_frame)
        sweep_list_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=2)
        sweep_list_frame.rowconfigure(0, weight=1)
        sweep_list_frame.columnconfigure(0, weight=1)

        self.sweep_config_list = tk.Listbox(sweep_list_frame, height=9)
        self.sweep_config_list.grid(row=0, column=0, sticky="nsew")
        self.sweep_config_list.bind(
            "<Double-1>", lambda _event: self._load_sweep_config()
        )
        self.sweep_config_list.bind(
            "<<ListboxSelect>>", lambda _event: self._on_sweep_config_selected()
        )

        sweep_scrollbar = ttk.Scrollbar(
            sweep_list_frame, orient="vertical", command=self.sweep_config_list.yview
        )
        sweep_scrollbar.grid(row=0, column=1, sticky="ns")
        self.sweep_config_list.configure(yscrollcommand=sweep_scrollbar.set)

        sweep_config_frame.rowconfigure(5, weight=1)

        sweep_btn_frame = ttk.Frame(sweep_config_frame)
        sweep_btn_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(5, 0))

        ttk.Button(
            sweep_btn_frame, text="Load", command=self._load_sweep_config, width=8
        ).pack(side="left", padx=2)
        ttk.Button(
            sweep_btn_frame, text="Save", command=self._save_sweep_config, width=8
        ).pack(side="left", padx=2)
        ttk.Button(
            sweep_btn_frame,
            text="Refresh",
            command=self._refresh_sweep_config_list,
            width=8,
        ).pack(side="left", padx=2)

        reset_frame = ttk.Frame(scrollable_frame)
        reset_frame.pack(fill="x", pady=(0, 10))
        ttk.Button(
            reset_frame,
            text="Reset All Directories to Defaults",
            command=self._reset_directories_to_defaults,
        ).pack(fill="x")

        status_frame = ttk.LabelFrame(panel, text="Status", padding=4)
        status_frame.pack(side="bottom", fill="x")

        self._refresh_sweep_config_list()
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w", pady=2)

        self._progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self._progress_bar.pack(fill="x", pady=5)

        control_frame = ttk.LabelFrame(panel, text="Controls", padding=4)
        control_frame.pack(side="bottom", fill="x", pady=(0, 5))

        self._run_button = ttk.Button(
            control_frame,
            text="▶ Run",
            command=self._trigger_run,
            style="Accent.TButton",
        )
        self._run_button.pack(fill="x", pady=2)
        self._run_button.configure(width=12)

        self._cancel_button = ttk.Button(
            control_frame,
            text="⬛ Cancel",
            command=self._trigger_cancel,
            state="disabled",
        )
        self._cancel_button.pack(fill="x", pady=2)
        self._cancel_button.configure(width=12)

        mode_frame = ttk.LabelFrame(panel, text="Run Mode", padding=4)
        mode_frame.pack(side="bottom", fill="x", pady=(0, 5))

        self.run_mode_var = tk.StringVar(value="single")

        ttk.Radiobutton(
            mode_frame,
            text="Single Run",
            variable=self.run_mode_var,
            value="single",
            command=self._on_run_mode_changed,
        ).pack(anchor="w", pady=2)

        ttk.Radiobutton(
            mode_frame,
            text="Sweep/Optim",
            variable=self.run_mode_var,
            value="sweep",
            command=self._on_run_mode_changed,
        ).pack(anchor="w", pady=2)

    def _on_run_mode_changed(self):
        """Handle run mode selection change."""
        mode = self.run_mode_var.get()
        if mode == "single":
            self._run_button.config(text="▶ Run", command=self._trigger_run)
        else:
            self._run_button.config(text="▶ Run Sweep", command=self._trigger_sweep)

    def _build_log_summary_panel(self, bottom_container: ttk.Frame) -> None:
        """Build the lower split panel for logs and initial summary."""
        lower_paned = ttk.Panedwindow(bottom_container, orient="horizontal")
        lower_paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        log_frame = ttk.LabelFrame(lower_paned, text="Logs", padding=8)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=0)
        log_frame.rowconfigure(1, weight=1)

        log_controls = ttk.Frame(log_frame)
        log_controls.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self.log_format_var = tk.StringVar(value="detailed")
        ttk.Radiobutton(
            log_controls,
            text="Summary",
            variable=self.log_format_var,
            value="summary",
            command=self._update_log_format,
        ).pack(side="left", padx=5)
        ttk.Radiobutton(
            log_controls,
            text="Detailed",
            variable=self.log_format_var,
            value="detailed",
            command=self._update_log_format,
        ).pack(side="left", padx=5)

        ttk.Button(log_controls, text="Clear", command=self._clear_log, width=8).pack(
            side="right", padx=5
        )

        self.log_output = scrolledtext.ScrolledText(
            log_frame, height=6, state="disabled", wrap="none"
        )
        self.log_output.grid(row=1, column=0, sticky="nsew")

        self._raw_log_lines = []
        self._log_summary = []

        lower_paned.add(log_frame, weight=3)

        summary_frame = ttk.LabelFrame(lower_paned, text="Initial summary", padding=8)
        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)

        summary_text_frame = ttk.Frame(summary_frame)
        summary_text_frame.grid(row=0, column=0, sticky="nsew")
        summary_text_frame.columnconfigure(0, weight=1)
        summary_text_frame.rowconfigure(0, weight=1)

        self.summary_text = tk.Text(
            summary_text_frame,
            height=3,
            width=40,
            wrap="word",
            state="disabled",
            relief="flat",
            borderwidth=0,
        )
        self.summary_text.grid(row=0, column=0, sticky="nsew")

        summary_scrollbar = ttk.Scrollbar(
            summary_text_frame, command=self.summary_text.yview
        )
        summary_scrollbar.grid(row=0, column=1, sticky="ns")
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)

        lower_paned.add(summary_frame, weight=1)
