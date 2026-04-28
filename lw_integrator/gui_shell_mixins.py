"""Preferences and shell integration helpers for the main GUI."""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox


class IntegratorGUIShellMixin:
    """Own GUI preferences, close handling, and keyboard input setup."""

    def _load_preferences(self) -> None:
        """Load saved directory preferences or use defaults."""
        self._default_config_dir = "configs/run_configs"
        self._default_output_dir = "results/runs"
        self._default_sweep_config_dir = "configs/sweep_configs"
        self._default_sweep_output_dir = "results/sweeps"

        if self._prefs_file.exists():
            try:
                with open(self._prefs_file, "r") as f:
                    prefs = json.load(f)
                self._last_config_dir = prefs.get(
                    "last_config_dir", self._default_config_dir
                )
                self._last_output_dir = prefs.get(
                    "last_output_dir", self._default_output_dir
                )
                self._last_sweep_config_dir = prefs.get(
                    "last_sweep_config_dir", self._default_sweep_config_dir
                )
                self._last_sweep_output_dir = prefs.get(
                    "last_sweep_output_dir", self._default_sweep_output_dir
                )
            except Exception:
                self._last_config_dir = self._default_config_dir
                self._last_output_dir = self._default_output_dir
                self._last_sweep_config_dir = self._default_sweep_config_dir
                self._last_sweep_output_dir = self._default_sweep_output_dir
        else:
            self._last_config_dir = self._default_config_dir
            self._last_output_dir = self._default_output_dir
            self._last_sweep_config_dir = self._default_sweep_config_dir
            self._last_sweep_output_dir = self._default_sweep_output_dir

    def _save_preferences(self) -> None:
        """Save current directory preferences."""
        try:
            prefs = {
                "last_config_dir": self._last_config_dir,
                "last_output_dir": self._last_output_dir,
                "last_sweep_config_dir": self._last_sweep_config_dir,
                "last_sweep_output_dir": self._last_sweep_output_dir,
            }
            with open(self._prefs_file, "w") as f:
                json.dump(prefs, f, indent=2)
        except Exception:
            pass

    def _reset_directories_to_defaults(self) -> None:
        """Reset directories to default values."""
        self.config_dir_var.set(self._default_config_dir)
        self.output_dir_var.set(self._default_output_dir)
        self._last_config_dir = self._default_config_dir
        self._last_output_dir = self._default_output_dir
        self._last_sweep_config_dir = self._default_sweep_config_dir
        self._last_sweep_output_dir = self._default_sweep_output_dir
        self._save_preferences()
        self._refresh_config_list()

        if hasattr(self, "optimization_tab"):
            self.optimization_tab.sweep_config_dir = self._default_sweep_config_dir
            self.optimization_tab.sweep_output_dir = self._default_sweep_output_dir

        messagebox.showinfo(
            "Reset Directories",
            "Directories reset to defaults:\n\n"
            f"Run Config: {self._default_config_dir}\n"
            f"Run Output: {self._default_output_dir}\n"
            f"Sweep Config: {self._default_sweep_config_dir}\n"
            f"Sweep Output: {self._default_sweep_output_dir}",
        )

    def _on_close(self) -> None:
        """Handle window close event."""
        self._save_preferences()
        self.root.destroy()

    def _setup_keyboard_fix(self) -> None:
        """Set up keyboard fix for non-US layouts (Swedish, German, etc.)."""
        swedish_keymap = {
            10: ("1", "!"),
            11: ("2", '"'),
            12: ("3", "#"),
            13: ("4", "¤"),
            14: ("5", "%"),
            15: ("6", "&"),
            16: ("7", "/"),
            17: ("8", "("),
            18: ("9", ")"),
            19: ("0", "="),
            20: ("+", "?"),
            21: ("´", "`"),
            24: ("q", "Q"),
            25: ("w", "W"),
            26: ("e", "E"),
            27: ("r", "R"),
            28: ("t", "T"),
            29: ("y", "Y"),
            30: ("u", "U"),
            31: ("i", "I"),
            32: ("o", "O"),
            33: ("p", "P"),
            34: ("å", "Å"),
            35: ("¨", "^"),
            38: ("a", "A"),
            39: ("s", "S"),
            40: ("d", "D"),
            41: ("f", "F"),
            42: ("g", "G"),
            43: ("h", "H"),
            44: ("j", "J"),
            45: ("k", "K"),
            46: ("l", "L"),
            47: ("ö", "Ö"),
            48: ("ä", "Ä"),
            49: ("'", "*"),
            52: ("z", "Z"),
            53: ("x", "X"),
            54: ("c", "C"),
            55: ("v", "V"),
            56: ("b", "B"),
            57: ("n", "N"),
            58: ("m", "M"),
            59: (",", ";"),
            60: (".", ":"),
            61: ("-", "_"),
            65: (" ", " "),
        }

        def fixed_key_handler(event):
            widget = event.widget
            char = event.char
            keysym = event.keysym
            keycode = event.keycode
            state = event.state

            if self._keyboard_debug:
                widget_name = widget.winfo_name()
                print(f"[KEY] Widget: {widget_name}")
                print(f"      keysym:  {keysym}")
                print(f"      keycode: {keycode}")
                print(f"      char:    {repr(char)} (from OS)")
                print(f"      state:   {state}")

            has_ctrl = bool(state & 0x4)
            has_alt = bool(state & 0x8 or state & 0x20000)
            has_shift = bool(state & 0x1)

            if has_ctrl or has_alt:
                if self._keyboard_debug:
                    print("      → Passing through (has Ctrl/Alt modifier)")
                    print("-" * 60)
                return None

            correct_char = None
            if keycode in swedish_keymap:
                unshifted, shifted = swedish_keymap[keycode]
                correct_char = shifted if has_shift else unshifted

                if self._keyboard_debug:
                    print(
                        f"      ✓ Swedish keymap: keycode {keycode} → {repr(correct_char)}"
                    )
                    if correct_char != char:
                        print(
                            f"      ⚠ FIXED: OS gave {repr(char)}, using {repr(correct_char)}"
                        )
            else:
                if not char or not char.isprintable():
                    if self._keyboard_debug:
                        print("      → Passing through (control/special key)")
                        print("-" * 60)
                    return None

                correct_char = char
                if self._keyboard_debug:
                    print(f"      ℹ Not in Swedish keymap, using OS char: {repr(char)}")

            if self._keyboard_debug:
                print(f"      ✓ Inserting: {repr(correct_char)}")
                print("-" * 60)

            if isinstance(widget, tk.Entry):
                try:
                    if widget.selection_present():
                        widget.delete("sel.first", "sel.last")
                except tk.TclError:
                    pass

                insert_pos = widget.index("insert")
                widget.insert(insert_pos, correct_char)
                return "break"

            elif isinstance(widget, tk.Text):
                try:
                    if widget.tag_ranges("sel"):
                        widget.delete("sel.first", "sel.last")
                except tk.TclError:
                    pass

                widget.insert("insert", correct_char)
                return "break"

            return None

        def bind_fix_recursive(widget):
            if isinstance(widget, (tk.Entry, tk.Text)):
                current_tags = list(widget.bindtags())
                custom_tag = f"CustomKey{id(widget)}"

                if len(current_tags) >= 2:
                    current_tags.insert(1, custom_tag)
                else:
                    current_tags.insert(0, custom_tag)

                widget.bindtags(tuple(current_tags))
                widget.bind_class(custom_tag, "<Key>", fixed_key_handler)

            for child in widget.winfo_children():
                bind_fix_recursive(child)

        bind_fix_recursive(self.root)
        if self._keyboard_debug:
            print(
                "[FIX] Swedish keyboard keycode remapping applied to all text widgets"
            )
