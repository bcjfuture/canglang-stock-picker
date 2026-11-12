from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

from .theme import COLORS, FONTS


def change_color(value: object) -> str | None:
    """A-share convention: red for up, green for down; flat stays plain."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number == 0:
        return None
    return COLORS["danger"] if number > 0 else COLORS["success"]


class ColoredCells:
    """Colored text laid over single cells; ttk.Treeview only colors whole rows."""

    def __init__(self, tree: ttk.Treeview, *, on_double=None) -> None:
        self.tree = tree
        self.on_double = on_double
        self.cells: dict[tuple[str, str], tk.Label] = {}
        for sequence in ("<Configure>", "<Map>", "<B1-Motion>", "<ButtonRelease-1>", "<<TreeviewSelect>>"):
            tree.bind(sequence, self.sync, add="+")

    def follow(self, bar):
        """Scroll command that moves the scrollbar and the overlaid cells together."""
        def update(first, last):
            bar.set(first, last)
            self.sync()
        return update

    def clear(self) -> None:
        for label in self.cells.values():
            label.destroy()
        self.cells.clear()

    def add(self, item_id: str, column: str, text: str, color: str | None, *, anchor: str) -> None:
        if color is None:
            return
        label = tk.Label(self.tree, text=text, fg=color, font=FONTS["small"], anchor=anchor, padx=4, bd=0)
        label.bind("<Button-1>", lambda _event: self._select(item_id))
        if self.on_double:
            label.bind("<Double-1>", self.on_double)
        label.bind(
            "<MouseWheel>",
            lambda event: self.tree.yview_scroll(-1 if event.delta > 0 else 1, "units"),
        )
        self.cells[(item_id, column)] = label

    def refresh(self) -> None:
        """Place cells once the table has laid out its new rows."""
        self.tree.after_idle(self.sync)
        self.tree.after(60, self.sync)

    def _select(self, item_id: str) -> None:
        self.tree.selection_set(item_id)
        self.tree.focus(item_id)

    def sync(self, _event=None) -> None:
        selected = set(self.tree.selection())
        for (item_id, column), label in self.cells.items():
            box = self.tree.bbox(item_id, column)
            # The table skips a row that only partly fits, but bbox still reports it.
            if not box or box[1] + box[3] > self.tree.winfo_height():
                label.place_forget()
                continue
            if item_id in selected:
                background = COLORS["nav_active"]
            elif "even" in self.tree.item(item_id, "tags"):
                background = COLORS["card"]
            else:
                background = COLORS["row_alt"]
            label.configure(bg=background)
            x, y, width, height = box
            label.place(x=x, y=y, width=width, height=height)
