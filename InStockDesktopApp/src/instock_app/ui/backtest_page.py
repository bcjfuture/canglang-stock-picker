from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import customtkinter as ctk

from instock_app.settings import STRATEGIES
from .table_cells import ColoredCells, change_color
from .theme import COLORS, FONTS, card, primary_button, status_pill


PROFILE_NAMES = {key: spec.title for key, spec in STRATEGIES.items()}
PROFILE_ORDER = tuple(PROFILE_NAMES)
DETAIL_COLUMNS = (
    "rank",
    "code",
    "name",
    "entry_price",
    "exit_price",
    "gross_return",
    "net_return",
    "status",
)


def _percent(value, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    number = float(value)
    return f"{number:+.2%}" if signed else f"{number:.1%}"


def _signed_tag(text: str) -> str | None:
    """Text tag for a formatted signed percentage such as +1.20%."""
    try:
        value = float(str(text).rstrip("%"))
    except ValueError:
        return None
    color = change_color(value)
    if color is None:
        return None
    return "up" if value > 0 else "down"


def profile_segments(profile: dict | None, fallback_title: str) -> list[tuple[str, str | None]]:
    """Card text as (text, tag) pieces so gains and losses can be colored."""
    if profile is None:
        return [(f"{fallback_title}  无结果", None)]
    if profile["status"] != "ok":
        return [(f"{profile['title']}  失败\n", None), (profile["count"], "muted")]
    return [
        (f"{profile['title']}  ", None),
        (profile["mean"], _signed_tag(profile["mean"])),
        (f"  胜率{profile['win_rate']}  超额", None),
        (profile["excess"], _signed_tag(profile["excess"])),
        ("\n前5：", None),
        (profile["top5_mean"], _signed_tag(profile["top5_mean"])),
        (f" / 胜率{profile['top5_win_rate']} / 超额", None),
        (profile["top5_excess"], _signed_tag(profile["top5_excess"])),
        (f"\n{profile['count']}", "muted"),
    ]


def build_view_model(result: dict) -> dict:
    evaluation_date = str(result.get("evaluation_date", "—"))
    requested_evaluation_date = str(
        result.get("requested_evaluation_date") or evaluation_date
    )
    evaluation_date_display = evaluation_date
    if requested_evaluation_date != evaluation_date:
        evaluation_date_display = (
            f"{evaluation_date}（数据最新 {requested_evaluation_date}）"
        )
    horizons = []
    offsets = result.get("offsets", {})
    for offset_key in sorted(offsets, key=lambda value: int(value)):
        horizon = offsets[offset_key]
        profile_items = []
        profiles = horizon.get("profiles", {})
        ordered_keys = [key for key in PROFILE_ORDER if key in profiles]
        ordered_keys.extend(key for key in profiles if key not in ordered_keys)
        for key in ordered_keys:
            profile = profiles[key]
            status = str(profile.get("status", "error"))
            summary = profile.get("summary", {})
            if status == "ok":
                evaluated = int(summary.get("evaluated_count", 0) or 0)
                unavailable = int(summary.get("unavailable_count", 0) or 0)
                count = f"{evaluated}只有效 · {unavailable}只缺行情"
                mean = _percent(summary.get("mean_net_return"), signed=True)
                win_rate = _percent(summary.get("win_rate"))
                excess = _percent(summary.get("excess_return"), signed=True)
                benchmark = _percent(
                    summary.get("benchmark_return"), signed=True
                )
                buckets = summary.get("buckets", {})
                top5 = buckets.get("top5", {}) if isinstance(buckets, dict) else {}
                tail = buckets.get("rank6_20", {}) if isinstance(buckets, dict) else {}
                baseline_top5 = (
                    buckets.get("baseline_top5", {}) if isinstance(buckets, dict) else {}
                )
                top5_mean = _percent(top5.get("mean_net_return"), signed=True)
                top5_win_rate = _percent(top5.get("win_rate"))
                top5_excess = _percent(top5.get("excess_return"), signed=True)
                tail_win_rate = _percent(tail.get("win_rate"))
                baseline_top5_win_rate = _percent(baseline_top5.get("win_rate"))
                head_rerank_gate_passed = bool(
                    summary.get("head_rerank_gate_passed", False)
                )
            else:
                count = str(profile.get("error", "回测失败"))
                mean = "失败"
                win_rate = "—"
                excess = "—"
                benchmark = "—"
                top5_mean = "—"
                top5_win_rate = "—"
                top5_excess = "—"
                tail_win_rate = "—"
                baseline_top5_win_rate = "—"
                head_rerank_gate_passed = False
            profile_items.append(
                {
                    "key": key,
                    "title": str(profile.get("title") or PROFILE_NAMES.get(key, key)),
                    "status": status,
                    "mean": mean,
                    "win_rate": win_rate,
                    "excess": excess,
                    "benchmark": benchmark,
                    "count": count,
                    "top5_mean": top5_mean,
                    "top5_win_rate": top5_win_rate,
                    "top5_excess": top5_excess,
                    "tail_win_rate": tail_win_rate,
                    "baseline_top5_win_rate": baseline_top5_win_rate,
                    "head_rerank_gate_passed": head_rerank_gate_passed,
                    "details": list(profile.get("details", [])),
                }
            )
        signal_date = str(horizon.get("signal_date", "—"))
        requested_signal_date = str(
            horizon.get("requested_signal_date") or signal_date
        )
        signal_date_display = signal_date
        if horizon.get("snapshot_fallback"):
            actual_holding = int(horizon.get("holding_trading_days", 0) or 0)
            signal_date_display = (
                f"{signal_date}（目标 {requested_signal_date}，"
                f"实际前{actual_holding}日）"
            )
        horizons.append(
            {
                "offset": int(offset_key),
                "label": str(horizon.get("label", f"前{offset_key}个交易日")),
                "signal_date": signal_date,
                "signal_date_display": signal_date_display,
                "profiles": profile_items,
            }
        )
    return {
        "evaluation_date": evaluation_date,
        "evaluation_date_display": evaluation_date_display,
        "cost": float(result.get("cost", 0.003) or 0.003),
        "horizons": horizons,
        "warnings": list(result.get("warnings", [])),
    }


class BacktestPage(ctk.CTkFrame):
    PROFILE_NAMES = PROFILE_NAMES

    def __init__(self, parent, controller) -> None:
        super().__init__(parent, fg_color=COLORS["content_bg"], corner_radius=0)
        self.controller = controller
        self.view_model: dict = {}
        self.result_date = tk.StringVar(value="评估日 —")
        self.horizon_date_vars: dict[int, tk.StringVar] = {}
        self.profile_cards: dict[tuple[int, str], tk.Text] = {}
        self.detail_title = tk.StringVar(value="股票收益明细")
        self.grid_columnconfigure((0, 1, 2), weight=1, uniform="recent-backtest")
        self.grid_rowconfigure(2, weight=1)

        self._build_toolbar()
        self._build_horizon_cards()
        self._build_detail_table()
        existing = self.controller.load_backtest()
        if existing:
            self.display_result(existing)

    def _build_toolbar(self) -> None:
        toolbar = card(self)
        toolbar.grid(
            row=0, column=0, columnspan=3, sticky="ew", padx=26, pady=(14, 14)
        )
        toolbar.grid_columnconfigure(0, weight=1)
        copy = ctk.CTkFrame(toolbar, fg_color="transparent")
        copy.grid(row=0, column=0, sticky="w", padx=18, pady=14)
        ctk.CTkLabel(
            copy,
            text="近期选股回测",
            text_color=COLORS["text"],
            font=FONTS["card_title"],
        ).pack(anchor="w")
        ctk.CTkLabel(
            copy,
            text="按交易日验证：前1日 · 前3日 · 前5日  ·  下一交易日开盘买入",
            text_color=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(anchor="w", pady=(3, 0))
        status_pill(
            toolbar, "", textvariable=self.result_date, tone="neutral"
        ).grid(row=0, column=1, padx=8)
        primary_button(
            toolbar,
            "刷新近期回测",
            self.controller.run_backtest,
            width=138,
        ).grid(row=0, column=2, padx=(8, 18), pady=14)

    def _build_horizon_cards(self) -> None:
        for column, offset in enumerate((1, 3, 5)):
            item = card(self)
            item.grid(
                row=1,
                column=column,
                sticky="nsew",
                padx=(26 if column == 0 else 8, 26 if column == 2 else 8),
                pady=(0, 14),
            )
            date_var = tk.StringVar(value="信号日 —")
            self.horizon_date_vars[offset] = date_var
            header = ctk.CTkFrame(item, fg_color="transparent")
            header.pack(fill="x", padx=16, pady=(15, 8))
            ctk.CTkLabel(
                header,
                text=f"前{offset}个交易日",
                text_color=COLORS["text"],
                font=FONTS["card_title"],
            ).pack(side="left")
            ctk.CTkLabel(
                header,
                textvariable=date_var,
                text_color=COLORS["text_muted"],
                font=FONTS["small"],
            ).pack(side="right")

            for strategy_key in PROFILE_ORDER:
                self._build_profile_card(item, offset, strategy_key)

    def _build_profile_card(self, parent, offset: int, strategy_key: str) -> None:
        frame = ctk.CTkFrame(parent, fg_color=COLORS["card_alt"], corner_radius=9)
        frame.pack(fill="x", padx=14, pady=(0, 8))
        text = tk.Text(
            frame, height=3, width=1, wrap="none", bd=0, highlightthickness=0,
            bg=COLORS["card_alt"], fg=COLORS["text"], font=FONTS["small"],
            cursor="hand2", spacing1=1, spacing3=1,
        )
        text.pack(fill="x", padx=10, pady=8)
        text.tag_configure("up", foreground=COLORS["danger"])
        text.tag_configure("down", foreground=COLORS["success"])
        text.tag_configure("muted", foreground=COLORS["text_muted"])

        def select(_event) -> str:
            self._select_profile(offset, strategy_key)
            return "break"

        def hover(color: str):
            def apply(_event) -> None:
                frame.configure(fg_color=color)
                text.configure(bg=color)
            return apply

        text.bind("<Button-1>", select)
        text.bind("<B1-Motion>", lambda _event: "break")
        text.bind("<Double-1>", lambda _event: "break")
        frame.bind("<Button-1>", select)
        for widget in (frame, text):
            widget.bind("<Enter>", hover(COLORS["card_hover"]))
            widget.bind("<Leave>", hover(COLORS["card_alt"]))
        self.profile_cards[(offset, strategy_key)] = text
        self._set_profile_card(offset, strategy_key, [(f"{PROFILE_NAMES[strategy_key]}  等待回测", None)])

    def _set_profile_card(self, offset: int, strategy_key: str, segments) -> None:
        text = self.profile_cards[(offset, strategy_key)]
        text.configure(state="normal")
        text.delete("1.0", "end")
        for piece, tag in segments:
            text.insert("end", piece, (tag,) if tag else ())
        text.configure(state="disabled")

    def _build_detail_table(self) -> None:
        container = card(self, corner_radius=12)
        container.grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="nsew",
            padx=26,
            pady=(0, 26),
        )
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            container,
            textvariable=self.detail_title,
            text_color=COLORS["text"],
            font=FONTS["card_title"],
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 8))

        style = ttk.Style(self)
        style.configure(
            "Backtest.Treeview",
            background=COLORS["card"],
            fieldbackground=COLORS["card"],
            foreground="#DCE7F7",
            rowheight=30,
            borderwidth=0,
            relief="flat",
            font=FONTS["small"],
        )
        style.configure(
            "Backtest.Treeview.Heading",
            background=COLORS["card_alt"],
            foreground=COLORS["text_muted"],
            borderwidth=0,
            padding=(8, 9),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map(
            "Backtest.Treeview",
            background=[("selected", COLORS["nav_active"])],
            foreground=[("selected", COLORS["text"])],
        )
        headings = {
            "rank": "排名",
            "code": "代码",
            "name": "名称",
            "entry_price": "买入价",
            "exit_price": "最新收盘",
            "gross_return": "毛收益",
            "net_return": "净收益",
            "status": "状态",
        }
        widths = {
            "rank": 65,
            "code": 95,
            "name": 130,
            "entry_price": 105,
            "exit_price": 105,
            "gross_return": 105,
            "net_return": 105,
            "status": 230,
        }
        self.tree = ttk.Treeview(
            container,
            columns=DETAIL_COLUMNS,
            show="headings",
            height=11,
            style="Backtest.Treeview",
        )
        for column in DETAIL_COLUMNS:
            self.tree.heading(column, text=headings[column])
            self.tree.column(
                column,
                width=widths[column],
                minwidth=60,
                stretch=column in {"name", "status"},
                anchor="w" if column in {"name", "status"} else "center",
            )
        self.tree.tag_configure("even", background=COLORS["card"])
        self.tree.tag_configure("odd", background=COLORS["row_alt"])
        self.tree.grid(row=1, column=0, sticky="nsew", padx=(1, 0), pady=(0, 1))
        scrollbar = ttk.Scrollbar(
            container, orient="vertical", command=self.tree.yview
        )
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(0, 1))
        self.return_cells = ColoredCells(self.tree)
        self.tree.configure(yscrollcommand=self.return_cells.follow(scrollbar))

        self.empty_state = ctk.CTkLabel(
            container,
            text="运行近期回测后显示前1、3、5个交易日的结果",
            text_color=COLORS["text_muted"],
            font=FONTS["body"],
        )
        self.empty_state.place(relx=0.5, rely=0.58, anchor="center")

    def display_result(self, result: dict) -> None:
        self.view_model = build_view_model(result)
        self.result_date.set(
            f"评估日 {self.view_model['evaluation_date_display']}"
        )
        horizon_map = {
            item["offset"]: item for item in self.view_model.get("horizons", [])
        }
        first_selection = None
        for offset in (1, 3, 5):
            horizon = horizon_map.get(offset)
            self.horizon_date_vars[offset].set(
                f"信号日 {horizon['signal_date_display']}"
                if horizon
                else "信号日 —"
            )
            profile_map = (
                {item["key"]: item for item in horizon["profiles"]}
                if horizon
                else {}
            )
            for strategy_key in PROFILE_ORDER:
                profile = profile_map.get(strategy_key)
                if profile is not None and profile["status"] == "ok" and first_selection is None:
                    first_selection = (offset, strategy_key)
                self._set_profile_card(
                    offset, strategy_key, profile_segments(profile, PROFILE_NAMES[strategy_key])
                )
        if first_selection:
            self._select_profile(*first_selection)
        else:
            self._populate_details([])

    def _select_profile(self, offset: int, strategy_key: str) -> None:
        for horizon in self.view_model.get("horizons", []):
            if horizon["offset"] != offset:
                continue
            for profile in horizon["profiles"]:
                if profile["key"] != strategy_key:
                    continue
                self.detail_title.set(
                    f"{horizon['label']} · {profile['title']} · "
                    f"信号日 {horizon['signal_date_display']}"
                )
                self._populate_details(profile["details"])
                return

    def _populate_details(self, details: list[dict]) -> None:
        self.return_cells.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        for index, row in enumerate(details):
            entry_price = row.get("entry_price")
            exit_price = row.get("exit_price")
            gross_return = row.get("gross_return")
            net_return = row.get("net_return")
            values = (
                row.get("rank", ""),
                row.get("code", ""),
                row.get("name", ""),
                f"{float(entry_price):.2f}" if entry_price is not None else "—",
                f"{float(exit_price):.2f}" if exit_price is not None else "—",
                _percent(gross_return, signed=True),
                _percent(net_return, signed=True),
                row.get("status_text", "—"),
            )
            item_id = self.tree.insert(
                "", "end", values=values, tags=("even" if index % 2 == 0 else "odd",)
            )
            for column, value in (("gross_return", gross_return), ("net_return", net_return)):
                self.return_cells.add(
                    item_id, column, _percent(value, signed=True), change_color(value), anchor="center"
                )
        self.return_cells.refresh()
        if details:
            self.empty_state.place_forget()
        else:
            self.empty_state.place(relx=0.5, rely=0.58, anchor="center")
