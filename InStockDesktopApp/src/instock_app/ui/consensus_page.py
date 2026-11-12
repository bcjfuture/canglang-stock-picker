from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

import customtkinter as ctk

from instock_app.settings import STRATEGIES
from .theme import COLORS, FONTS, card, secondary_button, status_pill


HEAT_GLYPHS = ("·", "▂", "▄", "▆", "█")
WINDOW_LABELS = {3: "三", 5: "五"}


def consensus_window_label(day_limit: int) -> str:
    return WINDOW_LABELS.get(day_limit, str(day_limit))


def consensus_columns(day_limit: int) -> tuple[tuple[str, str, int, str], ...]:
    label = consensus_window_label(day_limit)
    return (
        ("rank", "排名", 58, "center"),
        ("code", "代码", 90, "center"),
        ("name", "名称", 120, "w"),
        ("count", "出现次数", 100, "center"),
        ("days", "覆盖天数", 100, "center"),
        ("strategies", "覆盖策略", 110, "center"),
        ("average_rank", "平均入选排名", 118, "center"),
        ("heat", f"{label}日热度（旧 → 新）", 230, "center"),
        ("last_seen", "最近入选", 112, "center"),
    )


def format_daily_heat(values: list[int]) -> str:
    return "  ".join(
        HEAT_GLYPHS[max(0, min(4, int(value)))]
        for value in values
    )


def format_coverage(result: dict) -> str:
    covered = int(result.get("covered_days", 0) or 0)
    day_limit = int(result.get("day_limit", 5) or 5)
    strategy_days = int(result.get("available_strategy_days", 0) or 0)
    maximum = int(result.get("maximum_appearances", 0) or 0)
    text = f"覆盖 {covered}/{day_limit}日 · {strategy_days}/{maximum}样本"
    skipped = int(result.get("skipped_snapshot_count", 0) or 0)
    if skipped:
        text += f" · 跳过{skipped}个异常快照"
    return text


def format_consensus_row(row: dict, result: dict) -> tuple[object, ...]:
    average_rank = float(row.get("average_rank", 0) or 0)
    return (
        int(row.get("rank", 0) or 0),
        str(row.get("code") or "—"),
        str(row.get("name") or "—"),
        f"{int(row.get('appearance_count', 0) or 0)}/{int(result.get('maximum_appearances', 15) or 15)}",
        f"{int(row.get('day_count', 0) or 0)}/{int(result.get('day_limit', 5) or 5)}日",
        f"{int(row.get('strategy_count', 0) or 0)}/{int(result.get('strategy_count', 3) or 3)}策略",
        f"{average_rank:.1f}",
        format_daily_heat(list(row.get("daily_counts", []))),
        str(row.get("last_seen") or "—"),
    )


class ConsensusPage(ctk.CTkFrame):
    COLUMNS = consensus_columns(5)

    def __init__(self, parent, controller, *, day_limit: int = 5) -> None:
        super().__init__(parent, fg_color=COLORS["content_bg"], corner_radius=0)
        self.controller = controller
        self.day_limit = day_limit
        self.window_label = consensus_window_label(day_limit)
        self.columns = consensus_columns(day_limit)
        self.result: dict = {}
        self.rows_by_item: dict[str, dict] = {}
        self._refresh_serial = 0
        self._refresh_queue: queue.Queue[tuple[int, str, object]] = queue.Queue()
        self._refresh_thread: threading.Thread | None = None
        self._refresh_running = False
        self._refresh_pending = False
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.period_text = tk.StringVar(value="统计区间 —")
        self.coverage_text = tk.StringVar(value="等待历史数据")
        self.updated_text = tk.StringVar(value="尚未刷新")
        self.detail_title = tk.StringVar(value="选择一只股票查看每日入选明细")
        self.detail_text = tk.StringVar(value="—")

        self._build_summary()
        self._build_ranking()
        self._configure_tree_style()

    def _build_summary(self) -> None:
        summary = card(self)
        summary.grid(row=0, column=0, sticky="ew", padx=26, pady=(14, 14))
        summary.grid_columnconfigure(0, weight=1)

        copy = ctk.CTkFrame(summary, fg_color="transparent")
        copy.grid(row=0, column=0, sticky="w", padx=20, pady=16)
        ctk.CTkLabel(
            copy,
            text=f"三策略{self.window_label}日共识排行",
            text_color=COLORS["text"],
            font=FONTS["section_title"],
        ).pack(anchor="w")
        ctk.CTkLabel(
            copy,
            text=f"统计最近{self.day_limit}个交易日重复入选股票 · 同日同策略只计1次",
            text_color=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(anchor="w", pady=(4, 0))

        pills = ctk.CTkFrame(summary, fg_color="transparent")
        pills.grid(row=0, column=1, sticky="e", padx=8, pady=16)
        status_pill(pills, "", textvariable=self.period_text).pack(side="left", padx=4)
        status_pill(
            pills, "", textvariable=self.coverage_text, tone="primary"
        ).pack(side="left", padx=4)
        status_pill(
            pills, "", textvariable=self.updated_text, tone="success"
        ).pack(side="left", padx=4)

        secondary_button(
            summary, "刷新排行", self.refresh, width=108
        ).grid(row=0, column=2, sticky="e", padx=(8, 20), pady=16)

    def _build_ranking(self) -> None:
        body = card(self, corner_radius=12)
        body.grid(row=1, column=0, sticky="nsew", padx=26, pady=(0, 26))
        body.grid_rowconfigure(1, weight=1)
        body.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            body,
            text=(
                f"出现次数表示三个策略在{self.window_label}个交易日中的累计命中（不含全市场精选）；"
                "热度字符越高，代表当天命中的策略越多。"
            ),
            text_color=COLORS["text_muted"],
            font=FONTS["small"],
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(10, 7))

        columns = [column[0] for column in self.columns]
        self.tree = ttk.Treeview(
            body,
            columns=columns,
            show="headings",
            style="Consensus.Treeview",
            selectmode="browse",
        )
        for key, heading, width, anchor in self.columns:
            self.tree.heading(key, text=heading)
            self.tree.column(
                key, width=width, minwidth=width, stretch=key == "heat", anchor=anchor
            )
        self.tree.grid(row=1, column=0, sticky="nsew", padx=(1, 0))
        vertical = ttk.Scrollbar(body, orient="vertical", command=self.tree.yview)
        vertical.grid(row=1, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(body, orient="horizontal", command=self.tree.xview)
        horizontal.grid(row=2, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.bind("<<TreeviewSelect>>", self._show_selected_detail)

        detail = ctk.CTkFrame(body, fg_color=COLORS["card_alt"], corner_radius=9)
        detail.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        detail.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            detail,
            textvariable=self.detail_title,
            text_color=COLORS["text"],
            font=FONTS["body_bold"],
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 4))
        ctk.CTkLabel(
            detail,
            textvariable=self.detail_text,
            text_color=COLORS["text_muted"],
            font=FONTS["small"],
            anchor="w",
            justify="left",
            wraplength=1240,
        ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

        self.empty_label = ctk.CTkLabel(
            body,
            text="暂无选股历史，请先运行策略",
            text_color=COLORS["text_muted"],
            font=FONTS["body"],
            fg_color=COLORS["card_alt"],
            corner_radius=10,
            padx=18,
            pady=12,
        )
        self.empty_label.place(relx=0.5, rely=0.45, anchor="center")

    def _configure_tree_style(self) -> None:
        style = ttk.Style(self)
        style.configure(
            "Consensus.Treeview",
            background=COLORS["card"],
            fieldbackground=COLORS["card"],
            foreground=COLORS["text"],
            rowheight=48,
            borderwidth=0,
            font=FONTS["small"],
        )
        style.configure(
            "Consensus.Treeview.Heading",
            background=COLORS["card_alt"],
            foreground=COLORS["text_muted"],
            relief="flat",
            font=FONTS["small"],
        )
        style.map(
            "Consensus.Treeview",
            background=[("selected", COLORS["nav_active"])],
            foreground=[("selected", COLORS["text"])],
        )
        self.tree.tag_configure("top1", background="#2A2517")
        self.tree.tag_configure("top2", background="#1C2533")
        self.tree.tag_configure("top3", background="#2A211D")
        self.tree.tag_configure("even", background=COLORS["card"])
        self.tree.tag_configure("odd", background=COLORS["row_alt"])

    def refresh(self) -> None:
        if getattr(self, "_refresh_running", False):
            self._refresh_pending = True
            self.updated_text.set("等待再次刷新")
            return
        self._refresh_serial += 1
        serial = self._refresh_serial
        if not hasattr(self, "_refresh_queue"):
            self._refresh_queue = queue.Queue()
        self._refresh_running = True
        self._refresh_pending = False
        self.updated_text.set("刷新中")
        self._refresh_thread = threading.Thread(
            target=self._load_consensus,
            args=(serial,),
            name=f"consensus-{self.day_limit}-day-loader",
            daemon=True,
        )
        self._refresh_thread.start()
        self.after(50, lambda: self._poll_refresh(serial))

    def _load_consensus(self, serial: int) -> None:
        try:
            result = self.controller.load_consensus(self.day_limit)
        except Exception as exc:
            self._refresh_queue.put((serial, "error", exc))
        else:
            self._refresh_queue.put((serial, "result", result))

    def _poll_refresh(self, serial: int) -> None:
        if serial != self._refresh_serial:
            return
        while True:
            try:
                result_serial, kind, value = self._refresh_queue.get_nowait()
            except queue.Empty:
                if serial == self._refresh_serial:
                    self.after(50, lambda: self._poll_refresh(serial))
                return
            if result_serial != self._refresh_serial:
                continue
            self._refresh_running = False
            if self._refresh_pending:
                self._refresh_pending = False
                self.refresh()
                return
            if kind == "error":
                self._display_error(value)
            else:
                self.display_result(value)
            return

    def _display_error(self, error: object) -> None:
        self.result = {}
        self.tree.delete(*self.tree.get_children())
        self.rows_by_item.clear()
        self.period_text.set("统计区间 —")
        self.coverage_text.set("历史数据不可用")
        self.updated_text.set("读取失败")
        self.detail_title.set(f"无法读取{self.window_label}日共识")
        self.detail_text.set(str(error))
        self.empty_label.configure(text="共识排行读取失败")
        self.empty_label.place(relx=0.5, rely=0.45, anchor="center")

    def display_result(self, result: dict) -> None:
        self.result = result
        self.empty_label.configure(text="暂无选股历史，请先运行策略")
        trade_dates = list(result.get("trade_dates", []))
        if trade_dates:
            self.period_text.set(f"{trade_dates[0]} 至 {trade_dates[-1]}")
        else:
            self.period_text.set("统计区间 —")
        self.coverage_text.set(format_coverage(result))
        generated_at = str(result.get("generated_at") or "")
        self.updated_text.set(
            f"更新 {generated_at[11:16]}" if len(generated_at) >= 16 else "已刷新"
        )

        self.tree.delete(*self.tree.get_children())
        self.rows_by_item.clear()
        rows = list(result.get("rows", []))
        for index, row in enumerate(rows):
            tag = f"top{index + 1}" if index < 3 else ("even" if index % 2 == 0 else "odd")
            item = self.tree.insert(
                "", "end", values=format_consensus_row(row, result), tags=(tag,)
            )
            self.rows_by_item[item] = row

        if not rows:
            self.empty_label.place(relx=0.5, rely=0.45, anchor="center")
            self.detail_title.set("暂无选股历史")
            self.detail_text.set("运行任一选股策略后，此处会自动生成共识排行。")
            return

        self.empty_label.place_forget()
        first = self.tree.get_children()[0]
        self.tree.selection_set(first)
        self.tree.focus(first)
        self._show_selected_detail()

    def _show_selected_detail(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        row = self.rows_by_item.get(selection[0])
        if not row:
            return
        self.detail_title.set(
            f"{row.get('name') or '—'}（{row.get('code') or '—'}） · "
            f"累计{row.get('appearance_count', 0)}次 · 覆盖{row.get('day_count', 0)}个交易日"
        )
        lines = []
        for detail in row.get("details", []):
            hit_text = []
            for hit in detail.get("hits", []):
                key = str(hit.get("strategy") or "")
                title = STRATEGIES[key].title if key in STRATEGIES else key
                hit_text.append(f"{title} 第{int(hit.get('rank', 0) or 0)}名")
            lines.append(f"{detail.get('trade_date', '—')}    {' ｜ '.join(hit_text)}")
        self.detail_text.set("\n".join(lines) or "—")
