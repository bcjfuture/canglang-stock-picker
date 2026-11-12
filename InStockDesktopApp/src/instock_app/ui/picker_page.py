from __future__ import annotations

import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import customtkinter as ctk

from instock_app.settings import DEFAULT_STRATEGY_KEY, STRATEGIES
from .table_cells import ColoredCells, change_color
from .theme import COLORS, FONTS, card, primary_button, secondary_button, status_pill


PICKER_COLUMN_LAYOUT = {
    "rank": (58, 58, False),
    "code": (86, 86, False),
    "name": (100, 100, False),
    "industry": (110, 110, False),
    "price": (76, 76, False),
    "change_pct": (86, 86, False),
    "holding_days": (76, 76, False),
    "take_profit": (98, 98, False),
    "stop_loss": (98, 98, False),
    "score": (72, 72, False),
    "selection_tier": (92, 92, False),
    "pb": (66, 66, False),
    "market_earnings_ratio": (92, 92, False),
    "news_score": (76, 76, False),
    "news_summary": (220, 220, True),
    "win_probability": (100, 100, False),
    "market_regime": (82, 82, False),
    "probability_threshold": (88, 88, False),
    "large_order_flow": (100, 100, False),
    "retail_flow": (100, 100, False),
    "fund_flow_1d": (96, 96, False),
    "fund_flow_3d": (96, 96, False),
    "fund_flow_5d": (96, 96, False),
    "shareholder_score": (72, 72, False),
    "reason": (280, 280, True),
}
ADVICE_DISCLAIMER = (
    "标“参考”的持有期和止盈止损是通用规则（ATR测算），不是该策略的回测结论；"
    "价格按本次现价测算，实际成交后请按比例重算；历史回测不代表未来收益。"
)
PICKER_ROW_HEIGHT = 56
NUMERIC_COLUMNS = {
    "price", "change_pct", "take_profit", "stop_loss", "score", "pb", "market_earnings_ratio",
    "news_score", "large_order_flow", "retail_flow", "fund_flow_1d", "fund_flow_3d",
    "fund_flow_5d", "shareholder_score",
}
PICKER_VISIBLE_ROWS = 10
HIGH_PRECISION_PICKER_COLUMNS = (
    "win_probability",
    "market_regime",
    "probability_threshold",
)
MARKET_EARNINGS_PICKER_COLUMNS = ("pb", "market_earnings_ratio")


def build_picker_columns(strategies) -> tuple[str, ...]:
    return (
        "rank", "code", "name", "industry", "price", "change_pct", "holding_days",
        "take_profit", "stop_loss", "score", "selection_tier",
        *(MARKET_EARNINGS_PICKER_COLUMNS if "all_market" in strategies else ()),
        "news_score", "news_summary",
        *(HIGH_PRECISION_PICKER_COLUMNS if "high_precision" in strategies else ()),
        "large_order_flow", "retail_flow",
        "fund_flow_1d", "fund_flow_3d", "fund_flow_5d",
        "shareholder_score", "reason",
    )


def build_picker_headings() -> dict[str, str]:
    return {
        "rank": "排名", "code": "代码", "name": "名称", "industry": "行业",
        "price": "现价", "change_pct": "涨跌幅", "holding_days": "建议持有",
        "take_profit": "止盈价", "stop_loss": "止损价", "score": "评分",
        "selection_tier": "入选类型",
        "pb": "PB", "market_earnings_ratio": "市赚率3.0",
        "news_score": "新闻分", "news_summary": "新闻摘要",
        "large_order_flow": "大单净额", "retail_flow": "散户净额",
        "fund_flow_1d": "1日净流向", "fund_flow_3d": "3日净流向",
        "fund_flow_5d": "5日净流向",
        "shareholder_score": "股东分",
        "win_probability": "5日上涨概率",
        "market_regime": "市场状态",
        "probability_threshold": "概率门槛",
        "reason": "核心理由",
    }


WRAP_DELIMITERS = "；，,、"
DETAIL_TILES = {
    "评分": ("score", "{:.1f}"),
    "PE": ("pe", "{:.2f}"),
    "PB": ("pb", "{:.2f}"),
    "ROE": ("roe", "{:.1f}%"),
    "净利同比": ("profit_growth", "{:+.1f}%"),
}


def _format_change_pct(value: object) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    if number == 0:
        return "0.00%"
    if number > 0:
        return f"+{number:.2f}%"
    return f"{number:.2f}%"


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_signed_flow(value: object) -> str:
    number = _finite_number(value)
    if number is None:
        return "—"
    if number == 0:
        return "0万"
    if abs(number) >= 10000:
        return f"{number / 10000:+.2f}亿"
    return f"{number:+,.0f}万"


def format_order_flow_detail(row: dict) -> str:
    source = str(row.get("order_flow_source") or "分项暂缺")
    trade_date = str(row.get("order_flow_as_of") or "")
    provenance = f"{trade_date}｜{source}" if trade_date else source
    return (
        "大单净额（净超大单+净大单） "
        f"{format_signed_flow(row.get('large_order_flow'))}"
        "    散户净额（净中单+净小单） "
        f"{format_signed_flow(row.get('retail_flow'))}"
        f"    {provenance}"
        "\n资金净流向：1日 "
        f"{format_signed_flow(row.get('fund_flow_1d'))}    3日 "
        f"{format_signed_flow(row.get('fund_flow_3d'))}    5日 "
        f"{format_signed_flow(row.get('fund_flow_5d'))}"
    )


def format_shareholder_review(row: dict) -> str:
    score = _finite_number(row.get("shareholder_score"))
    score_text = f"{score:+.1f}分" if score is not None else "—"
    summary = str(row.get("shareholder_review") or "股东结构数据不足")
    return f"股东结构 {score_text}｜{summary}｜定期披露口径，存在滞后"


def format_news_detail(row: dict) -> str:
    score = _finite_number(row.get("news_score"))
    status = str(row.get("news_status") or "未评估")
    summary = str(row.get("news_summary") or "").strip()
    if status == "资讯不可用":
        summary = "资讯不可用（按0分处理）"
    elif not summary:
        summary = status
    score_text = f"{score:+.1f}分" if score is not None else "—"
    cutoff = str(row.get("news_as_of") or "").strip()
    suffix = f"｜截至{cutoff}" if cutoff else ""
    return f"新闻因子 {score_text}｜{summary}{suffix}"


def summarize_reason(value: object, limit: int = 3) -> str:
    parts = [
        part.strip()
        for part in str(value or "").replace("，", "；").split("；")
        if part.strip()
    ]
    unique = list(dict.fromkeys(parts))
    if not unique:
        return "—"
    visible = "；".join(unique[:limit])
    remaining = len(unique) - min(limit, len(unique))
    suffix = f"；等{remaining}项" if remaining else ""
    return f"✓ {visible}{suffix}"


def _format_holding_days(value: object, advice_status: object = None) -> str:
    try:
        days = int(value)
    except (TypeError, ValueError):
        return "—"
    if days <= 0:
        return "—"
    return f"{days}日\n参考" if advice_status == "reference_only" else f"{days}日"


def format_selection_tier(row: dict) -> str:
    tier = str(row.get("selection_tier") or "").strip() or "—"
    runs = _finite_number(row.get("holding_runs"))
    return f"{tier}\n第{int(runs)}次" if runs is not None and runs > 1 else tier


def format_result_status(result: dict, latest_data_date: str | None) -> tuple[str, bool]:
    """Status line for a result, and whether it needs the reader's attention."""
    trade_date = str(result.get("trade_date") or "")[:10]
    if not trade_date:
        return "", False
    generated = str(result.get("generated_at") or "")[:16].replace("T", " ")
    text = f"选股交易日 {trade_date}"
    if generated:
        text += f" · 生成于 {generated}"
    warnings = []
    if latest_data_date and trade_date < latest_data_date:
        warnings.append(f"数据已更新到{latest_data_date}，这份结果已过期，请重新选股")
    summary = result.get("source_summary")
    data_warning = str(summary.get("data_warning") or "") if isinstance(summary, dict) else ""
    if data_warning:
        warnings.append(data_warning)
    if warnings:
        text += "｜" + "；".join(warnings)
    return text, bool(warnings)


def _format_advice_price(price: object, pct: object) -> str:
    numeric_price = _finite_number(price)
    numeric_pct = _finite_number(pct)
    if numeric_price is None or numeric_pct is None:
        return "—"
    return f"{numeric_price:.2f}\n{numeric_pct:+.1%}"


def format_market_regime(value: object) -> str:
    regimes = {
        "strong": "强势",
        "weak": "弱势",
        "good": "良好",
        "normal": "一般",
        "poor": "糟糕",
    }
    text = str(value or "")
    return regimes.get(text, text or "—")


def _two_line_text(value: object, prefix: str = "", target: int = 50) -> str:
    text = " ".join(str(value or "—").split())
    rendered = prefix + text
    if len(rendered) <= target:
        return rendered

    midpoint = len(rendered) // 2
    candidates = [
        index + 1
        for index, char in enumerate(rendered)
        if char in WRAP_DELIMITERS and len(prefix) < index + 1 < len(rendered)
    ]
    split_at = (
        min(
            candidates,
            key=lambda index: (max(index, len(rendered) - index), abs(index - midpoint)),
        )
        if candidates
        else midpoint
    )
    return rendered[:split_at].rstrip() + "\n" + rendered[split_at:].lstrip()


def format_picker_row(row: dict) -> tuple:
    formatted = dict(row)
    formatted["change_pct"] = _format_change_pct(row.get("change_pct"))
    formatted["holding_days"] = _format_holding_days(
        row.get("advice_holding_days"), row.get("advice_status")
    )
    formatted["selection_tier"] = format_selection_tier(row)
    take_pct = _finite_number(row.get("take_profit_pct"))
    formatted["take_profit"] = _format_advice_price(
        row.get("take_profit_price"), abs(take_pct) if take_pct is not None else None
    )
    stop_pct = _finite_number(row.get("stop_loss_pct"))
    formatted["stop_loss"] = _format_advice_price(
        row.get("stop_loss_price"), -abs(stop_pct) if stop_pct is not None else None
    )
    formatted["large_order_flow"] = format_signed_flow(row.get("large_order_flow"))
    formatted["retail_flow"] = format_signed_flow(row.get("retail_flow"))
    formatted["fund_flow_1d"] = format_signed_flow(row.get("fund_flow_1d"))
    formatted["fund_flow_3d"] = format_signed_flow(row.get("fund_flow_3d"))
    formatted["fund_flow_5d"] = format_signed_flow(row.get("fund_flow_5d"))
    shareholder_score = _finite_number(row.get("shareholder_score"))
    formatted["shareholder_score"] = (
        f"{shareholder_score:+.1f}" if shareholder_score is not None else "—"
    )
    news_score = _finite_number(row.get("news_score"))
    formatted["news_score"] = f"{news_score:+.1f}" if news_score is not None else "—"
    news_status = str(row.get("news_status") or "未评估")
    news_summary = str(row.get("news_summary") or "").strip()
    if news_status == "资讯不可用":
        news_summary = "资讯不可用（按0分处理）"
    elif not news_summary:
        news_summary = news_status
    formatted["news_summary"] = _two_line_text(news_summary, target=22)
    formatted["reason"] = _two_line_text(summarize_reason(row.get("reason")), target=24)
    probability = _finite_number(row.get("win_probability"))
    formatted["win_probability"] = (
        f"{probability:.1%}" if probability is not None else "—"
    )
    threshold = _finite_number(row.get("probability_threshold"))
    formatted["probability_threshold"] = (
        f"{threshold:.1%}" if threshold is not None else "—"
    )
    formatted["market_regime"] = format_market_regime(row.get("market_regime"))
    pb = _finite_number(row.get("pb"))
    formatted["pb"] = f"{pb:.2f}" if pb is not None else "—"
    market_ratio = _finite_number(row.get("market_earnings_ratio"))
    formatted["market_earnings_ratio"] = (
        f"{market_ratio:.3f}" if market_ratio is not None else "—"
    )
    return tuple(formatted.get(column, "") for column in PickerPage.COLUMNS)


class PickerPage(ctk.CTkFrame):
    _HIGH_PRECISION_COLUMNS = HIGH_PRECISION_PICKER_COLUMNS
    COLUMNS = build_picker_columns(STRATEGIES)

    def __init__(self, parent, controller) -> None:
        super().__init__(parent, fg_color=COLORS["content_bg"], corner_radius=0)
        self.controller = controller
        self.rows: list[dict] = []
        self.visible_rows_by_item: dict[str, dict] = {}
        self.abstain_reason = ""
        self.title_to_key = {spec.title: key for key, spec in STRATEGIES.items()}
        self.strategy = tk.StringVar(
            value=STRATEGIES[DEFAULT_STRATEGY_KEY].title
        )
        self.search = tk.StringVar()
        self.result_count = tk.StringVar(value="等待选股")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._build_toolbar()
        self._build_table()
        self.search.trace_add("write", lambda *_: self._render())

    def _build_toolbar(self) -> None:
        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.grid(row=1, column=0, sticky="ew", padx=26, pady=(14, 12))
        toolbar.grid_columnconfigure(3, weight=1)
        self.strategy_control = ctk.CTkSegmentedButton(
            toolbar,
            values=list(self.title_to_key),
            variable=self.strategy,
            command=self.on_strategy_change,
            fg_color=COLORS["content_bg"],
            selected_color=COLORS["primary"],
            selected_hover_color=COLORS["primary_hover"],
            unselected_color=COLORS["card_alt"],
            unselected_hover_color=COLORS["card_hover"],
            text_color=COLORS["text"],
            border_width=0,
            font=FONTS["body_bold"],
            height=38,
            corner_radius=9,
        )
        self.strategy_control.grid(row=0, column=0, padx=(0, 12))
        primary_button(toolbar, "立即选股", self.run_selected, width=96).grid(row=0, column=1, padx=4)
        secondary_button(toolbar, "导出 Excel", self.export_selected, width=96).grid(row=0, column=2, padx=4)
        search_box = ctk.CTkFrame(toolbar, fg_color="transparent")
        search_box.grid(row=0, column=3, sticky="e")
        self.search_entry = ctk.CTkEntry(
            search_box,
            placeholder_text="搜索股票代码或名称",
            width=180,
            height=38,
            corner_radius=10,
            fg_color=COLORS["card"],
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            placeholder_text_color=COLORS["text_muted"],
            font=FONTS["body"],
        )
        self.search_entry.pack(side="left", padx=8)
        # A bound textvariable suppresses the placeholder, so sync on key release.
        self.search_entry.bind("<KeyRelease>", lambda _event: self.search.set(self.search_entry.get()))
        self.count_pill = status_pill(search_box, "", textvariable=self.result_count, tone="primary")
        self.count_pill.pack(side="left", padx=(8, 0))

    def _build_table(self) -> None:
        container = card(self, corner_radius=12)
        container.grid(row=2, column=0, sticky="nsew", padx=26, pady=(0, 26))
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)

        style = ttk.Style(self)
        style.configure(
            "Picker.Treeview",
            background=COLORS["card"],
            fieldbackground=COLORS["card"],
            foreground="#DCE7F7",
            rowheight=PICKER_ROW_HEIGHT,
            borderwidth=0,
            relief="flat",
            font=FONTS["small"],
        )
        style.configure(
            "Picker.Treeview.Heading",
            background=COLORS["card_alt"],
            foreground=COLORS["text_muted"],
            borderwidth=0,
            relief="flat",
            padding=(8, 10),
            font=("Microsoft YaHei UI", 11, "bold"),
        )
        style.map(
            "Picker.Treeview",
            background=[("selected", COLORS["nav_active"])],
            foreground=[("selected", COLORS["text"])],
        )
        style.map("Picker.Treeview.Heading", background=[("active", COLORS["card_hover"])])

        headings = build_picker_headings()
        notes = ctk.CTkFrame(container, fg_color="transparent")
        notes.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(7, 5))
        self.result_status = tk.StringVar(value="")
        self.result_status_label = ctk.CTkLabel(
            notes, textvariable=self.result_status, text_color=COLORS["accent"],
            font=FONTS["small"], anchor="w", justify="left", wraplength=1200,
        )
        self.result_status_label.pack(fill="x")
        ctk.CTkLabel(
            notes, text=ADVICE_DISCLAIMER, text_color=COLORS["text_muted"],
            font=FONTS["small"], anchor="w", justify="left", wraplength=1200,
        ).pack(fill="x")
        self.tree = ttk.Treeview(
            container,
            columns=self.COLUMNS,
            show="headings",
            height=PICKER_VISIBLE_ROWS,
            style="Picker.Treeview",
        )
        for column in self.COLUMNS:
            if column in NUMERIC_COLUMNS:
                anchor = "e"
            else:
                anchor = "w" if column in {"reason", "name", "industry"} else "center"
            self.tree.heading(column, text=headings[column], anchor=anchor)
            width, minwidth, stretch = PICKER_COLUMN_LAYOUT[column]
            self.tree.column(
                column,
                width=width,
                minwidth=minwidth,
                stretch=stretch,
                anchor=anchor,
            )
        self.tree.tag_configure("even", background=COLORS["card"])
        self.tree.tag_configure("odd", background=COLORS["row_alt"])
        self.tree.grid(row=1, column=0, sticky="nsew", padx=(1, 0), pady=(1, 0))
        vertical = ttk.Scrollbar(container, orient="vertical", command=self.tree.yview)
        vertical.grid(row=1, column=1, sticky="ns", pady=1)
        horizontal = ttk.Scrollbar(container, orient="horizontal", command=self.tree.xview)
        horizontal.grid(row=2, column=0, sticky="ew", padx=1)
        self.tree.bind("<Double-1>", self.copy_code)
        self.tree.bind("<<TreeviewSelect>>", self._show_selected_detail)
        self.change_cells = ColoredCells(self.tree, on_double=self.copy_code)
        self.tree.configure(
            yscrollcommand=self.change_cells.follow(vertical),
            xscrollcommand=self.change_cells.follow(horizontal),
        )

        detail_card = ctk.CTkFrame(container, fg_color=COLORS["card_alt"], corner_radius=8)
        detail_card.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 10))
        detail_card.grid_columnconfigure(0, weight=1)
        detail_card.grid_columnconfigure(1, weight=1)
        self.detail_title = tk.StringVar(value="选择一只股票查看完整分析")
        self.detail_flow = tk.StringVar(value=format_order_flow_detail({}))
        self.detail_reason = tk.StringVar(value="入选理由：—")
        self.detail_performance = tk.StringVar(value="业绩复核：—")
        self.detail_shareholder = tk.StringVar(value=format_shareholder_review({}))
        self.detail_risk = tk.StringVar(value="风险提示：—")
        header = ctk.CTkFrame(detail_card, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(8, 4))
        ctk.CTkLabel(
            header, textvariable=self.detail_title, text_color=COLORS["text"],
            font=FONTS["body_bold"], anchor="w", justify="left",
        ).pack(side="left")
        tiles = ctk.CTkFrame(header, fg_color="transparent")
        tiles.pack(side="right")
        self.detail_tiles: dict[str, tk.StringVar] = {}
        for label in DETAIL_TILES:
            tile = ctk.CTkFrame(tiles, fg_color=COLORS["card"], corner_radius=8)
            tile.pack(side="left", padx=4)
            value = tk.StringVar(value="—")
            self.detail_tiles[label] = value
            ctk.CTkLabel(tile, textvariable=value, text_color=COLORS["text"], font=FONTS["body_bold"], width=62, height=22).pack(padx=6, pady=(5, 0))
            ctk.CTkLabel(tile, text=label, text_color=COLORS["text_muted"], font=FONTS["small"], height=18).pack(padx=6, pady=(0, 5))
        detail_card.grid_columnconfigure((0, 1), uniform="detail")
        self.detail_half_labels = []
        for row, column, variable, color in (
            (1, 0, self.detail_reason, COLORS["text"]),
            (1, 1, self.detail_risk, COLORS["warning"]),
            (2, 0, self.detail_performance, COLORS["text_muted"]),
            (2, 1, self.detail_flow, COLORS["text_muted"]),
        ):
            label = ctk.CTkLabel(
                detail_card, textvariable=variable, text_color=color,
                font=FONTS["small"], anchor="nw", justify="left", wraplength=560,
            )
            label.grid(
                row=row, column=column, sticky="new",
                padx=(12, 8) if column == 0 else (8, 12), pady=4,
            )
            self.detail_half_labels.append(label)
        self._detail_width = 0
        detail_card.bind("<Configure>", self._fit_detail_text)
        ctk.CTkLabel(
            detail_card, textvariable=self.detail_shareholder, text_color=COLORS["text_muted"],
            font=FONTS["small"], anchor="w", justify="left", wraplength=1200,
        ).grid(row=3, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 9))

        self.empty_state = ctk.CTkFrame(container, fg_color=COLORS["card"], corner_radius=0)
        ctk.CTkLabel(self.empty_state, text="◇", text_color=COLORS["primary"], font=("Segoe UI Symbol", 44)).pack(pady=(24, 4))
        self.empty_title = tk.StringVar(value="还没有选股结果")
        self.empty_detail = tk.StringVar(value="选择上方策略并点击“立即选股”")
        ctk.CTkLabel(self.empty_state, textvariable=self.empty_title, text_color=COLORS["text"], font=FONTS["section_title"]).pack()
        ctk.CTkLabel(self.empty_state, textvariable=self.empty_detail, text_color=COLORS["text_muted"], font=FONTS["body"]).pack(pady=(6, 22))
        self.empty_state.place(relx=0.5, rely=0.5, anchor="center")

    def _fit_detail_text(self, event) -> None:
        """Wrap to the real column width so text never runs past the card."""
        if event.width == self._detail_width:
            return
        self._detail_width = event.width
        column = int(event.width / self._get_widget_scaling() / 2) - 30
        for label in self.detail_half_labels:
            label.configure(wraplength=max(200, column))

    def load_latest(self) -> None:
        """Show the saved result on start-up instead of an empty table."""
        try:
            result = self.controller.load_picker_result(self.selected_key())
        except (OSError, ValueError):
            return
        if result:
            self.display_result(result)

    def selected_key(self) -> str:
        return self.title_to_key[self.strategy.get()]

    def run_selected(self) -> None:
        self.controller.run_picker(self.selected_key())

    def on_strategy_change(self, title: str) -> None:
        key = self.title_to_key[title]
        result = self.controller.load_picker_result(key)
        self.display_result(result or {"strategy": key, "rows": []})

    def export_selected(self) -> None:
        try:
            desktop = Path.home() / "Desktop"
            self.controller.export_latest(self.selected_key(), desktop)
        except Exception as exc:
            messagebox.showerror("无法导出", str(exc))

    def display_result(self, result: dict) -> None:
        key = str(result.get("strategy", "all_market"))
        if key in STRATEGIES:
            self.strategy.set(STRATEGIES[key].title)
        self.rows = list(result.get("rows", []))
        status, needs_attention = format_result_status(result, self.controller.latest_data_date())
        self.result_status.set(status)
        self.result_status_label.configure(
            text_color=COLORS["warning"] if needs_attention else COLORS["accent"]
        )
        self.abstain_reason = str(
            result.get("abstain_reason")
            or result.get("source_summary", {}).get("abstain_reason")
            or ""
        )
        self._render()

    def _render(self) -> None:
        query = self.search.get().strip().lower()
        selected = self.tree.selection()
        selected_row = self.visible_rows_by_item.get(selected[0]) if selected else None
        selected_code = str(selected_row.get("code", "")) if selected_row else None
        self._clear_detail()
        self.visible_rows_by_item.clear()
        self.change_cells.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        visible = []
        for row in self.rows:
            if query and query not in str(row.get("code", "")) and query not in str(row.get("name", "")).lower():
                continue
            visible.append(row)
        for index, row in enumerate(visible):
            item_id = self.tree.insert(
                "",
                "end",
                values=format_picker_row(row),
                tags=("even" if index % 2 == 0 else "odd",),
            )
            self.visible_rows_by_item[item_id] = row
            self.change_cells.add(
                item_id, "change_pct", _format_change_pct(row.get("change_pct")),
                change_color(row.get("change_pct")), anchor="e",
            )
        self.change_cells.refresh()
        self.result_count.set(
            f"{len(visible)} 只结果"
            if visible
            else ("本次空仓" if self.abstain_reason else "等待选股")
        )
        if visible:
            self.empty_state.place_forget()
            selected_item = next(
                (
                    item_id
                    for item_id, row in self.visible_rows_by_item.items()
                    if str(row.get("code", "")) == selected_code
                ),
                None,
            )
            selected_item = selected_item or self.tree.get_children()[0]
            self.tree.selection_set(selected_item)
            self.tree.focus(selected_item)
            self._show_selected_detail()
        else:
            self.empty_title.set(
                f"{self.strategy.get()}本次保持空仓"
                if self.abstain_reason
                else "还没有选股结果"
            )
            self.empty_detail.set(
                self.abstain_reason
                or "选择上方策略并点击“立即选股”"
            )
            self.empty_state.place(relx=0.5, rely=0.5, anchor="center")

    def _clear_detail(self) -> None:
        self.detail_title.set("选择一只股票查看完整分析")
        self.detail_flow.set(format_order_flow_detail({}))
        self.detail_reason.set("入选理由：—")
        self.detail_performance.set("业绩复核：—")
        self.detail_shareholder.set(format_shareholder_review({}))
        self.detail_risk.set("风险提示：—")
        for value in self.detail_tiles.values():
            value.set("—")

    def _show_selected_detail(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            self._clear_detail()
            return
        row = self.visible_rows_by_item.get(selected[0])
        if row is None:
            self._clear_detail()
            return
        name = str(row.get("name") or "—")
        code = str(row.get("code") or "—")
        industry = str(row.get("industry") or "—")
        self.detail_title.set(f"{name}（{code}）  {industry}")
        self.detail_flow.set(format_order_flow_detail(row))
        self.detail_reason.set(
            f"{format_news_detail(row)}\n入选理由：{row.get('reason') or '—'}"
        )
        self.detail_performance.set(f"业绩复核：{row.get('performance_review') or '—'}")
        self.detail_shareholder.set(format_shareholder_review(row))
        self.detail_risk.set(f"风险提示：{row.get('risk') or '—'}")
        for label, (key, template) in DETAIL_TILES.items():
            number = _finite_number(row.get(key))
            self.detail_tiles[label].set(template.format(number) if number is not None else "—")

    def copy_code(self, _event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        code = str(self.tree.item(selected[0], "values")[1])
        self.clipboard_clear()
        self.clipboard_append(code)
