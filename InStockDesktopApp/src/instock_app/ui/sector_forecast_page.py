from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import customtkinter as ctk

from .theme import COLORS, FONTS, card, primary_button, secondary_button, status_pill


STATUS_TONES = {
    "success": ("#123B2B", COLORS["success"]),
    "warning": ("#3B2D12", COLORS["warning"]),
    "danger": ("#3D1725", COLORS["danger"]),
}


def _joined(values: object) -> str:
    if not isinstance(values, list):
        return "—"
    text = "；".join(str(value) for value in values if value)
    return text or "—"


def format_sector_row(row: dict) -> tuple[object, ...]:
    previous = row.get("previous_realized_excess")
    previous_text = "—" if previous is None else f"{float(previous):+.2%}"
    return (
        int(row.get("rank", 0) or 0),
        str(row.get("sector_name") or "—"),
        str(row.get("strength") or "—"),
        str(int(row.get("confidence", 0) or 0)),
        str(row.get("trend") or "—"),
        str(row.get("fund_state") or "—"),
        str(row.get("breadth") or "—"),
        _joined(row.get("reasons")),
        _joined(row.get("risks")),
        previous_text,
    )


def forecast_status_summary(result: dict) -> tuple[str, str]:
    statuses = {
        str(universe.get("status"))
        for universe in result.get("universes", {}).values()
    }
    if "ready" in statuses:
        return "预测可用", "success"
    if "head_ready" in statuses:
        return "前5已验证｜其余参考", "success"
    if "forced_fallback" in statuses:
        return "参考榜单｜模型验证未通过", "warning"
    return "本期数据不可用", "danger"


def universe_status_summary(universe: dict) -> tuple[str, str]:
    status = str(universe.get("status"))
    if status == "ready":
        return "已验证预测", "success"
    if status == "head_ready":
        return "前5已验证｜第6-10名参考", "success"
    if status == "forced_fallback":
        return "参考榜单｜模型验证未通过", "warning"
    return "本期数据不可用", "danger"


class SectorForecastPage(ctk.CTkFrame):
    UNIVERSES = {"industry": "行业板块", "concept": "概念板块"}
    COLUMNS = (
        ("rank", "排名", 54, "center"),
        ("name", "板块", 110, "w"),
        ("strength", "强度", 62, "center"),
        ("confidence", "可信度", 72, "center"),
        ("trend", "当前趋势", 140, "w"),
        ("fund", "资金状态", 140, "w"),
        ("breadth", "板块广度", 140, "w"),
        ("reasons", "主要依据", 310, "w"),
        ("risks", "风险提示", 270, "w"),
        ("previous", "上期实际", 86, "center"),
    )

    def __init__(self, parent, controller) -> None:
        super().__init__(parent, fg_color=COLORS["content_bg"], corner_radius=0)
        self.controller = controller
        self.result: dict = {}
        self.trees: dict[str, ttk.Treeview] = {}
        self.empty_labels: dict[str, ctk.CTkLabel] = {}
        self.universe_status_labels: dict[str, ctk.CTkLabel] = {}
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.as_of_text = tk.StringVar(value="数据日期 —")
        self.model_text = tk.StringVar(value="模型 —")
        self.status_text = tk.StringVar(value="等待首次预测")
        self._build_summary()
        self._build_tabs()
        self._configure_tree_style()

    def _build_summary(self) -> None:
        summary = card(self)
        summary.grid(row=0, column=0, sticky="ew", padx=26, pady=(14, 14))
        summary.grid_columnconfigure(0, weight=1)
        copy = ctk.CTkFrame(summary, fg_color="transparent")
        copy.grid(row=0, column=0, sticky="w", padx=20, pady=16)
        ctk.CTkLabel(
            copy,
            text="未来一周热门板块",
            text_color=COLORS["text"],
            font=FONTS["section_title"],
        ).pack(anchor="w")
        ctk.CTkLabel(
            copy,
            text="未来5日相对沪深300 · 20日仅作风控",
            text_color=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(anchor="w", pady=(4, 0))

        pills = ctk.CTkFrame(summary, fg_color="transparent")
        pills.grid(row=0, column=1, sticky="e", padx=8, pady=16)
        status_pill(pills, "", textvariable=self.as_of_text).pack(side="left", padx=4)
        status_pill(pills, "", textvariable=self.model_text, tone="primary").pack(
            side="left", padx=4
        )
        self.status_pill = status_pill(
            pills, "", textvariable=self.status_text, tone="warning"
        )
        self.status_pill.pack(side="left", padx=4)

        actions = ctk.CTkFrame(summary, fg_color="transparent")
        actions.grid(row=0, column=2, sticky="e", padx=(8, 20), pady=16)
        self.run_button = primary_button(
            actions, "重新预测", self._run, width=108
        )
        self.run_button.pack(side="left", padx=4)
        self.refresh_button = secondary_button(
            actions, "刷新历史并预测", self._refresh_history, width=138
        )
        self.refresh_button.pack(side="left", padx=4)

    def _build_tabs(self) -> None:
        body = card(self)
        body.grid(row=1, column=0, sticky="nsew", padx=26, pady=(0, 26))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        self.tabs = ctk.CTkTabview(
            body,
            fg_color=COLORS["card"],
            segmented_button_fg_color=COLORS["card_alt"],
            segmented_button_selected_color=COLORS["primary"],
            segmented_button_selected_hover_color=COLORS["primary_hover"],
            segmented_button_unselected_color=COLORS["card_alt"],
            segmented_button_unselected_hover_color=COLORS["card_hover"],
            text_color=COLORS["text"],
        )
        self.tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        for sector_type, title in self.UNIVERSES.items():
            tab = self.tabs.add(title)
            tab.grid_rowconfigure(1, weight=1)
            tab.grid_columnconfigure(0, weight=1)
            universe_status = ctk.CTkLabel(
                tab,
                text="等待首次预测",
                text_color=COLORS["text_muted"],
                fg_color=COLORS["card_alt"],
                font=FONTS["small"],
                corner_radius=8,
                anchor="w",
                padx=12,
                pady=6,
            )
            universe_status.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 0))
            self.universe_status_labels[sector_type] = universe_status
            frame = tk.Frame(tab, bg=COLORS["card"])
            frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
            frame.grid_rowconfigure(0, weight=1)
            frame.grid_columnconfigure(0, weight=1)
            columns = [item[0] for item in self.COLUMNS]
            tree = ttk.Treeview(
                frame,
                columns=columns,
                show="headings",
                style="Sector.Treeview",
                selectmode="browse",
            )
            for key, heading, width, anchor in self.COLUMNS:
                tree.heading(key, text=heading)
                tree.column(key, width=width, minwidth=width, stretch=False, anchor=anchor)
            ybar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            xbar = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
            tree.grid(row=0, column=0, sticky="nsew")
            ybar.grid(row=0, column=1, sticky="ns")
            xbar.grid(row=1, column=0, sticky="ew")
            self.trees[sector_type] = tree
            label = ctk.CTkLabel(
                tab,
                text="等待首次预测",
                text_color=COLORS["text_muted"],
                font=FONTS["body"],
                fg_color=COLORS["card_alt"],
                corner_radius=10,
                padx=18,
                pady=12,
            )
            label.place(relx=0.5, rely=0.5, anchor="center")
            self.empty_labels[sector_type] = label

        self.disclaimer = ctk.CTkLabel(
            body,
            text="预测仅供研究参考，不构成投资建议。",
            text_color=COLORS["text_muted"],
            font=FONTS["small"],
        )
        self.disclaimer.grid(row=1, column=0, sticky="w", padx=18, pady=(0, 12))

    def _configure_tree_style(self) -> None:
        style = ttk.Style(self)
        style.configure(
            "Sector.Treeview",
            background=COLORS["card_alt"],
            fieldbackground=COLORS["card_alt"],
            foreground=COLORS["text"],
            rowheight=46,
            borderwidth=0,
            font=FONTS["small"],
        )
        style.configure(
            "Sector.Treeview.Heading",
            background=COLORS["card_hover"],
            foreground=COLORS["text_muted"],
            font=FONTS["small"],
            relief="flat",
        )
        style.map(
            "Sector.Treeview",
            background=[("selected", COLORS["nav_active"])],
            foreground=[("selected", COLORS["text"])],
        )

    def _run(self) -> None:
        try:
            self.controller.start_sector_forecast()
            self._set_running(True)
        except Exception as exc:
            messagebox.showerror("无法开始预测", str(exc))

    def _refresh_history(self) -> None:
        try:
            self.controller.start_sector_forecast(refresh_history=True)
            self._set_running(True)
        except Exception as exc:
            messagebox.showerror("无法开始预测", str(exc))

    def _set_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)
        self.refresh_button.configure(state=state)
        if running:
            self.status_text.set("预测运行中")
            self.status_pill.configure(fg_color="#123761", text_color=COLORS["accent"])

    def refresh(self) -> None:
        result = self.controller.load_sector_forecast()
        if result:
            self.display_result(result)

    def handle_event(self, event) -> None:
        if (
            self.controller.state.current_kind == "sector_forecast"
            or event.payload.get("sector_forecast")
            or event.payload.get("scope") == "sector_forecast"
        ):
            self._set_running(event.kind not in {"result", "error"})
            self.status_text.set(event.message)
            if event.kind == "result":
                result = self.controller.load_sector_forecast()
                if result:
                    self.display_result(result)

    def handle_done(self, result) -> None:
        self._set_running(False)
        if result.returncode != 0:
            self.status_text.set("预测任务失败，保留上次有效结果")
            self.status_pill.configure(
                fg_color="#3D1725", text_color=COLORS["danger"]
            )

    def display_result(self, result: dict) -> None:
        self.result = result
        self._set_running(False)
        self.as_of_text.set(f"数据 {result.get('as_of', '—')}")
        self.model_text.set(str(result.get("model_version") or "模型 —"))
        status_text, status_tone = forecast_status_summary(result)
        self.status_text.set(status_text)
        tone = STATUS_TONES[status_tone]
        self.status_pill.configure(fg_color=tone[0], text_color=tone[1])

        for sector_type, tree in self.trees.items():
            tree.delete(*tree.get_children())
            universe = result.get("universes", {}).get(sector_type, {})
            universe_status, universe_tone = universe_status_summary(universe)
            status_tone = STATUS_TONES[universe_tone]
            self.universe_status_labels[sector_type].configure(
                text=universe_status,
                fg_color=status_tone[0],
                text_color=status_tone[1],
            )
            rows = list(universe.get("rows", []))
            if rows:
                self.empty_labels[sector_type].place_forget()
                for index, row in enumerate(rows):
                    tag = "alt" if index % 2 else "normal"
                    tree.insert("", "end", values=format_sector_row(row), tags=(tag,))
                tree.tag_configure("normal", background=COLORS["card_alt"])
                tree.tag_configure("alt", background=COLORS["row_alt"])
            else:
                self.empty_labels[sector_type].configure(
                    text=str(universe.get("message") or "本期暂不预测")
                )
                self.empty_labels[sector_type].place(relx=0.5, rely=0.5, anchor="center")
        disclaimer = str(result.get("disclaimer") or "预测仅供研究参考，不构成投资建议。")
        statuses = {
            str(universe.get("status"))
            for universe in result.get("universes", {}).values()
        }
        if "head_ready" in statuses:
            disclaimer = (
                f"{disclaimer}\n"
                "标记为“前5已验证”的板块仅代表前5名通过滚动样本外门槛；第6-10名仍是观察顺序。"
            )
        if "forced_fallback" in statuses:
            disclaimer = (
                f"{disclaimer}\n"
                "当前榜单由3/5/10日短期动量、资金、广度和活跃度因子生成，20日信号仅用于过热风险；模型滚动验证未通过，仅作为观察顺序。"
            )
        self.disclaimer.configure(text=disclaimer)
