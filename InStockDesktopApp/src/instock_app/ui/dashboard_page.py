from __future__ import annotations

from datetime import datetime
import tkinter as tk

import customtkinter as ctk
from instock_app.settings import STRATEGIES as REGISTERED_STRATEGIES

from .theme import COLORS, FONTS, card, primary_button, secondary_button, status_pill


class DashboardPage(ctk.CTkScrollableFrame):
    STRATEGIES = {
        "all_market": (
            "全市场精选",
            "市赚率3.0 × 基本面",
            "PB小于1，基本面过滤加分红修正估值",
            COLORS["primary"],
        ),
        "tech_attack": ("科技进攻", "低波 × 回调", "在科技方向中挑选低波动、未过热的回调标的", COLORS["primary"]),
        "rebound_trend": ("反弹趋势", "回撤 × 企稳", "回撤后企稳并站上10日线，达标才入选", COLORS["primary"]),
        "zhengxi_prosperity": ("郑希景气精选", "景气 × 成长", "跟踪行业景气变化与企业成长质量", COLORS["primary"]),
    }

    def __init__(self, parent, controller) -> None:
        super().__init__(
            parent,
            fg_color=COLORS["content_bg"],
            corner_radius=0,
            scrollbar_button_color=COLORS["card_hover"],
            scrollbar_button_hover_color=COLORS["primary"],
        )
        self.controller = controller
        self.metric_cards: dict[str, ctk.CTkFrame] = {}
        self.metric_values: dict[str, tk.StringVar] = {}
        self.metric_notes: dict[str, tk.StringVar] = {}
        self.metric_note_labels: dict[str, ctk.CTkLabel] = {}
        # Set by the main window; opens a page without starting any task.
        self.on_navigate = None
        self.strategy_cards: dict[str, ctk.CTkFrame] = {}
        self.strategy_status: dict[str, tk.StringVar] = {}
        # Equal card widths: the outer margin lives in spacer columns, not in padx.
        self.grid_columnconfigure((1, 2, 3, 4), weight=1, uniform="dashboard")
        self.grid_columnconfigure((0, 5), minsize=18)

        self._build_hero()
        self._build_metrics()
        self._build_strategies()
        self._build_activity()

    def _build_hero(self) -> None:
        hero = card(self, fg_color="#101722", border_color="#1F3350")
        hero.grid(row=0, column=1, columnspan=4, sticky="ew", padx=8, pady=(14, 16))
        hero.grid_columnconfigure(0, weight=1)
        copy = ctk.CTkFrame(hero, fg_color="transparent")
        copy.grid(row=0, column=0, sticky="w", padx=22, pady=14)
        ctk.CTkLabel(
            copy,
            text=f"{self._greeting()}，欢迎回到选股工作台",
            text_color=COLORS["text"],
            font=FONTS["page_title"],
        ).pack(anchor="w")
        ctk.CTkLabel(
            copy,
            text="更新成功后自动运行4套策略及回测，选股中心可切换查看结果。",
            text_color="#B7C7DD",
            font=FONTS["small"],
        ).pack(anchor="w", pady=(4, 0))

        actions = ctk.CTkFrame(hero, fg_color="transparent")
        actions.grid(row=0, column=1, sticky="e", padx=22, pady=14)
        secondary_button(actions, "查看回测", self._open_backtest, width=112).pack(side="left", padx=6)
        primary_button(actions, "更新全部数据", self.controller.start_update, width=138).pack(side="left", padx=6)

    def _build_metrics(self) -> None:
        definitions = [
            ("data", "数据健康", "检查中", "●", COLORS["text_muted"], "正在核对各表日期"),
            ("database", "数据库", "检查中", "", COLORS["text_muted"], "本机服务状态"),
            ("strategy", "策略套数", f"{len(REGISTERED_STRATEGIES)} 套", "", COLORS["text_muted"], "当前启用"),
            ("benchmark", "回测基准", "沪深300", "", COLORS["text_muted"], "收益对比基准"),
        ]
        for column, (key, title, value, icon, tone, note) in enumerate(definitions):
            item = card(self)
            item.grid(
                row=1,
                column=column + 1,
                sticky="nsew",
                padx=8,
                pady=(0, 18),
            )
            self.metric_cards[key] = item
            var = tk.StringVar(value=value)
            self.metric_values[key] = var
            header = ctk.CTkFrame(item, fg_color="transparent")
            header.pack(fill="x", padx=18, pady=(16, 4))
            ctk.CTkLabel(header, text=title, text_color=COLORS["text_muted"], font=FONTS["small"]).pack(side="left")
            dot = ctk.CTkLabel(header, text=icon, text_color=tone, font=("Segoe UI Symbol", 13))
            dot.pack(side="right")
            if key == "data":
                self.data_dot = dot
            ctk.CTkLabel(item, textvariable=var, text_color=COLORS["text"], font=FONTS["metric"]).pack(anchor="w", padx=18, pady=(2, 2))
            note_var = tk.StringVar(value=note)
            self.metric_notes[key] = note_var
            note_label = ctk.CTkLabel(
                item, textvariable=note_var, text_color=tone, font=FONTS["small"],
                justify="left", wraplength=220,
            )
            note_label.pack(anchor="w", padx=18, pady=(0, 16))
            self.metric_note_labels[key] = note_label

    def _build_strategies(self) -> None:
        ctk.CTkLabel(
            self, text="核心选股策略", text_color=COLORS["text"], font=FONTS["section_title"]
        ).grid(row=2, column=1, columnspan=4, sticky="w", padx=8, pady=(2, 10))

        for column, (key, details) in enumerate(self.STRATEGIES.items()):
            title, tag, description, tone = details
            item = card(self)
            item.grid(
                row=3,
                column=column + 1,
                sticky="nsew",
                padx=8,
                pady=(0, 18),
            )
            self.strategy_cards[key] = item
            accent = ctk.CTkFrame(item, height=2, fg_color=tone, corner_radius=1)
            accent.pack(fill="x", padx=18, pady=(16, 14))
            top = ctk.CTkFrame(item, fg_color="transparent")
            top.pack(fill="x", padx=18)
            ctk.CTkLabel(top, text=title, text_color=COLORS["text"], font=FONTS["card_title"]).pack(side="left")
            status_pill(top, tag, tone="primary").pack(side="right")
            ctk.CTkLabel(
                item,
                text=description,
                text_color=COLORS["text_muted"],
                font=FONTS["body"],
                justify="left",
                wraplength=220,
                anchor="nw",
                height=44,
            ).pack(fill="x", padx=18, pady=(14, 8))
            status_var = tk.StringVar(value="今日结果读取中")
            self.strategy_status[key] = status_var
            ctk.CTkLabel(
                item, textvariable=status_var, text_color=COLORS["accent"], font=FONTS["small"],
            ).pack(anchor="w", padx=18, pady=(0, 14))
            secondary_button(
                item,
                "运行策略  →",
                lambda strategy=key: self.controller.run_picker(strategy),
                height=36,
            ).pack(side="bottom", fill="x", padx=18, pady=(0, 18))

    def _build_activity(self) -> None:
        panel = card(self)
        panel.grid(row=4, column=1, columnspan=4, sticky="ew", padx=8, pady=(0, 26))
        panel.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(panel, text="今日任务动态", text_color=COLORS["text"], font=FONTS["card_title"]).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 4))
        self.activity_text = tk.StringVar(value="系统已就绪，建议先检查数据日期。")
        ctk.CTkLabel(panel, textvariable=self.activity_text, text_color=COLORS["text_muted"], font=FONTS["body"]).grid(row=1, column=0, columnspan=2, sticky="w", padx=18, pady=(2, 16))
        status_pill(panel, "本地运行", tone="success").grid(row=0, column=2, rowspan=2, padx=18)

    @staticmethod
    def _greeting() -> str:
        hour = datetime.now().hour
        if hour < 11:
            return "早上好"
        if hour < 18:
            return "下午好"
        return "晚上好"

    def _refresh_strategy_status(self) -> None:
        for key, variable in self.strategy_status.items():
            try:
                result = self.controller.load_picker_result(key) or {}
            except Exception:
                result = {}
            trade_date = str(result.get("trade_date") or "")[5:10]
            count = len(result.get("rows") or [])
            if not trade_date:
                variable.set("还没有结果")
            else:
                variable.set(f"{trade_date} · " + (f"入选 {count} 只" if count else "本次空仓"))

    def _open_backtest(self) -> None:
        if self.on_navigate:
            self.on_navigate("回测中心")

    def refresh_status(self) -> None:
        status = self.controller.runtime.status()
        health = self.controller.data_health(status.db_running)
        self.metric_values["database"].set("运行中" if status.db_running else "未运行")
        self.metric_values["data"].set(health["label"])
        self.metric_notes["data"].set(health["detail"])
        self.metric_note_labels["data"].configure(text_color=COLORS[health["tone"]])
        self.data_dot.configure(text_color=COLORS[health["tone"]])
        self._refresh_strategy_status()
        if not status.db_running:
            activity = "数据库尚未运行，请先到系统管理启动服务。"
        elif health["tone"] == "success":
            activity = "数据正常，可以开始选股。"
        else:
            activity = f"{health['detail']}。建议先更新数据再选股。"
        self.activity_text.set(activity)
