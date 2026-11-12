from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from .theme import COLORS, FONTS, card, primary_button, secondary_button, status_pill


class SystemPage(ctk.CTkFrame):
    def __init__(self, parent, controller) -> None:
        super().__init__(parent, fg_color=COLORS["content_bg"], corner_radius=0)
        self.controller = controller
        self.service_cards: dict[str, ctk.CTkFrame] = {}
        self.service_values = {
            "database": tk.StringVar(value="检查中"),
            "web": tk.StringVar(value="检查中"),
        }
        self.grid_columnconfigure((0, 1), weight=1, uniform="services")

        intro = card(self, fg_color="#101722", border_color="#1F3350")
        intro.grid(row=0, column=0, columnspan=2, sticky="ew", padx=26, pady=(14, 18))
        ctk.CTkLabel(intro, text="本地服务管理", text_color=COLORS["text"], font=FONTS["section_title"]).pack(anchor="w", padx=20, pady=(18, 3))
        ctk.CTkLabel(intro, text="只管理沧浪之水使用的数据库和 Web 服务，不影响其他程序。", text_color="#B7C7DD", font=FONTS["body"]).pack(anchor="w", padx=20, pady=(0, 18))

        self._build_service_card(0, "database", "本地数据库", "MySQL · 端口 3306", "▣")
        self._build_service_card(1, "web", "沧浪之水 Web", "本地服务 · 端口 9988", "◎")

        actions = card(self)
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", padx=26, pady=(0, 18))
        actions.grid_columnconfigure(0, weight=1)
        copy = ctk.CTkFrame(actions, fg_color="transparent")
        copy.grid(row=0, column=0, sticky="w", padx=18, pady=16)
        ctk.CTkLabel(copy, text="服务操作", text_color=COLORS["text"], font=FONTS["card_title"]).pack(anchor="w")
        ctk.CTkLabel(copy, text="启动数据库、停止沧浪之水或查看运行日志", text_color=COLORS["text_muted"], font=FONTS["small"]).pack(anchor="w", pady=(3, 0))
        buttons = ctk.CTkFrame(actions, fg_color="transparent")
        buttons.grid(row=0, column=1, sticky="e", padx=18, pady=16)
        primary_button(buttons, "启动数据库", self.start, width=120).pack(side="left", padx=5)
        secondary_button(buttons, "打开日志", self.open_logs, width=108).pack(side="left", padx=5)
        self.stop_button = ctk.CTkButton(
            buttons,
            text="停止沧浪之水",
            command=self.stop,
            width=120,
            height=40,
            corner_radius=10,
            fg_color="#3D1725",
            hover_color="#5A1B30",
            text_color=COLORS["danger"],
            border_width=1,
            border_color="#6B243A",
            font=FONTS["body_bold"],
        )
        self.stop_button.pack(side="left", padx=5)

        self.message = tk.StringVar(value="系统状态已就绪")
        footer = card(self, fg_color=COLORS["card_alt"])
        footer.grid(row=3, column=0, columnspan=2, sticky="ew", padx=26)
        ctk.CTkLabel(footer, textvariable=self.message, text_color=COLORS["text_muted"], font=FONTS["body"]).pack(anchor="w", padx=18, pady=14)
        self.refresh()

    def _build_service_card(self, column: int, key: str, title: str, subtitle: str, icon: str) -> None:
        item = card(self)
        item.grid(
            row=1,
            column=column,
            sticky="nsew",
            padx=(26 if column == 0 else 8, 26 if column == 1 else 8),
            pady=(0, 18),
        )
        self.service_cards[key] = item
        top = ctk.CTkFrame(item, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=(18, 8))
        ctk.CTkLabel(top, text=icon, text_color=COLORS["accent"], font=("Segoe UI Symbol", 28, "bold")).pack(side="left")
        pill = status_pill(top, "", textvariable=self.service_values[key])
        pill.pack(side="right")
        setattr(self, f"{key}_pill", pill)
        ctk.CTkLabel(item, text=title, text_color=COLORS["text"], font=FONTS["section_title"]).pack(anchor="w", padx=20, pady=(6, 0))
        ctk.CTkLabel(item, text=subtitle, text_color=COLORS["text_muted"], font=FONTS["body"]).pack(anchor="w", padx=20, pady=(4, 20))

    def refresh(self) -> None:
        state = self.controller.runtime.status()
        for key, running in (("database", state.db_running), ("web", state.web_running)):
            self.service_values[key].set("运行中" if running else "未运行")
            pill = getattr(self, f"{key}_pill")
            if running:
                pill.configure(fg_color="#123B2B", text_color=COLORS["success"])
            else:
                pill.configure(fg_color="#3D1725", text_color=COLORS["danger"])

    def start(self) -> None:
        try:
            pid = self.controller.start_services()
            self.message.set(f"数据库已启动 · PID {pid}")
        except Exception as exc:
            messagebox.showerror("启动失败", str(exc))
        self.refresh()

    def stop(self) -> None:
        stopped = self.controller.stop_services()
        summary = ", ".join(map(str, stopped)) if stopped else "无"
        self.message.set(f"已停止沧浪之水进程：{summary}")
        self.refresh()

    def open_logs(self) -> None:
        self.controller.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(self.controller.paths.logs_dir)
