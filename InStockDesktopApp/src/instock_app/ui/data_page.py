from __future__ import annotations

import customtkinter as ctk

from .theme import COLORS, FONTS, card, primary_button, secondary_button, status_pill


class DataPage(ctk.CTkFrame):
    def __init__(self, parent, controller) -> None:
        super().__init__(parent, fg_color=COLORS["content_bg"], corner_radius=0)
        self.controller = controller
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        summary = card(self)
        summary.grid(row=0, column=0, sticky="ew", padx=26, pady=(14, 16))
        summary.grid_columnconfigure(0, weight=1)
        copy = ctk.CTkFrame(summary, fg_color="transparent")
        copy.grid(row=0, column=0, sticky="w", padx=20, pady=18)
        ctk.CTkLabel(copy, text="每日数据流水线", text_color=COLORS["text"], font=FONTS["section_title"]).pack(anchor="w")
        ctk.CTkLabel(copy, text="数据更新 → 健康检查 → 4套选股 → 回测 → 板块预测", text_color=COLORS["text_muted"], font=FONTS["body"]).pack(anchor="w", pady=(4, 0))
        status_pill(summary, "6 个更新阶段", tone="primary").grid(row=0, column=1, padx=8)
        status_pill(summary, "失败自动重试", tone="success").grid(row=0, column=2, padx=(8, 20))
        status_pill(summary, "自动选股及回测", tone="neutral").grid(row=0, column=3, padx=(0, 20))

        controls = card(self)
        controls.grid(row=1, column=0, sticky="ew", padx=26, pady=(0, 16))
        controls.grid_columnconfigure(1, weight=1)
        buttons = ctk.CTkFrame(controls, fg_color="transparent")
        buttons.grid(row=0, column=0, sticky="w", padx=18, pady=16)
        self.run_button = primary_button(buttons, "开始更新数据", controller.start_update, width=132)
        self.run_button.pack(side="left")
        secondary_button(buttons, "取消后续任务", controller.cancel, width=120).pack(side="left", padx=10)

        progress_area = ctk.CTkFrame(controls, fg_color="transparent")
        progress_area.grid(row=0, column=1, sticky="ew", padx=18, pady=14)
        progress_area.grid_columnconfigure(0, weight=1)
        self.step_label = ctk.CTkLabel(progress_area, text="等待开始", text_color=COLORS["text_muted"], font=FONTS["small"])
        self.step_label.grid(row=0, column=0, sticky="w")
        self.progress_label = ctk.CTkLabel(progress_area, text="0%", text_color=COLORS["accent"], font=FONTS["body_bold"])
        self.progress_label.grid(row=0, column=1, sticky="e")
        self.progress = ctk.CTkProgressBar(
            progress_area,
            height=10,
            corner_radius=5,
            progress_color=COLORS["primary"],
            fg_color=COLORS["card_hover"],
        )
        self.progress.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.progress.set(0)

        terminal = card(self, fg_color=COLORS["terminal"])
        terminal.grid(row=2, column=0, sticky="nsew", padx=26, pady=(0, 26))
        terminal.grid_rowconfigure(1, weight=1)
        terminal.grid_columnconfigure(0, weight=1)
        header = ctk.CTkFrame(terminal, fg_color="transparent", height=42)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(8, 0))
        ctk.CTkLabel(header, text="●  ●  ●", text_color="#64748B", font=("Segoe UI", 11)).pack(side="left")
        ctk.CTkLabel(header, text="UPDATE LOG", text_color=COLORS["text_muted"], font=("Consolas", 10, "bold")).pack(side="right")
        self.log = ctk.CTkTextbox(
            terminal,
            fg_color=COLORS["terminal"],
            text_color="#C9D7EA",
            font=FONTS["mono"],
            corner_radius=8,
            border_width=0,
            wrap="word",
        )
        self.log.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.log.insert("end", "[READY] 数据更新任务等待启动\n")

    def handle_event(self, event) -> None:
        value = max(0, min(100, int(event.progress)))
        self.progress.set(value / 100)
        self.progress_label.configure(text=f"{value}%")
        self.step_label.configure(text=event.message)
        marker = "OK" if event.kind == "result" else event.kind.upper()
        self.log.insert("end", f"[{marker}] {event.message}\n")
        self.log.see("end")
