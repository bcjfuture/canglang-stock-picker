from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import tkinter as tk
from tkinter import messagebox, ttk

import customtkinter as ctk

from instock_app.settings import STRATEGIES
from .backtest_page import BacktestPage
from .consensus_page import ConsensusPage
from .dashboard_page import DashboardPage
from .data_page import DataPage
from .picker_page import PickerPage
from .sector_forecast_page import SectorForecastPage
from .system_page import SystemPage
from .theme import COLORS, FONTS, icon_image, status_pill


WINDOW_SIZE = (1440, 900)
MINIMUM_WINDOW_SIZE = (1180, 720)
TITLE_BAR_ALLOWANCE = 60


def fit_window_size(area_width: int, area_height: int, scaling: float) -> tuple[int, int]:
    """Largest default size (unscaled units) that fits the usable screen area."""
    width = min(WINDOW_SIZE[0], int(area_width / scaling))
    height = min(WINDOW_SIZE[1], int((area_height - TITLE_BAR_ALLOWANCE) / scaling))
    return width, height


def usable_screen_area(root) -> tuple[int, int, int, int]:
    """Left, top, width and height of the screen excluding the taskbar."""
    try:
        rect = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0):  # SPI_GETWORKAREA
            return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
    except (AttributeError, OSError):
        pass
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


class MainWindow:
    PAGE_NAMES = (
        "首页", "数据中心", "选股中心", "三日共识", "五日共识", "板块预测", "回测中心", "系统管理"
    )
    PAGE_META = {
        "首页": ("工作台", "今日概览与快捷操作", "\uE80F"),
        "数据中心": ("数据中心", "更新行情、资金流与策略数据", "\uE895"),
        "选股中心": ("选股中心", "运行策略并查看入选股票", "\uE71C"),
        "三日共识": ("三日共识", "查看最近3个交易日的三策略重复入选排行（不含全市场精选）", "\uE161"),
        "五日共识": ("五日共识", "查看最近5个交易日的三策略重复入选排行（不含全市场精选）", "\uE8C0"),
        "板块预测": ("板块预测", "预测未来5个交易日的行业与概念热点", "\uECA5"),
        "回测中心": ("回测中心", "验证前1、3、5个交易日的选股表现", "\uE9D2"),
        "系统管理": ("系统管理", "管理本地服务与日志", "\uE713"),
    }

    def __init__(self, root: tk.Tk, controller) -> None:
        self.root = root
        self.controller = controller
        self.active_page = ""
        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        self.nav_icons: dict[str, tuple] = {}

        ctk.set_appearance_mode("dark")
        self.root.title("沧浪之水智能选股")
        self._fit_to_screen()
        self.root.configure(bg=COLORS["app_bg"])
        icon = controller.paths.install_root / "app.ico"
        if icon.exists():
            # CustomTkinter sets its own icon shortly after start-up.
            self.root.after(400, lambda: self.root.iconbitmap(str(icon)))
        self._configure_ttk_style()

        self.shell = ctk.CTkFrame(root, fg_color=COLORS["app_bg"], corner_radius=0)
        self.shell.pack(fill="both", expand=True)
        self.shell.grid_rowconfigure(0, weight=1)
        self.shell.grid_columnconfigure(1, weight=1)

        self.sidebar = ctk.CTkFrame(
            self.shell, width=220, corner_radius=0, fg_color=COLORS["sidebar"]
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(len(self.PAGE_NAMES) + 1, weight=1)
        self._build_sidebar()

        self.workspace = ctk.CTkFrame(
            self.shell, fg_color=COLORS["content_bg"], corner_radius=0
        )
        self.workspace.grid(row=0, column=1, sticky="nsew")
        self.workspace.grid_rowconfigure(1, weight=1)
        self.workspace.grid_columnconfigure(0, weight=1)

        self.data_date_text = tk.StringVar(value="数据 —")
        self.service_text = tk.StringVar(value="服务检查中")
        self.task_text = tk.StringVar(value="任务空闲")
        self.page_title_text = tk.StringVar(value="工作台")
        self.page_subtitle_text = tk.StringVar(value="今日概览与快捷操作")
        self._build_topbar()

        self.container = ctk.CTkFrame(
            self.workspace, fg_color=COLORS["content_bg"], corner_radius=0
        )
        self.container.grid(row=1, column=0, sticky="nsew")
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.pages = {
            "首页": DashboardPage(self.container, controller),
            "数据中心": DataPage(self.container, controller),
            "选股中心": PickerPage(self.container, controller),
            "三日共识": ConsensusPage(self.container, controller, day_limit=3),
            "五日共识": ConsensusPage(self.container, controller, day_limit=5),
            "板块预测": SectorForecastPage(self.container, controller),
            "回测中心": BacktestPage(self.container, controller),
            "系统管理": SystemPage(self.container, controller),
        }
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

        self.pages["首页"].on_navigate = self.show_page
        self.root.report_callback_exception = self._report_callback_exception
        self.pages["选股中心"].load_latest()
        self.show_page("首页")
        controller.set_observers(self.enqueue_event, self.enqueue_done)
        self._schedule_data_date_refresh()

    def _fit_to_screen(self) -> None:
        left, top, area_width, area_height = usable_screen_area(self.root)
        scaling = self.root._get_window_scaling()
        width, height = fit_window_size(area_width, area_height, scaling)
        self.root.minsize(min(MINIMUM_WINDOW_SIZE[0], width), min(MINIMUM_WINDOW_SIZE[1], height))
        x = left + max(0, (area_width - round(width * scaling)) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{top}")

    @property
    def page_names(self) -> tuple[str, ...]:
        return tuple(self.pages)

    def _build_sidebar(self) -> None:
        brand = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=22, pady=(28, 30))
        ctk.CTkLabel(
            brand, text="沧浪之水", text_color=COLORS["text"], font=FONTS["brand"]
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand,
            text="智能选股工作台",
            text_color=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(anchor="w", pady=(2, 0))

        for row, name in enumerate(self.PAGE_NAMES, start=1):
            _title, _subtitle, glyph = self.PAGE_META[name]
            self.nav_icons[name] = (
                icon_image(glyph, COLORS["text_muted"]),
                icon_image(glyph, COLORS["accent"]),
            )
            button = ctk.CTkButton(
                self.sidebar,
                text=name,
                image=self.nav_icons[name][0],
                compound="left",
                border_spacing=10,
                command=lambda value=name: self.show_page(value),
                fg_color="transparent",
                hover_color=COLORS["card_hover"],
                text_color=COLORS["text_muted"],
                anchor="w",
                corner_radius=10,
                height=46,
                font=FONTS["body_bold"],
            )
            button.grid(row=row, column=0, sticky="ew", padx=14, pady=4)
            self.nav_buttons[name] = button

        footer = ctk.CTkFrame(
            self.sidebar,
            fg_color=COLORS["card_alt"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"],
        )
        footer.grid(
            row=len(self.PAGE_NAMES) + 2,
            column=0,
            sticky="sew",
            padx=14,
            pady=18,
        )
        ctk.CTkLabel(
            footer, text="●  本地安全运行", text_color=COLORS["success"], font=FONTS["small"]
        ).pack(anchor="w", padx=14, pady=(12, 2))
        ctk.CTkLabel(
            footer, text="沧浪之水 Desktop 2.0", text_color=COLORS["text_muted"], font=FONTS["small"]
        ).pack(anchor="w", padx=14, pady=(0, 12))

    def _build_topbar(self) -> None:
        topbar = ctk.CTkFrame(
            self.workspace, height=74, fg_color=COLORS["content_bg"], corner_radius=0
        )
        topbar.grid(row=0, column=0, sticky="ew", padx=26, pady=(8, 0))
        topbar.grid_columnconfigure(1, weight=1)
        titles = ctk.CTkFrame(topbar, fg_color="transparent")
        titles.grid(row=0, column=0, sticky="w", pady=8)
        ctk.CTkLabel(
            titles,
            textvariable=self.page_title_text,
            text_color=COLORS["text"],
            font=FONTS["section_title"],
        ).pack(anchor="w")
        ctk.CTkLabel(
            titles,
            textvariable=self.page_subtitle_text,
            text_color=COLORS["text_muted"],
            font=FONTS["small"],
        ).pack(anchor="w")

        pills = ctk.CTkFrame(topbar, fg_color="transparent")
        pills.grid(row=0, column=2, sticky="e", pady=12)
        status_pill(pills, text="", textvariable=self.data_date_text).pack(side="left", padx=4)
        self.service_pill = status_pill(
            pills, text="", textvariable=self.service_text, tone="primary"
        )
        self.service_pill.pack(side="left", padx=4)
        self.task_pill = status_pill(pills, text="", textvariable=self.task_text, tone="success")
        self.task_pill.pack(side="left", padx=4)

    def _configure_ttk_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        # One dark look for every table and scrollbar in the app.
        style.layout("Treeview", [("Treeview.treearea", {"sticky": "nsew"})])
        style.configure(
            "Treeview.Heading", bordercolor=COLORS["border"],
            lightcolor=COLORS["card_alt"], darkcolor=COLORS["card_alt"],
        )
        for orientation in ("Vertical", "Horizontal"):
            style.configure(
                f"{orientation}.TScrollbar",
                background=COLORS["card_hover"], troughcolor=COLORS["card"],
                bordercolor=COLORS["card"], lightcolor=COLORS["card_hover"],
                darkcolor=COLORS["card_hover"], arrowcolor=COLORS["text_muted"],
                relief="flat", borderwidth=0,
            )
            style.map(
                f"{orientation}.TScrollbar",
                background=[("active", COLORS["primary"])],
                arrowcolor=[("active", COLORS["text"])],
            )
        style.configure("Content.TFrame", background=COLORS["content_bg"])
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure(
            "PageTitle.TLabel",
            background=COLORS["content_bg"],
            foreground=COLORS["text"],
            font=FONTS["page_title"],
        )
        style.configure(
            "SubTitle.TLabel",
            background=COLORS["content_bg"],
            foreground=COLORS["text_muted"],
            font=FONTS["body"],
        )
        style.configure(
            "CardTitle.TLabel",
            background=COLORS["card"],
            foreground=COLORS["text"],
            font=FONTS["card_title"],
        )
        style.configure(
            "Body.TLabel",
            background=COLORS["content_bg"],
            foreground=COLORS["text_muted"],
            font=FONTS["body"],
        )

    def show_page(self, name: str) -> None:
        if name not in self.pages:
            raise KeyError(name)
        self.active_page = name
        title, subtitle, _icon = self.PAGE_META[name]
        self.page_title_text.set(title)
        self.page_subtitle_text.set(subtitle)
        for page_name, button in self.nav_buttons.items():
            active = page_name == name
            button.configure(
                fg_color=COLORS["nav_active"] if active else "transparent",
                text_color=COLORS["text"] if active else COLORS["text_muted"],
            )
            icons = getattr(self, "nav_icons", {}).get(page_name)
            if icons:
                button.configure(image=icons[1 if active else 0])
        page = self.pages[name]
        display_widget = getattr(page, "_parent_frame", page)
        display_widget.tkraise()
        if name == "首页":
            self.pages[name].refresh_status()
        if name == "系统管理":
            self.pages[name].refresh()
        if name in {"三日共识", "五日共识"}:
            self.pages[name].refresh()
        if name == "板块预测":
            self.pages[name].refresh()
        self._refresh_data_date()
        self._refresh_service_status()

    def _refresh_data_date(self) -> None:
        date_value = self.controller.latest_data_date()
        if date_value:
            self.data_date_text.set(f"数据 {date_value}")

    def _schedule_data_date_refresh(self) -> None:
        self._refresh_data_date()
        self.root.after(60_000, self._schedule_data_date_refresh)

    def _refresh_service_status(self) -> None:
        status = self.controller.runtime.status()
        if status.db_running:
            text = "数据库在线"
            if status.web_running:
                text = "全部服务在线"
            self.service_pill.configure(fg_color="#123B2B", text_color=COLORS["success"])
        else:
            text = "数据库离线"
            self.service_pill.configure(fg_color="#3D1725", text_color=COLORS["danger"])
        self.service_text.set(text)

    def _refresh_dashboard(self) -> None:
        dashboard = getattr(self, "pages", {}).get("首页")
        if hasattr(dashboard, "refresh_status"):
            dashboard.refresh_status()

    def _report_callback_exception(self, exc_type, exc, traceback) -> None:
        """Button handlers raise for expected refusals; show them instead of dropping them."""
        if issubclass(exc_type, RuntimeError):
            messagebox.showinfo("暂时无法执行", str(exc))
            return
        logging.error("界面操作失败", exc_info=(exc_type, exc, traceback))
        messagebox.showerror("操作失败", str(exc))

    def enqueue_event(self, event) -> None:
        self.root.after(0, lambda: self.handle_event(event))

    def enqueue_done(self, result) -> None:
        self.root.after(0, lambda: self.handle_done(result))

    def handle_event(self, event) -> None:
        self.task_text.set(f"{event.progress}%  {event.message}")
        self.task_pill.configure(fg_color="#123761", text_color=COLORS["accent"])
        self.pages["数据中心"].handle_event(event)
        self.pages["板块预测"].handle_event(event)
        if (
            event.kind == "result"
            and event.payload.get("strategy") in STRATEGIES
            and isinstance(event.payload.get("rows"), list)
        ):
            picker = self.pages["选股中心"]
            if not event.payload.get("automatic"):
                picker.display_result(event.payload)
                self.show_page("选股中心")
            elif event.payload.get("strategy") == picker.selected_key():
                # Automatic runs refresh what is on screen but never move the user.
                picker.display_result(event.payload)
        if event.kind == "result" and event.payload.get("recent_backtest"):
            result = self.controller.load_backtest()
            if result:
                self.pages["回测中心"].display_result(result)
                if not event.payload.get("automatic"):
                    self.show_page("回测中心")
        if event.kind == "result" and event.payload.get("sector_forecast"):
            result = self.controller.load_sector_forecast()
            if result:
                self.pages["板块预测"].display_result(result)
                if not bool(event.payload.get("automatic")):
                    self.show_page("板块预测")
        if event.kind == "error":
            self.task_pill.configure(fg_color="#3D1725", text_color=COLORS["danger"])
            if not (
                event.payload.get("scope") == "sector_forecast"
                and event.payload.get("automatic")
            ):
                messagebox.showerror("任务失败", event.message)

    def handle_done(self, result) -> None:
        completed_kind = result.payload.get("_completed_kind", self.controller.state.last_completed_kind)
        automatic = result.payload.get("_automatic", False)
        if completed_kind == "sector_forecast":
            self.pages["板块预测"].handle_done(result)
        if result.returncode == 0:
            self.task_text.set("任务已完成")
            self.task_pill.configure(fg_color="#123B2B", text_color=COLORS["success"])
            if completed_kind == "picker":
                self.pages["三日共识"].refresh()
                self.pages["五日共识"].refresh()
            if completed_kind == "update":
                self._refresh_data_date()
                self._refresh_dashboard()
            if completed_kind == "backtest":
                backtest = self.controller.load_backtest()
                if backtest:
                    self.pages["回测中心"].display_result(backtest)
                    if not automatic:
                        self.show_page("回测中心")
            if completed_kind == "sector_forecast":
                forecast = self.controller.load_sector_forecast()
                if forecast:
                    self.pages["板块预测"].display_result(forecast)
        else:
            self.task_text.set(f"任务失败 · 退出码 {result.returncode}")
            self.task_pill.configure(fg_color="#3D1725", text_color=COLORS["danger"])
        if result.payload.get("_pipeline_pending"):
            self.task_text.set("自动流程运行中 · 正在执行后续任务")
            self.task_pill.configure(fg_color="#123761", text_color=COLORS["accent"])
        if result.payload.get("_cancelled"):
            self.task_text.set("自动流程已取消")
