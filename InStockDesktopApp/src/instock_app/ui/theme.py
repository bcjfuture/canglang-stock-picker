from __future__ import annotations

from pathlib import Path

import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "app_bg": "#000000",
    "content_bg": "#000000",
    "sidebar": "#0A0A0A",
    "card": "#12151A",
    "card_hover": "#1C2129",
    "card_alt": "#0C0E12",
    "primary": "#2F80FF",
    "primary_hover": "#1E6DE0",
    "accent": "#22D3EE",
    "success": "#22C55E",
    "danger": "#F43F5E",
    "warning": "#F59E0B",
    "text": "#F8FAFC",
    "text_muted": "#94A3B8",
    "border": "#262C36",
    "nav_active": "#183B67",
    "terminal": "#000000",
    "row_alt": "#0E1014",
}

FONTS = {
    "brand": ("KaiTi", 30),
    "page_title": ("Microsoft YaHei UI", 20, "bold"),
    "section_title": ("Microsoft YaHei UI", 18, "bold"),
    "card_title": ("Microsoft YaHei UI", 14, "bold"),
    "metric": ("Microsoft YaHei UI", 25, "bold"),
    "body": ("Microsoft YaHei UI", 13),
    "body_bold": ("Microsoft YaHei UI", 13, "bold"),
    "small": ("Microsoft YaHei UI", 11),
    "mono": ("Consolas", 11),
}


ICON_FONT = Path(r"C:\Windows\Fonts\segmdl2.ttf")


def icon_image(glyph: str, color: str, size: int = 18) -> ctk.CTkImage | None:
    """Line icon from the Windows icon font, at a fixed size so labels align."""
    if not ICON_FONT.exists():
        return None
    scale = 4
    canvas = size * scale
    image = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(ICON_FONT), int(canvas * 0.9))
    left, top, right, bottom = draw.textbbox((0, 0), glyph, font=font)
    draw.text(((canvas - right - left) / 2, (canvas - bottom - top) / 2), glyph, font=font, fill=color)
    return ctk.CTkImage(light_image=image, dark_image=image, size=(size, size))


def card(parent, **kwargs) -> ctk.CTkFrame:
    options = {
        "fg_color": COLORS["card"],
        "corner_radius": 14,
        "border_width": 1,
        "border_color": COLORS["border"],
    }
    options.update(kwargs)
    return ctk.CTkFrame(parent, **options)


def status_pill(parent, text: str, tone: str = "neutral", **kwargs) -> ctk.CTkLabel:
    tone_colors = {
        "neutral": (COLORS["card_hover"], COLORS["text_muted"]),
        "primary": ("#123761", COLORS["accent"]),
        "success": ("#123B2B", COLORS["success"]),
        "warning": ("#3B2D12", COLORS["warning"]),
        "danger": ("#3D1725", COLORS["danger"]),
    }
    background, foreground = tone_colors.get(tone, tone_colors["neutral"])
    options = {
        "text": text,
        "fg_color": background,
        "text_color": foreground,
        "corner_radius": 12,
        "font": FONTS["small"],
        "height": 26,
        "padx": 10,
    }
    options.update(kwargs)
    return ctk.CTkLabel(parent, **options)


def primary_button(parent, text: str, command, **kwargs) -> ctk.CTkButton:
    options = {
        "text": text,
        "command": command,
        "fg_color": COLORS["primary"],
        "hover_color": COLORS["primary_hover"],
        "text_color": "#FFFFFF",
        "corner_radius": 10,
        "height": 40,
        "font": FONTS["body_bold"],
    }
    options.update(kwargs)
    return ctk.CTkButton(parent, **options)


def secondary_button(parent, text: str, command, **kwargs) -> ctk.CTkButton:
    options = {
        "text": text,
        "command": command,
        "fg_color": COLORS["card_hover"],
        "hover_color": "#1D3554",
        "border_width": 1,
        "border_color": COLORS["border"],
        "text_color": COLORS["text"],
        "corner_radius": 10,
        "height": 40,
        "font": FONTS["body_bold"],
    }
    options.update(kwargs)
    return ctk.CTkButton(parent, **options)
