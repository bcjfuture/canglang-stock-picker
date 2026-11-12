from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


COLUMNS = [
    ("排名", "rank", 8),
    ("代码", "code", 12),
    ("名称", "name", 14),
    ("行业", "industry", 18),
    ("现价", "price", 11),
    ("涨跌幅%", "change_pct", 12),
    ("建议持有天数", "advice_holding_days", 14),
    ("止盈价", "take_profit_price", 12),
    ("止盈率", "take_profit_pct", 12),
    ("止损价", "stop_loss_price", 12),
    ("止损率", "stop_loss_pct", 12),
    ("策略版本", "policy_version", 20),
    ("建议日期", "advice_as_of", 14),
    ("建议状态", "advice_status", 20),
    ("综合评分", "score", 12),
    ("大单净额(超大单+大单,万元)", "large_order_flow", 25),
    ("散户净额(中单+小单,万元)", "retail_flow", 25),
    ("1日资金净流向(万元)", "fund_flow_1d", 20),
    ("3日资金净流向(万元)", "fund_flow_3d", 20),
    ("5日资金净流向(万元)", "fund_flow_5d", 20),
    ("股东结构评分", "shareholder_score", 16),
    ("机构持股比例%", "institutional_holding_ratio", 18),
    ("机构持股家数", "institutional_holder_count", 16),
    ("股东户数变化%", "shareholder_count_change", 18),
    ("十大股东持股比例%", "top10_holding_ratio", 22),
    ("近3月机构调研数", "institutional_survey_count", 20),
    ("股东结构说明", "shareholder_review", 55),
    ("入选理由", "reason", 65),
    ("业绩复核", "performance_review", 38),
    ("风险提示", "risk", 35),
    ("概念", "concept", 40),
]


def _write_rows(sheet, rows: list[dict]) -> None:
    sheet.append([title for title, _, _ in COLUMNS])
    header_fill = PatternFill("solid", fgColor="17365D")
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in rows:
        sheet.append([row.get(key, "") for _, key, _ in COLUMNS])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for index, (_, _, width) in enumerate(COLUMNS, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=cell.column >= 9)
        row[1].number_format = "@"
        row[4].number_format = "0.00"
        for index, (_, key, _) in enumerate(COLUMNS):
            if key in {
                "price",
                "take_profit_price",
                "stop_loss_price",
                "large_order_flow",
                "retail_flow",
                "fund_flow_1d",
                "fund_flow_3d",
                "fund_flow_5d",
                "institutional_holding_ratio",
                "shareholder_count_change",
                "top10_holding_ratio",
            }:
                row[index].number_format = "0.00"
            elif key in {"take_profit_pct", "stop_loss_pct"}:
                row[index].number_format = "0.0%"
            elif key == "score":
                row[index].number_format = "0.0"
            elif key == "shareholder_score":
                row[index].number_format = "+0.0;-0.0;0.0"


def export_workbook(result: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    top = workbook.active
    top.title = "策略Top"
    _write_rows(top, list(result.get("rows", [])))

    all_sheet = workbook.create_sheet("全部候选")
    _write_rows(all_sheet, list(result.get("rows", [])))

    note = workbook.create_sheet("说明")
    note.append(["项目", "内容"])
    note.append(["策略", result.get("title", "")])
    note.append(["交易日期", result.get("trade_date", "")])
    note.append(["生成时间", result.get("generated_at", "")])
    note.append(["声明", "结果仅作为研究候选池，不构成投资建议，不保证未来收益。"]) 
    note.column_dimensions["A"].width = 18
    note.column_dimensions["B"].width = 80
    workbook.save(output_path)
    workbook.close()
    return output_path
