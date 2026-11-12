# 沧浪之水智能选股

Windows 本地智能选股工作台的源码快照，整理日期：2026-11-12。

包含全市场精选、科技进攻、反弹趋势、郑希景气精选，以及数据更新、三日/五日共识、板块预测和近期选股回测等模块。界面使用 Python、CustomTkinter，数据服务基于 InStock 和 MariaDB。

## 目录

- `InStockDesktopApp/src/`：桌面界面、任务控制和运行环境管理。
- `InStockDesktopApp/engine/`：选股策略、模型和策略配置。
- `InStockDesktopApp/workers/`：数据更新、选股、回测、板块预测及 Excel 导出。
- `InStockDesktopApp/tests/`：现有测试。
- `InStockDesktopApp/*.spec`：PyInstaller 打包配置。
- `InStock/`：本地使用的 InStock 后端、网页资源和原有文档。

## 运行环境

当前本机使用 Windows、Python 3.11、MariaDB 12.3。根目录 `requirements.txt` 汇总后端依赖与本机桌面端主要依赖版本；这些依赖没有在全新环境中重新安装验证。

项目现有代码多处使用 `C:\InStockLocal` 路径。使用现有目录结构运行时，应将两个源码目录放在 `C:\InStockLocal` 下，并准备 `venv`、MariaDB 与本地数据目录。本机实际文件位于 D 盘，并通过 C 盘目录联接访问。其他安装位置需要检查 `src/instock_app/settings.py` 及源码中的固定路径。

```powershell
python -m pip install -r requirements.txt
python InStockDesktopApp/src/instock_desktop_entry.py
```

以上命令只展示依赖安装和入口；MariaDB 初始化、行情下载及本机路径配置需要另行完成。仓库不包含数据库或历史行情，因此新环境不会直接出现截图中的选股与回测结果。

## 上传范围与配置

上传桌面端和后端源码、策略配置、模型、测试、打包配置及后端静态资源。未上传安装后的 EXE、虚拟环境、数据库、行情缓存、运行结果、日志、旧备份及自动运行的 GitHub 工作流。

交易账户配置 `InStock/instock/config/trade_client.json` 未上传。上传副本中的行情接口 Token 改为读取 `INSTOCK_CHIP_RACE_TOKEN`；网页 Cookie 密钥改为读取 `INSTOCK_COOKIE_SECRET`，未设置时使用进程启动时生成的随机值。本机安装文件未改动。

原有源码仍包含本地数据库的默认连接配置。部署到其他环境前需要核对数据库配置；这些默认配置不代表可用于公网部署的完整配置。

## 上游来源

`InStock/` 保留上游文档与 Apache-2.0 许可证，详见 [原项目说明](InStock/README.md) 和 [原许可证](InStock/LICENSE)。本仓库保存本机已有修改，并非上游项目的官方发行版。

选股和回测结果用于研究，历史表现不代表未来收益。
