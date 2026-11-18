"""自包含 HTML 报告生成（无外部依赖，可直接内网分发）。

内容：评估指标卡片、告警明细表、证据贡献条形图（纯 SVG）、
园区平面示意图（真值位置 vs 估计位置 + 不确定半径）、时间轴。
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone, timedelta

from fbs_detector.common.layout import PARK_BY_ID
from fbs_detector.server.detector import Alert, DetectionResult

_CN_TZ = timezone(timedelta(hours=8))


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=_CN_TZ).strftime("%m-%d %H:%M")


def _bar_chart(evidence: list[dict], width: int = 520) -> str:
    if not evidence:
        return "<p>无证据</p>"
    rows = []
    mx = max(e["contribution"] for e in evidence) or 1.0
    for e in evidence:
        w = int(e["contribution"] / mx * (width - 180))
        color = "#c0392b" if e["contribution"] >= 1.5 else "#e67e22" if e["contribution"] >= 0.8 else "#7f8c8d"
        rows.append(
            f'<div class="bar-row"><span class="bar-label">{html.escape(e["code"])}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{w}px;background:{color}"></span></span>'
            f'<span class="bar-val">{e["contribution"]:.2f}</span></div>'
            f'<div class="bar-detail">{html.escape(e["detail"])}</div>')
    return '<div class="bars">' + "".join(rows) + "</div>"


def _park_map(alerts: list[Alert], gt_rogues: list[dict], park_id: str, size: int = 360) -> str:
    park = PARK_BY_ID[park_id]
    scale = size / (park.radius_m * 2.4)
    cx = cy = size / 2

    def xy(lat: float, lon: float) -> tuple[float, float]:
        from fbs_detector.common.geo import project_xy
        x, y = project_xy(lat, lon, park.center_lat, park.center_lon)
        return cx + x * scale, cy - y * scale

    parts = [f'<svg viewBox="0 0 {size} {size}" class="map">',
             f'<circle cx="{cx}" cy="{cy}" r="{park.radius_m * scale}" fill="#eef3f8" stroke="#b0bec5"/>',
             f'<text x="{cx}" y="20" text-anchor="middle" class="map-title">{html.escape(park.name)} ({park_id})</text>']
    # 真值
    for r in gt_rogues:
        if r["park_id"] != park_id:
            continue
        x, y = xy(r["lat"], r["lon"])
        parts.append(f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="none" stroke="#16a085" stroke-width="2"/>'
                     f'<line x1="{x-5:.1f}" y1="{y-5:.1f}" x2="{x+5:.1f}" y2="{y+5:.1f}" stroke="#16a085" stroke-width="2"/>'
                     f'<line x1="{x+5:.1f}" y1="{y-5:.1f}" x2="{x-5:.1f}" y2="{y+5:.1f}" stroke="#16a085" stroke-width="2"/>'
                     f'<text x="{x+12:.1f}" y="{y+4:.1f}" class="map-label">真值 {html.escape(r["rogue_id"][:2])}</text></g>')
    # 估计
    for a in alerts:
        if a.park_id != park_id:
            continue
        x, y = xy(a.lat, a.lon)
        r = a.radius_m * scale
        sev_color = {"critical": "#8e1c1c", "high": "#c0392b",
                     "medium": "#e67e22", "low": "#f1c40f"}[a.severity]
        parts.append(f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{sev_color}" fill-opacity="0.12" stroke="{sev_color}" stroke-dasharray="4 3"/>'
                     f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{sev_color}"/>'
                     f'<text x="{x+10:.1f}" y="{y-8:.1f}" class="map-label">{html.escape(a.event_id)} {a.confidence:.2f}</text></g>')
    parts.append("</svg>")
    return "".join(parts)


def render_html(result: DetectionResult, metrics: dict, gt_rogues: list[dict],
                n_observations: int, n_envelopes: int, out_path: str) -> None:
    parks = sorted({a.park_id for a in result.alerts} |
                   {r["park_id"] for r in gt_rogues})
    level_badge = {"critical": "badge-critical", "high": "badge-high",
                   "medium": "badge-medium", "low": "badge-low"}

    cards = f"""
    <div class="cards">
      <div class="card"><div class="num">{metrics['tp']}/{metrics['n_rogues_truth']}</div><div class="cap">命中伪基站</div></div>
      <div class="card"><div class="num">{metrics['fp']}</div><div class="cap">误报事件</div></div>
      <div class="card"><div class="num">{metrics['fn']}</div><div class="cap">漏报</div></div>
      <div class="card"><div class="num">{metrics['precision']:.2f}</div><div class="cap">精确率</div></div>
      <div class="card"><div class="num">{metrics['recall']:.2f}</div><div class="cap">召回率</div></div>
      <div class="card"><div class="num">{metrics['f1']:.2f}</div><div class="cap">F1</div></div>
      <div class="card"><div class="num">{metrics.get('median_spatial_error_m') or '-'}m</div><div class="cap">定位误差中位数</div></div>
      <div class="card"><div class="num">{metrics.get('median_detection_delay_min') or '-'}min</div><div class="cap">发现时延中位数</div></div>
    </div>"""

    alert_rows = []
    for a in result.alerts:
        win = "，".join(f"{_fmt_ts(s)}~{_fmt_ts(e)}" for s, e in a.time_windows)
        ev = "<br>".join(f"<b>{e['code']}</b>×{e['count']}（{e['contribution']:.2f}）：{html.escape(e['detail'])}"
                         for e in a.top_evidence(6))
        alert_rows.append(f"""
        <tr>
          <td>{a.event_id}<br><span class="muted">{html.escape(PARK_BY_ID[a.park_id].name)}</span></td>
          <td>{a.rat}</td>
          <td><span class="badge {level_badge[a.severity]}">{a.level} {a.confidence:.2f}</span></td>
          <td>{a.n_devices} 台 / {a.n_observations} 条</td>
          <td>{_fmt_ts(a.first_seen)} ~ {_fmt_ts(a.last_seen)}<br><span class="muted">{win}</span></td>
          <td>{a.lat:.5f}, {a.lon:.5f}<br><span class="muted">半径 {a.radius_m:.0f}m</span></td>
          <td class="ev-cell">{ev}</td>
        </tr>""")

    suppressed_rows = "".join(
        f'<li><b>{a.event_id}</b>（{html.escape(PARK_BY_ID[a.park_id].name)}）：{html.escape(a.suppress_reason)}</li>'
        for a in result.suppressed) or "<li>无</li>"

    maps = "".join(f'<div class="map-wrap">{_park_map(result.alerts, gt_rogues, p)}</div>' for p in parks)
    charts = "".join(
        f'<h3>{a.event_id} 证据分解（{html.escape(PARK_BY_ID[a.park_id].name)}，{a.rat}）</h3>{_bar_chart(a.evidence)}'
        for a in result.alerts[:6])

    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>伪基站发现系统 · 检测报告</title>
<style>
:root {{ --bg:#f7f9fb; --fg:#1c2833; --accent:#2471a3; --line:#d5dbdb; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#17202a; --fg:#eaecee; --accent:#5dade2; --line:#34495e; }}
}}
body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif; padding:24px; max-width:1180px; margin-inline:auto; }}
h1 {{ font-size:22px; }} h2 {{ font-size:17px; margin-top:32px; border-left:4px solid var(--accent); padding-left:8px; }}
h3 {{ font-size:14px; margin:18px 0 6px; }}
.muted {{ color:#7f8c8d; font-size:12px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin:16px 0; }}
.card {{ background:var(--bg); border:1px solid var(--line); border-radius:10px; padding:12px; text-align:center; }}
.card .num {{ font-size:22px; font-weight:700; color:var(--accent); }}
.card .cap {{ font-size:12px; color:#7f8c8d; }}
table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
th,td {{ border:1px solid var(--line); padding:8px; vertical-align:top; text-align:left; }}
th {{ background:color-mix(in srgb, var(--accent) 12%, transparent); }}
.badge {{ padding:2px 8px; border-radius:10px; color:#fff; font-size:12px; white-space:nowrap; }}
.badge-critical {{ background:#8e1c1c; }} .badge-high {{ background:#c0392b; }}
.badge-medium {{ background:#e67e22; }} .badge-low {{ background:#b7950b; }}
.ev-cell {{ font-size:11.5px; min-width:320px; }}
.bars {{ margin:6px 0 14px; }}
.bar-row {{ display:flex; align-items:center; gap:8px; margin-top:6px; }}
.bar-label {{ width:170px; font-family:monospace; font-size:11.5px; }}
.bar-track {{ flex:1; background:var(--line); height:14px; border-radius:7px; max-width:340px; }}
.bar-fill {{ display:block; height:14px; border-radius:7px; }}
.bar-val {{ width:42px; text-align:right; font-family:monospace; }}
.bar-detail {{ font-size:11px; color:#7f8c8d; margin:1px 0 0 178px; }}
.map-wrap {{ display:inline-block; margin:8px 16px 8px 0; }}
.map {{ width:360px; height:360px; border:1px solid var(--line); border-radius:10px; background:#fff; }}
.map-title {{ font-size:12px; fill:#566573; }} .map-label {{ font-size:10.5px; fill:var(--fg); }}
.legend span {{ display:inline-block; margin-right:18px; font-size:12px; }}
.legend i {{ display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:4px; }}
ul {{ margin:6px 0; }}
</style></head><body>
<h1>伪基站发现系统 · 众包检测报告</h1>
<p class="muted">生成时间 {_fmt_ts(datetime.now(tz=_CN_TZ).timestamp())}（仿真时区 UTC+8）｜
观测总量 {n_observations} 条 / {n_envelopes} 个签名批次 ｜ 候选发射源 {result.candidate_count} 个 ｜
图例：<span class="legend"><i style="background:#16a085"></i>地面真值
<i style="background:#c0392b"></i>估计位置（虚线圆为不确定半径）</span></p>

<h2>一、总体指标</h2>
{cards}
<p>观察名单 {len(result.watchlist)} 个候选（证据不足未告警，持续跟踪）；
白名单压制事件 {len(result.suppressed)} 起：</p>
<ul>{suppressed_rows}</ul>

<h2>二、告警明细与判定依据</h2>
<table><thead><tr><th>事件</th><th>制式</th><th>置信度</th><th>佐证规模</th><th>活动时间窗</th><th>估计位置</th><th>关键证据（LLR 贡献）</th></tr></thead>
<tbody>{''.join(alert_rows) or '<tr><td colspan=7>无告警</td></tr>'}</tbody></table>

<h2>三、空间定位（真值对照）</h2>
{maps}

<h2>四、证据融合分解</h2>
{charts}

<h2>五、方法与局限</h2>
<ul>
<li>置信度 = sigmoid(先验对数几率 + Σ 信号权重×√触发次数 + 多设备佐证 LLR)，先验按稀有事件设千分之一，压制先入为主。</li>
<li>同类信号按 √n 次线性累加，防止单条规则靠样本量刷分；身份漂移的候选通过 DBSCAN 空间聚类合并为同一事件。</li>
<li>计划内工程（COW/扩容/退服）以运营商注册表窗口一票压制，但保留审计记录。</li>
<li>局限：纯被动采集无法检测仅被动监听的沉默伪基站；室内 GPS 漂移会放大定位半径；真机落地需处理 iOS 字段限制与安卓权限差异。</li>
</ul>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
