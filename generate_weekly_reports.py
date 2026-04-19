from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


VISITOR_FILE_PATTERN = "*Visitor_viewer*.csv"
SEGMENT_FILES = {
    "countries": "*Country*.csv",
    "devices": "*Device*.csv",
    "operating_systems": "*Operating System*.csv",
    "referrals": "*Referral*.csv",
}


@dataclass(frozen=True)
class LoadNotes:
    visitor_file: Path
    fallback_visitor_rows: int
    dropped_empty_rows: int
    base_year: int


@dataclass(frozen=True)
class ReportFile:
    week_number: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    path: Path


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Read CSV exports that may contain BOMs or local spreadsheet encodings."""
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    raise ValueError(f"Could not read {path} with supported encodings") from last_error


def find_one_file(data_dir: Path, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No CSV file matching {pattern!r} in {data_dir}")
    return matches[0]


def parse_date_labels(labels: Iterable[object], base_year: int) -> pd.Series:
    parsed_dates: list[pd.Timestamp | pd.NaT] = []
    year = base_year
    previous_month: int | None = None

    for label in labels:
        label_text = str(label).strip()
        if not label_text or label_text.lower() == "nan":
            parsed_dates.append(pd.NaT)
            continue

        try:
            partial = datetime.strptime(f"{label_text} 2000", "%b %d %Y")
        except ValueError:
            parsed_dates.append(pd.NaT)
            continue

        if previous_month is not None and partial.month < previous_month:
            year += 1
        previous_month = partial.month

        parsed_dates.append(pd.Timestamp(year=year, month=partial.month, day=partial.day))

    return pd.Series(parsed_dates)


def load_daily_metrics(data_dir: Path, year: int | None) -> tuple[pd.DataFrame, LoadNotes]:
    visitor_file = find_one_file(data_dir, VISITOR_FILE_PATTERN)
    base_year = year or datetime.fromtimestamp(visitor_file.stat().st_mtime).year

    raw = read_csv_with_fallback(visitor_file)
    if len(raw.columns) < 4:
        raise ValueError(
            f"{visitor_file} needs at least four columns: date, helper/visitor, visitors, page views"
        )

    raw = raw.iloc[:, :4].copy()
    raw.columns = ["date_label", "helper_value", "visitors_named", "page_views"]

    helper_value = pd.to_numeric(raw["helper_value"], errors="coerce")
    visitors_named = pd.to_numeric(raw["visitors_named"], errors="coerce")
    page_views = pd.to_numeric(raw["page_views"], errors="coerce")

    # Some rows in the current export have the visitor count in the second
    # column while the named visitor column is blank. Only use that fallback
    # when it behaves like a count and the row has page-view data.
    fallback_mask = (
        visitors_named.isna()
        & helper_value.notna()
        & page_views.notna()
        & (helper_value <= page_views)
    )
    visitors = visitors_named.copy()
    visitors.loc[fallback_mask] = helper_value.loc[fallback_mask]

    date = parse_date_labels(raw["date_label"], base_year=base_year)
    daily = pd.DataFrame(
        {
            "date": date,
            "visitors": visitors,
            "page_views": page_views,
        }
    )

    empty_mask = daily[["visitors", "page_views"]].isna().all(axis=1)
    dropped_empty_rows = int(empty_mask.sum())

    daily = daily.loc[~empty_mask].dropna(subset=["date"]).copy()
    daily["visitors"] = daily["visitors"].fillna(0).round().astype(int)
    daily["page_views"] = daily["page_views"].fillna(0).round().astype(int)

    daily = daily.groupby("date", as_index=False).agg(
        visitors=("visitors", "sum"),
        page_views=("page_views", "sum"),
    )

    if daily.empty:
        raise ValueError(f"No usable daily visitor/page-view rows found in {visitor_file}")

    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = (
        daily.set_index("date")
        .reindex(full_range, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    notes = LoadNotes(
        visitor_file=visitor_file,
        fallback_visitor_rows=int(fallback_mask.sum()),
        dropped_empty_rows=dropped_empty_rows,
        base_year=base_year,
    )
    return daily, notes


def load_segments(data_dir: Path) -> dict[str, pd.DataFrame]:
    segments: dict[str, pd.DataFrame] = {}

    for name, pattern in SEGMENT_FILES.items():
        try:
            path = find_one_file(data_dir, pattern)
        except FileNotFoundError:
            continue

        raw = read_csv_with_fallback(path)
        if len(raw.columns) < 3:
            continue

        df = raw.iloc[:, :3].copy()
        df.columns = ["segment", "visitors", "total"]
        df["segment"] = df["segment"].astype(str).str.strip()
        df["visitors"] = pd.to_numeric(df["visitors"], errors="coerce").fillna(0).astype(int)
        df["total"] = pd.to_numeric(df["total"], errors="coerce").fillna(0).astype(int)
        df = df[df["segment"].ne("") & df["segment"].str.lower().ne("nan")]
        df = df.sort_values(["total", "visitors"], ascending=False).reset_index(drop=True)
        segments[name] = df

    if "referrals" in segments:
        segments["channels"] = build_channel_summary(segments["referrals"])

    return segments


def build_channel_summary(referrals: pd.DataFrame) -> pd.DataFrame:
    def classify(referrer: str) -> str:
        value = referrer.lower()
        if "facebook" in value:
            return "Facebook"
        if "messenger" in value:
            return "Messenger"
        if "instagram" in value:
            return "Instagram"
        if "google" in value or "yahoo" in value or "bing" in value:
            return "Search"
        if "payos" in value:
            return "Payment flow"
        return "Other referral"

    rows = referrals.copy()
    rows["segment"] = rows["segment"].map(classify)
    return (
        rows.groupby("segment", as_index=False)
        .agg(visitors=("visitors", "sum"), total=("total", "sum"))
        .sort_values(["total", "visitors"], ascending=False)
        .reset_index(drop=True)
    )


def weekly_slices(daily: pd.DataFrame) -> list[tuple[int, pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    start = daily["date"].min().normalize()
    final_day = daily["date"].max().normalize()
    weeks = []
    week_number = 1
    cursor = start

    while cursor <= final_day:
        end = min(cursor + pd.Timedelta(days=6), final_day)
        week = daily[(daily["date"] >= cursor) & (daily["date"] <= end)].copy()
        weeks.append((week_number, cursor, end, week))
        week_number += 1
        cursor = end + pd.Timedelta(days=1)

    return weeks


def summarize_period(data: pd.DataFrame) -> dict[str, object]:
    visitors = int(data["visitors"].sum())
    page_views = int(data["page_views"].sum())
    active_days = int((data["visitors"] > 0).sum())
    avg_daily_visitors = visitors / len(data) if len(data) else 0
    avg_daily_page_views = page_views / len(data) if len(data) else 0
    views_per_visitor = page_views / visitors if visitors else 0

    if data.empty:
        best_visitor_day = None
        best_page_view_day = None
    else:
        best_visitor_day = data.loc[data["visitors"].idxmax()]
        best_page_view_day = data.loc[data["page_views"].idxmax()]

    return {
        "visitors": visitors,
        "page_views": page_views,
        "views_per_visitor": views_per_visitor,
        "active_days": active_days,
        "days": len(data),
        "avg_daily_visitors": avg_daily_visitors,
        "avg_daily_page_views": avg_daily_page_views,
        "best_visitor_day": best_visitor_day,
        "best_page_view_day": best_page_view_day,
    }


def format_int(value: object) -> str:
    return f"{int(value):,}"


def format_float(value: object, digits: int = 2) -> str:
    return f"{float(value):,.{digits}f}"


def format_percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def format_date(date: pd.Timestamp) -> str:
    return date.strftime("%b %-d, %Y") if not is_windows() else date.strftime("%b %#d, %Y")


def format_date_short(date: pd.Timestamp) -> str:
    return date.strftime("%b %-d") if not is_windows() else date.strftime("%b %#d")


def is_windows() -> bool:
    return re.match(r"^[A-Za-z]:\\", str(Path.cwd())) is not None


def pct_delta(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return (current - previous) / previous


def format_delta(current: float, previous: float, *, suffix: str = "") -> str:
    absolute = current - previous
    percent = pct_delta(current, previous)
    if abs(absolute) < 0.005:
        if percent is None or abs(percent) < 0.005:
            return "No change vs previous"
        return f"No material change ({format_percent(percent)}) vs previous"
    else:
        absolute_text = format_float(absolute, 2).rstrip("0").rstrip(".")
        sign = "+" if absolute > 0 else ""
    if percent is None:
        return f"{sign}{absolute_text}{suffix} vs previous"
    return (
        f"{sign}{absolute_text}{suffix} "
        f"({format_percent(percent)}) vs previous"
    )


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(value: object) -> str:
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    output = [
        "| " + " | ".join(clean(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(output)


def top_segment_line(segments: dict[str, pd.DataFrame], name: str, label: str) -> str | None:
    df = segments.get(name)
    if df is None or df.empty:
        return None

    total_visitors = df["visitors"].sum()
    leader = df.iloc[0]
    share = leader["visitors"] / total_visitors if total_visitors else 0
    return f"{label}: {leader['segment']} with {format_percent(share)} of known visitors"


def segment_table(df: pd.DataFrame, limit: int = 5) -> str:
    total_visitors = df["visitors"].sum()
    total_views = df["total"].sum()
    rows: list[list[object]] = []

    for _, row in df.head(limit).iterrows():
        visitor_share = row["visitors"] / total_visitors if total_visitors else 0
        view_share = row["total"] / total_views if total_views else 0
        rows.append(
            [
                row["segment"],
                format_int(row["visitors"]),
                format_percent(visitor_share),
                format_int(row["total"]),
                format_percent(view_share),
            ]
        )

    return markdown_table(["Segment", "Visitors", "Visitor share", "Total", "Total share"], rows)


def make_week_signals(
    week: pd.DataFrame,
    summary: dict[str, object],
    previous_summary: dict[str, object] | None,
) -> list[str]:
    signals: list[str] = []
    visitors = int(summary["visitors"])
    page_views = int(summary["page_views"])
    views_per_visitor = float(summary["views_per_visitor"])

    if previous_summary:
        previous_visitors = int(previous_summary["visitors"])
        previous_page_views = int(previous_summary["page_views"])
        visitor_change = pct_delta(visitors, previous_visitors)
        page_view_change = pct_delta(page_views, previous_page_views)
        if visitor_change is not None:
            if visitor_change >= 0.2:
                signals.append("Traffic grew strongly from the previous week.")
            elif visitor_change <= -0.2:
                signals.append("Traffic dropped materially from the previous week.")
            else:
                signals.append("Traffic stayed broadly stable compared with the previous week.")
        if page_view_change is not None and page_view_change > (visitor_change or 0) + 0.15:
            signals.append("Page views grew faster than visitors, which points to better browsing depth.")
    else:
        signals.append("This is the first reporting week, so it sets the baseline for later comparisons.")

    if views_per_visitor >= 1.6:
        signals.append("Engagement was healthy, with users viewing more than one page on average.")
    elif visitors > 0 and views_per_visitor <= 1.2:
        signals.append("Engagement was shallow, so the site may need clearer next steps or internal links.")

    if not week.empty and visitors > 0:
        average = float(summary["avg_daily_visitors"])
        best_day = week.loc[week["visitors"].idxmax()]
        if average and best_day["visitors"] >= max(5, average * 2):
            signals.append(
                f"{format_date_short(best_day['date'])} was a clear traffic spike "
                f"with {format_int(best_day['visitors'])} visitors."
            )

        weekend = week[week["date"].dt.weekday >= 5]["visitors"].sum()
        weekend_share = weekend / visitors if visitors else 0
        if weekend_share >= 0.4:
            signals.append("Weekend traffic carried a large share of the week.")
        elif weekend_share <= 0.15 and len(week) >= 6:
            signals.append("Most traffic happened on weekdays, which is useful for campaign timing.")

    return signals


def make_recommendations(
    summary: dict[str, object],
    previous_summary: dict[str, object] | None,
    segments: dict[str, pd.DataFrame],
) -> list[str]:
    recommendations: list[str] = []
    visitors = int(summary["visitors"])
    views_per_visitor = float(summary["views_per_visitor"])

    if previous_summary:
        visitor_change = pct_delta(visitors, int(previous_summary["visitors"]))
        if visitor_change is not None and visitor_change <= -0.2:
            recommendations.append(
                "Audit what changed in acquisition this week: post cadence, ad spend, campaign links, and ranking pages."
            )
        elif visitor_change is not None and visitor_change >= 0.2:
            recommendations.append(
                "Identify the posts, products, or campaigns active on the strongest day and reuse that playbook."
            )
    else:
        recommendations.append(
            "Use this week as the baseline for weekly traffic, engagement, and campaign comparisons."
        )

    if visitors > 0 and views_per_visitor <= 1.2:
        recommendations.append(
            "Improve browsing depth with clearer product links, related content, and stronger calls to action."
        )
    elif views_per_visitor >= 1.6:
        recommendations.append(
            "Preserve the paths that encourage multi-page browsing, then test conversion prompts on those pages."
        )

    countries = segments.get("countries")
    if countries is not None and not countries.empty:
        total_country_visitors = countries["visitors"].sum()
        leader = countries.iloc[0]
        leader_share = leader["visitors"] / total_country_visitors if total_country_visitors else 0
        if leader_share >= 0.75:
            recommendations.append(
                f"Prioritize localization, shipping/payment clarity, and social proof for {leader['segment']} visitors."
            )

    devices = segments.get("devices")
    if devices is not None and not devices.empty:
        mobile = devices[devices["segment"].str.lower().eq("mobile")]
        if not mobile.empty and mobile.iloc[0]["visitors"] >= devices["visitors"].sum() * 0.5:
            recommendations.append(
                "Treat mobile UX as the default experience: speed, checkout friction, and above-the-fold product clarity."
            )

    channels = segments.get("channels")
    if channels is not None and not channels.empty:
        total_channel_visitors = channels["visitors"].sum()
        top_channel = channels.iloc[0]
        top_share = top_channel["visitors"] / total_channel_visitors if total_channel_visitors else 0
        if top_share >= 0.35:
            recommendations.append(
                f"Keep campaign tracking tight for {top_channel['segment']} because it dominates known referral traffic."
            )

    return dedupe_keep_order(recommendations)[:5]


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


def make_daily_table(week: pd.DataFrame, visitors: int) -> str:
    rows: list[list[object]] = []
    for _, row in week.iterrows():
        views_per_visitor = row["page_views"] / row["visitors"] if row["visitors"] else 0
        visitor_share = row["visitors"] / visitors if visitors else 0
        rows.append(
            [
                format_date_short(row["date"]),
                format_int(row["visitors"]),
                format_int(row["page_views"]),
                format_float(views_per_visitor),
                format_percent(visitor_share),
            ]
        )

    return markdown_table(["Date", "Visitors", "Page views", "Views/visitor", "Visitor share"], rows)


def write_week_report(
    output_dir: Path,
    week_number: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    week: pd.DataFrame,
    previous_summary: dict[str, object] | None,
    segments: dict[str, pd.DataFrame],
) -> ReportFile:
    summary = summarize_period(week)
    visitors = int(summary["visitors"])
    page_views = int(summary["page_views"])
    best_visitor_day = summary["best_visitor_day"]
    best_page_view_day = summary["best_page_view_day"]
    signals = make_week_signals(week, summary, previous_summary)
    recommendations = make_recommendations(summary, previous_summary, segments)

    filename = f"week_{week_number:02d}_{start.date()}_to_{end.date()}.md"
    path = output_dir / filename

    lines = [
        f"# Weekly Analytics Report - Week {week_number}",
        "",
        f"Period: {format_date(start)} to {format_date(end)}",
        "",
        "## Executive summary",
        "",
        f"- Visitors: {format_int(visitors)}",
        f"- Page views: {format_int(page_views)}",
        f"- Views per visitor: {format_float(summary['views_per_visitor'])}",
        f"- Active days: {format_int(summary['active_days'])} of {format_int(summary['days'])}",
        f"- Average daily visitors: {format_float(summary['avg_daily_visitors'])}",
        f"- Average daily page views: {format_float(summary['avg_daily_page_views'])}",
    ]

    if best_visitor_day is not None:
        lines.append(
            f"- Best visitor day: {format_date_short(best_visitor_day['date'])} "
            f"with {format_int(best_visitor_day['visitors'])} visitors"
        )
    if best_page_view_day is not None:
        lines.append(
            f"- Best page-view day: {format_date_short(best_page_view_day['date'])} "
            f"with {format_int(best_page_view_day['page_views'])} page views"
        )

    lines.extend(["", "## Week-over-week movement", ""])
    if previous_summary is None:
        lines.append("No previous week is available. This report is the baseline.")
    else:
        lines.extend(
            [
                f"- Visitors: {format_delta(visitors, int(previous_summary['visitors']))}",
                f"- Page views: {format_delta(page_views, int(previous_summary['page_views']))}",
                (
                    "- Views per visitor: "
                    f"{format_delta(float(summary['views_per_visitor']), float(previous_summary['views_per_visitor']))}"
                ),
            ]
        )

    lines.extend(["", "## Daily performance", "", make_daily_table(week, visitors)])

    lines.extend(["", "## Business signals", ""])
    lines.extend(f"- {signal}" for signal in signals)

    lines.extend(["", "## Recommended actions", ""])
    lines.extend(f"- {recommendation}" for recommendation in recommendations)

    context_lines = [
        top_segment_line(segments, "countries", "Primary country"),
        top_segment_line(segments, "devices", "Primary device"),
        top_segment_line(segments, "operating_systems", "Primary operating system"),
        top_segment_line(segments, "channels", "Primary known acquisition channel"),
    ]
    context_lines = [line for line in context_lines if line]
    if context_lines:
        lines.extend(["", "## Overall audience context", ""])
        lines.append(
            "These segment breakdowns come from aggregate files in `total-data`, so they are period context rather than week-specific attribution."
        )
        lines.extend(f"- {line}" for line in context_lines)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ReportFile(week_number=week_number, start_date=start, end_date=end, path=path)


def write_index_report(
    output_dir: Path,
    daily: pd.DataFrame,
    weeks: list[tuple[int, pd.Timestamp, pd.Timestamp, pd.DataFrame]],
    report_files: list[ReportFile],
    segments: dict[str, pd.DataFrame],
    notes: LoadNotes,
) -> Path:
    overall = summarize_period(daily)
    weekly_rows: list[list[object]] = []
    previous_summary: dict[str, object] | None = None

    for (week_number, start, end, week), report_file in zip(weeks, report_files):
        summary = summarize_period(week)
        best_day = summary["best_visitor_day"]
        if previous_summary is None:
            visitor_change = "Baseline"
            page_view_change = "Baseline"
        else:
            visitor_change = format_delta(int(summary["visitors"]), int(previous_summary["visitors"]))
            page_view_change = format_delta(int(summary["page_views"]), int(previous_summary["page_views"]))

        weekly_rows.append(
            [
                f"[Week {week_number}]({report_file.path.name})",
                f"{format_date_short(start)} to {format_date_short(end)}",
                format_int(summary["visitors"]),
                format_int(summary["page_views"]),
                format_float(summary["views_per_visitor"]),
                (
                    f"{format_date_short(best_day['date'])} ({format_int(best_day['visitors'])})"
                    if best_day is not None
                    else "n/a"
                ),
                visitor_change,
                page_view_change,
            ]
        )
        previous_summary = summary

    strongest_week = max(
        (summarize_period(week) | {"week_number": week_number, "start": start, "end": end}
         for week_number, start, end, week in weeks),
        key=lambda item: int(item["visitors"]),
    )

    lines = [
        "# Bong Lem Weekly Analytics Reports",
        "",
        f"Data period: {format_date(daily['date'].min())} to {format_date(daily['date'].max())}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overall performance",
        "",
        f"- Visitors: {format_int(overall['visitors'])}",
        f"- Page views: {format_int(overall['page_views'])}",
        f"- Views per visitor: {format_float(overall['views_per_visitor'])}",
        f"- Active days: {format_int(overall['active_days'])} of {format_int(overall['days'])}",
        (
            f"- Strongest week: Week {strongest_week['week_number']} "
            f"({format_date_short(strongest_week['start'])} to {format_date_short(strongest_week['end'])}) "
            f"with {format_int(strongest_week['visitors'])} visitors"
        ),
        "",
        "## Weekly reports",
        "",
        markdown_table(
            [
                "Report",
                "Period",
                "Visitors",
                "Page views",
                "Views/visitor",
                "Best day",
                "Visitor movement",
                "Page-view movement",
            ],
            weekly_rows,
        ),
        "",
        "## Audience and acquisition context",
        "",
        "The following tables use the aggregate segment files from `total-data`.",
    ]

    section_titles = {
        "channels": "Known acquisition channels",
        "referrals": "Top referrers",
        "countries": "Top countries",
        "devices": "Top devices",
        "operating_systems": "Top operating systems",
    }
    for key in ("channels", "referrals", "countries", "devices", "operating_systems"):
        df = segments.get(key)
        if df is not None and not df.empty:
            lines.extend(["", f"### {section_titles[key]}", "", segment_table(df)])

    lines.extend(["", "## Business takeaways", ""])
    lines.extend(make_overall_takeaways(overall, weeks, segments))

    lines.extend(
        [
            "",
            "## Data notes",
            "",
            f"- Daily source file: `{notes.visitor_file.as_posix()}`",
            f"- Base year used for month/day labels: {notes.base_year}",
            (
                "- Visitor fallback rows: "
                f"{notes.fallback_visitor_rows} rows used the second numeric column as visitors "
                "because the named visitor column was blank."
            ),
            f"- Empty future/no-data rows ignored: {notes.dropped_empty_rows}",
        ]
    )

    path = output_dir / "index.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_overall_takeaways(
    overall: dict[str, object],
    weeks: list[tuple[int, pd.Timestamp, pd.Timestamp, pd.DataFrame]],
    segments: dict[str, pd.DataFrame],
) -> list[str]:
    takeaways: list[str] = []
    views_per_visitor = float(overall["views_per_visitor"])

    if views_per_visitor >= 1.5:
        takeaways.append(
            f"- Overall engagement is solid at {format_float(views_per_visitor)} views per visitor. Use conversion prompts on the pages that get second-page traffic."
        )
    else:
        takeaways.append(
            f"- Overall engagement is light at {format_float(views_per_visitor)} views per visitor. The biggest lever is guiding visitors to another product, collection, or checkout step."
        )

    week_summaries = [
        (week_number, start, end, summarize_period(week))
        for week_number, start, end, week in weeks
    ]
    if len(week_summaries) >= 2:
        latest_week = week_summaries[-1]
        previous_week = week_summaries[-2]
        latest_visitors = int(latest_week[3]["visitors"])
        previous_visitors = int(previous_week[3]["visitors"])
        change = pct_delta(latest_visitors, previous_visitors)
        if change is not None and change <= -0.2:
            takeaways.append(
                "- The latest week is down meaningfully from the prior week, so campaign consistency and referral sources should be checked first."
            )
        elif change is not None and change >= 0.2:
            takeaways.append(
                "- The latest week is gaining momentum. Preserve the active campaigns and compare creative or product messages from the best days."
            )

    devices = segments.get("devices")
    if devices is not None and not devices.empty:
        mobile = devices[devices["segment"].str.lower().eq("mobile")]
        if not mobile.empty:
            mobile_share = mobile.iloc[0]["visitors"] / devices["visitors"].sum()
            takeaways.append(
                f"- Mobile represents {format_percent(mobile_share)} of known visitors, so mobile speed and checkout clarity should lead QA."
            )

    channels = segments.get("channels")
    if channels is not None and not channels.empty:
        top_channel = channels.iloc[0]
        top_share = top_channel["visitors"] / channels["visitors"].sum()
        takeaways.append(
            f"- {top_channel['segment']} contributes {format_percent(top_share)} of known referral visitors. Use UTM tags so future weekly reports can connect posts to outcomes."
        )

    countries = segments.get("countries")
    if countries is not None and not countries.empty:
        top_country = countries.iloc[0]
        top_share = top_country["visitors"] / countries["visitors"].sum()
        takeaways.append(
            f"- {top_country['segment']} contributes {format_percent(top_share)} of known visitors. Keep copy, pricing, fulfillment, and support details strongest for that market."
        )

    return takeaways


def generate_reports(data_dir: Path, output_dir: Path, year: int | None) -> tuple[Path, list[ReportFile]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    daily, notes = load_daily_metrics(data_dir, year=year)
    segments = load_segments(data_dir)
    weeks = weekly_slices(daily)

    report_files: list[ReportFile] = []
    previous_summary: dict[str, object] | None = None
    for week_number, start, end, week in weeks:
        report_file = write_week_report(
            output_dir=output_dir,
            week_number=week_number,
            start=start,
            end=end,
            week=week,
            previous_summary=previous_summary,
            segments=segments,
        )
        report_files.append(report_file)
        previous_summary = summarize_period(week)

    index_path = write_index_report(
        output_dir=output_dir,
        daily=daily,
        weeks=weeks,
        report_files=report_files,
        segments=segments,
        notes=notes,
    )
    return index_path, report_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate weekly Markdown business analytics reports from total-data CSV exports."
    )
    parser.add_argument(
        "--data-dir",
        default="total-data",
        type=Path,
        help="Directory containing the exported analytics CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="report",
        type=Path,
        help="Directory where Markdown reports will be written.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year for month/day labels in the visitor CSV. Defaults to the visitor file modification year.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path, report_files = generate_reports(args.data_dir, args.output_dir, args.year)
    print(f"Wrote {len(report_files)} weekly reports")
    print(f"Index: {index_path}")
    for report_file in report_files:
        print(f"- {report_file.path}")


if __name__ == "__main__":
    main()
