from __future__ import annotations

import json
import textwrap
from pathlib import Path


NOTEBOOK_PATH = Path("report.ipynb")


def source(text: str) -> list[str]:
    return textwrap.dedent(text).strip("\n").splitlines(True)


def md_cell(text: str, cell_id: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source(text),
    }


def code_cell(text: str, cell_id: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


cells = [
    md_cell(
        """
        # Bong Lem Business Analytics Report

        This notebook builds a business analytics report from:

        - Website traffic exports in `total-data/`
        - Order data in `bonglem.orders.json`

        Improvements in this version:

        - Cleans similar referral links into canonical sources, for example `m.facebook.com`, `l.facebook.com`, `lm.facebook.com`, and `facebook.com` become `facebook.com`.
        - Groups cleaned referrals into business acquisition channels such as Facebook, Search, Messenger, Instagram, and Payment flow.
        - Tracks performance weekly instead of only evaluating at the end of the period.
        - Adds order, payment, product, revenue, conversion, and risk metrics.
        - Saves standalone chart images to `report/assets/` and exports a complete Markdown report with those images at `report/final_campaign_report.md`.
        """,
        "cell-00",
    ),
    code_cell(
        """
        import importlib.util
        import subprocess
        import sys

        if importlib.util.find_spec("matplotlib") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        """,
        "cell-01",
    ),
    code_cell(
        """
        from __future__ import annotations

        import json
        import re
        from datetime import datetime
        from pathlib import Path
        from urllib.parse import urlparse

        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        try:
            from IPython.display import Markdown, display
        except Exception:
            def display(value):
                print(value)

            class Markdown(str):
                pass

        pd.set_option("display.max_columns", 120)
        pd.set_option("display.width", 160)
        plt.style.use("seaborn-v0_8-whitegrid")

        BASE_DIR = Path(".")
        DATA_DIR = BASE_DIR / "total-data"
        ORDERS_PATH = BASE_DIR / "bonglem.orders.json"
        REPORT_DIR = BASE_DIR / "report"
        ASSET_DIR = REPORT_DIR / "assets"
        REPORT_PATH = REPORT_DIR / "final_campaign_report.md"


        def read_csv_with_fallback(path: Path) -> pd.DataFrame:
            last_error = None
            for encoding in ("utf-8-sig", "utf-8", "cp1258", "cp1252"):
                try:
                    return pd.read_csv(path, encoding=encoding)
                except UnicodeDecodeError as exc:
                    last_error = exc
            raise ValueError(f"Could not decode {path}") from last_error


        def find_one_file(data_dir: Path, pattern: str) -> Path:
            matches = sorted(data_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if not matches:
                raise FileNotFoundError(f"No files matching pattern {pattern!r} in {data_dir}")
            return matches[0]


        def parse_date_labels(labels: pd.Series, base_year: int) -> pd.Series:
            parsed_dates = []
            year = base_year
            previous_month = None

            for value in labels.astype(str):
                text = value.strip()
                if not text or text.lower() == "nan":
                    parsed_dates.append(pd.NaT)
                    continue

                try:
                    partial = datetime.strptime(f"{text} 2000", "%b %d %Y")
                except ValueError:
                    parsed_dates.append(pd.NaT)
                    continue

                if previous_month is not None and partial.month < previous_month:
                    year += 1
                previous_month = partial.month
                parsed_dates.append(pd.Timestamp(year=year, month=partial.month, day=partial.day))

            return pd.Series(parsed_dates)


        def load_daily_traffic(data_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
            visitor_file = find_one_file(data_dir, "*Visitor_viewer*.csv")
            base_year = datetime.fromtimestamp(visitor_file.stat().st_mtime).year

            raw = read_csv_with_fallback(visitor_file).iloc[:, :4].copy()
            raw.columns = ["date_label", "helper_value", "visitors_named", "page_views"]

            helper = pd.to_numeric(raw["helper_value"], errors="coerce")
            visitors_named = pd.to_numeric(raw["visitors_named"], errors="coerce")
            page_views = pd.to_numeric(raw["page_views"], errors="coerce")

            fallback_mask = visitors_named.isna() & helper.notna() & page_views.notna() & (helper <= page_views)
            visitors = visitors_named.copy()
            visitors.loc[fallback_mask] = helper.loc[fallback_mask]

            daily = pd.DataFrame(
                {
                    "date": parse_date_labels(raw["date_label"], base_year=base_year),
                    "visitors": visitors,
                    "page_views": page_views,
                }
            )

            empty_mask = daily[["visitors", "page_views"]].isna().all(axis=1)
            daily = daily.loc[~empty_mask].dropna(subset=["date"]).copy()
            daily["visitors"] = daily["visitors"].fillna(0).round().astype(int)
            daily["page_views"] = daily["page_views"].fillna(0).round().astype(int)

            daily = daily.groupby("date", as_index=False).agg(
                visitors=("visitors", "sum"),
                page_views=("page_views", "sum"),
            )
            full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
            daily = daily.set_index("date").reindex(full_dates, fill_value=0).rename_axis("date").reset_index()

            notes = {
                "source_file": visitor_file.as_posix(),
                "base_year": base_year,
                "fallback_visitor_rows": int(fallback_mask.sum()),
                "ignored_empty_rows": int(empty_mask.sum()),
            }
            return daily, notes


        def load_segment_table(data_dir: Path, filename_keyword: str, segment_name: str) -> pd.DataFrame:
            path = find_one_file(data_dir, f"*{filename_keyword}*.csv")
            raw = read_csv_with_fallback(path).iloc[:, :3].copy()
            raw.columns = [segment_name, "Visitors", "Total"]
            raw[segment_name] = raw[segment_name].astype(str).str.strip()
            raw["Visitors"] = pd.to_numeric(raw["Visitors"], errors="coerce").fillna(0).astype(int)
            raw["Total"] = pd.to_numeric(raw["Total"], errors="coerce").fillna(0).astype(int)
            raw = raw[raw[segment_name].ne("") & raw[segment_name].str.lower().ne("nan")]
            return raw.sort_values(["Total", "Visitors"], ascending=False).reset_index(drop=True)


        def canonical_domain(value: object) -> str:
            text = str(value).strip().lower()
            if not text or text == "nan":
                return "unknown"

            parsed = urlparse(text if "://" in text else f"https://{text}")
            domain = parsed.netloc or parsed.path.split("/")[0]
            domain = domain.split(":")[0].strip().strip(".")
            if domain.startswith("www."):
                domain = domain[4:]

            parts = [part for part in domain.split(".") if part]
            while len(parts) > 2 and parts[0] in {"m", "l", "lm", "mobile", "touch", "web"}:
                parts = parts[1:]
            domain = ".".join(parts)

            if domain.endswith("facebook.com"):
                return "facebook.com"
            if domain.endswith("messenger.com"):
                return "messenger.com"
            if domain.endswith("instagram.com"):
                return "instagram.com"
            if domain.endswith("google.com"):
                return "google.com"
            if domain.endswith("search.yahoo.com") or domain.endswith("yahoo.com"):
                return "yahoo.com"
            if domain.endswith("payos.vn"):
                return "payos.vn"
            return domain or "unknown"


        def classify_referral_channel(clean_referrer: str) -> str:
            value = str(clean_referrer).lower()
            if value == "facebook.com":
                return "Facebook"
            if value == "messenger.com":
                return "Messenger"
            if value == "instagram.com":
                return "Instagram"
            if value in {"google.com", "yahoo.com", "bing.com", "duckduckgo.com"}:
                return "Search"
            if value == "payos.vn":
                return "Payment flow"
            return "Other referral"


        def clean_referrals(referrals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            mapping = referrals.copy()
            mapping["CleanReferrer"] = mapping["Referrer"].map(canonical_domain)
            mapping["Channel"] = mapping["CleanReferrer"].map(classify_referral_channel)

            cleaned = (
                mapping.groupby(["CleanReferrer", "Channel"], as_index=False)
                .agg(Visitors=("Visitors", "sum"), Total=("Total", "sum"), RawLinks=("Referrer", lambda s: ", ".join(sorted(s))))
                .sort_values(["Total", "Visitors"], ascending=False)
                .reset_index(drop=True)
            )
            channels = (
                mapping.groupby("Channel", as_index=False)
                .agg(Visitors=("Visitors", "sum"), Total=("Total", "sum"), Sources=("CleanReferrer", lambda s: ", ".join(sorted(set(s)))))
                .sort_values(["Total", "Visitors"], ascending=False)
                .reset_index(drop=True)
            )
            return cleaned, mapping, channels


        def parse_mongo_date(value) -> pd.Timestamp | pd.NaT:
            if isinstance(value, dict) and "$date" in value:
                value = value["$date"]
            ts = pd.to_datetime(value, errors="coerce", utc=True)
            if pd.isna(ts):
                return pd.NaT
            return ts.tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)


        def get_col(df: pd.DataFrame, name: str, default: object) -> pd.Series:
            if name in df.columns:
                return df[name]
            return pd.Series([default] * len(df), index=df.index)


        def order_id(value) -> str:
            if isinstance(value, dict):
                return str(value.get("$oid", ""))
            return str(value)


        def normalize_product_name(value: object) -> str:
            text = " ".join(str(value).strip().split())
            if not text or text.lower() == "nan":
                return "Unknown product"
            key = text.casefold().replace("  ", " ")
            if key == "noodle (miu miu)":
                return "Noodle (Miu Miu)"
            return text


        def load_orders(path: Path) -> pd.DataFrame:
            if not path.exists():
                return pd.DataFrame()

            with path.open("r", encoding="utf-8") as f:
                orders = json.load(f)

            df = pd.DataFrame(orders)
            if df.empty:
                return df

            df["order_id"] = get_col(df, "_id", "").map(order_id)
            df["created_at"] = get_col(df, "createdAt", pd.NaT).apply(parse_mongo_date)
            df["updated_at"] = get_col(df, "updatedAt", pd.NaT).apply(parse_mongo_date)
            df["status"] = get_col(df, "status", "unknown").fillna("unknown").astype(str).str.lower()
            df["paymentMethod"] = get_col(df, "paymentMethod", "unknown").fillna("unknown").astype(str).str.lower()
            df["paymentStatus"] = get_col(df, "paymentStatus", "unknown").fillna("unknown").astype(str).str.lower()
            df["orderKind"] = get_col(df, "orderKind", "unknown").fillna("unknown").astype(str).str.lower()
            df["total"] = pd.to_numeric(get_col(df, "total", 0), errors="coerce").fillna(0)
            df["subtotal"] = pd.to_numeric(get_col(df, "subtotal", 0), errors="coerce").fillna(0)
            df["is_completed"] = df["status"].eq("completed")
            df["is_paid"] = df["paymentStatus"].eq("paid")
            df["is_cancelled"] = df["status"].eq("cancelled")
            return df


        def explode_order_items(orders_df: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for _, order in orders_df.iterrows():
                for item in order.get("items", []) or []:
                    quantity = int(pd.to_numeric(item.get("quantity", 0), errors="coerce") or 0)
                    price = float(pd.to_numeric(item.get("price", 0), errors="coerce") or 0)
                    rows.append(
                        {
                            "order_id": order["order_id"],
                            "created_at": order["created_at"],
                            "week_no": order.get("week_no", np.nan),
                            "status": order["status"],
                            "paymentMethod": order["paymentMethod"],
                            "paymentStatus": order["paymentStatus"],
                            "is_completed": bool(order["is_completed"]),
                            "is_paid": bool(order["is_paid"]),
                            "Product": normalize_product_name(item.get("name", "Unknown product")),
                            "Quantity": quantity,
                            "Price": price,
                            "ItemRevenue": quantity * price,
                        }
                    )
            return pd.DataFrame(rows)


        def add_week_index(df: pd.DataFrame, date_col: str, campaign_start: pd.Timestamp) -> pd.DataFrame:
            out = df.copy()
            if out.empty:
                out["day_index"] = pd.Series(dtype=int)
                out["week_no"] = pd.Series(dtype=int)
                return out
            out["day_index"] = (out[date_col] - campaign_start).dt.days
            out["week_no"] = (out["day_index"] // 7 + 1).astype("Int64")
            return out


        def safe_div(numerator: float, denominator: float) -> float:
            return float(numerator) / float(denominator) if denominator else 0.0


        def add_share(df: pd.DataFrame, value_col: str, share_col: str = "Share") -> pd.DataFrame:
            out = df.copy()
            total = out[value_col].sum()
            out[share_col] = np.where(total > 0, out[value_col] / total, 0)
            return out


        def format_int(value: object) -> str:
            if pd.isna(value):
                return "-"
            return f"{int(round(float(value))):,}"


        def format_currency(value: object) -> str:
            if pd.isna(value):
                return "-"
            return f"{float(value):,.0f} VND"


        def format_pct(value: object, digits: int = 1) -> str:
            if pd.isna(value):
                return "-"
            return f"{100 * float(value):.{digits}f}%"


        def format_float(value: object, digits: int = 2) -> str:
            if pd.isna(value):
                return "-"
            return f"{float(value):,.{digits}f}"


        def md_table(df: pd.DataFrame) -> str:
            if df.empty:
                return "_No data available._"
            headers = [str(col) for col in df.columns]
            lines = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            for _, row in df.iterrows():
                values = []
                for value in row.values:
                    if isinstance(value, pd.Timestamp):
                        text = value.date().isoformat()
                    else:
                        text = str(value)
                    values.append(text.replace("|", "\\\\|").replace("\\n", " "))
                lines.append("| " + " | ".join(values) + " |")
            return "\\n".join(lines)
        """,
        "cell-02",
    ),
    code_cell(
        """
        daily, traffic_notes = load_daily_traffic(DATA_DIR)

        countries = load_segment_table(DATA_DIR, "Country", "Country")
        devices = load_segment_table(DATA_DIR, "Device", "Device")
        os_table = load_segment_table(DATA_DIR, "Operating System", "OperatingSystem")
        referrals_raw = load_segment_table(DATA_DIR, "Referral", "Referrer")
        cleaned_referrals, referral_mapping, channel_table = clean_referrals(referrals_raw)
        orders = load_orders(ORDERS_PATH)

        campaign_start = daily["date"].min()
        campaign_end = daily["date"].max()
        campaign_days = len(daily)

        if orders.empty:
            orders_campaign = orders.copy()
        else:
            orders_campaign = orders[
                (orders["created_at"] >= campaign_start)
                & (orders["created_at"] < campaign_end + pd.Timedelta(days=1))
            ].copy()

        daily = add_week_index(daily, "date", campaign_start)
        orders_campaign = add_week_index(orders_campaign, "created_at", campaign_start)
        items_campaign = explode_order_items(orders_campaign)

        print(f"Campaign period: {campaign_start.date()} -> {campaign_end.date()} ({campaign_days} days)")
        print(f"Rows loaded | daily traffic: {len(daily)} | orders: {len(orders)} | campaign orders: {len(orders_campaign)} | order items: {len(items_campaign)}")
        print(f"Referral cleanup | raw links: {len(referrals_raw)} | cleaned sources: {len(cleaned_referrals)} | channels: {len(channel_table)}")

        display(daily.head())
        display(referral_mapping[["Referrer", "CleanReferrer", "Channel", "Visitors", "Total"]])
        if not orders_campaign.empty:
            display(orders_campaign[["created_at", "status", "paymentMethod", "paymentStatus", "total"]].head())
        """,
        "cell-03",
    ),
    code_cell(
        """
        visitors_total = int(daily["visitors"].sum())
        page_views_total = int(daily["page_views"].sum())
        completed_mask = orders_campaign["status"].eq("completed") if not orders_campaign.empty else pd.Series(dtype=bool)
        paid_mask = orders_campaign["paymentStatus"].eq("paid") if not orders_campaign.empty else pd.Series(dtype=bool)
        cancelled_mask = orders_campaign["status"].eq("cancelled") if not orders_campaign.empty else pd.Series(dtype=bool)

        completed_orders = int(completed_mask.sum()) if not orders_campaign.empty else 0
        paid_orders = int(paid_mask.sum()) if not orders_campaign.empty else 0
        cancelled_orders = int(cancelled_mask.sum()) if not orders_campaign.empty else 0
        completed_gmv = float(orders_campaign.loc[completed_mask, "total"].sum()) if not orders_campaign.empty else 0
        paid_gmv = float(orders_campaign.loc[paid_mask, "total"].sum()) if not orders_campaign.empty else 0
        all_gmv = float(orders_campaign["total"].sum()) if not orders_campaign.empty else 0

        peak_traffic_day = daily.loc[daily["visitors"].idxmax()]
        peak_pageview_day = daily.loc[daily["page_views"].idxmax()]

        traffic_kpis = {
            "Visitors": visitors_total,
            "Page Views": page_views_total,
            "Views / Visitor": safe_div(page_views_total, visitors_total),
            "Average Daily Visitors": daily["visitors"].mean(),
            "Average Daily Page Views": daily["page_views"].mean(),
            "Peak Visitor Day": peak_traffic_day["date"].date().isoformat(),
            "Peak Visitors": int(peak_traffic_day["visitors"]),
            "Peak Page-View Day": peak_pageview_day["date"].date().isoformat(),
            "Peak Page Views": int(peak_pageview_day["page_views"]),
        }

        order_kpis = {
            "Orders": int(len(orders_campaign)),
            "Completed Orders": completed_orders,
            "Paid Orders": paid_orders,
            "Cancelled Orders": cancelled_orders,
            "Completion Rate": safe_div(completed_orders, len(orders_campaign)),
            "Paid Rate": safe_div(paid_orders, len(orders_campaign)),
            "Cancellation Rate": safe_div(cancelled_orders, len(orders_campaign)),
            "Visitor -> Order Rate": safe_div(len(orders_campaign), visitors_total),
            "Visitor -> Completed Order Rate": safe_div(completed_orders, visitors_total),
            "Visitor -> Paid Order Rate": safe_div(paid_orders, visitors_total),
            "GMV (All Orders)": all_gmv,
            "GMV (Completed Orders)": completed_gmv,
            "GMV (Paid Orders)": paid_gmv,
            "AOV (Completed)": safe_div(completed_gmv, completed_orders),
            "Revenue / Visitor (Completed GMV)": safe_div(completed_gmv, visitors_total),
        }

        kpi_df = pd.DataFrame(
            {
                "Metric": list(traffic_kpis.keys()) + list(order_kpis.keys()),
                "Value": list(traffic_kpis.values()) + list(order_kpis.values()),
            }
        )

        funnel_df = pd.DataFrame(
            [
                ["Visitors", visitors_total, "100.00%"],
                ["Orders", len(orders_campaign), format_pct(safe_div(len(orders_campaign), visitors_total), 2)],
                ["Completed orders", completed_orders, format_pct(safe_div(completed_orders, visitors_total), 2)],
                ["Paid orders", paid_orders, format_pct(safe_div(paid_orders, visitors_total), 2)],
            ],
            columns=["Stage", "Count", "Rate vs Visitors"],
        )

        weekly_traffic = daily.groupby("week_no", as_index=False).agg(
            StartDate=("date", "min"),
            EndDate=("date", "max"),
            Visitors=("visitors", "sum"),
            PageViews=("page_views", "sum"),
            ActiveDays=("visitors", lambda s: int((s > 0).sum())),
        )
        weekly_traffic["ViewsPerVisitor"] = np.where(
            weekly_traffic["Visitors"] > 0,
            weekly_traffic["PageViews"] / weekly_traffic["Visitors"],
            0,
        )

        if orders_campaign.empty:
            weekly_orders = pd.DataFrame(columns=["week_no"])
        else:
            weekly_orders = orders_campaign.groupby("week_no", as_index=False).agg(
                Orders=("status", "size"),
                CompletedOrders=("status", lambda s: int((s == "completed").sum())),
                PaidOrders=("paymentStatus", lambda s: int((s == "paid").sum())),
                CancelledOrders=("status", lambda s: int((s == "cancelled").sum())),
                PendingOrders=("status", lambda s: int((s == "pending").sum())),
                GMV=("total", "sum"),
                CompletedGMV=("total", lambda s: float(s[orders_campaign.loc[s.index, "status"].eq("completed")].sum())),
                PaidGMV=("total", lambda s: float(s[orders_campaign.loc[s.index, "paymentStatus"].eq("paid")].sum())),
            )

        weekly = weekly_traffic.merge(weekly_orders, on="week_no", how="left").fillna(0)
        for col in ["Orders", "CompletedOrders", "PaidOrders", "CancelledOrders", "PendingOrders"]:
            weekly[col] = weekly[col].astype(int)
        weekly["CompletionRate"] = np.where(weekly["Orders"] > 0, weekly["CompletedOrders"] / weekly["Orders"], 0)
        weekly["PaidRate"] = np.where(weekly["Orders"] > 0, weekly["PaidOrders"] / weekly["Orders"], 0)
        weekly["CancelRate"] = np.where(weekly["Orders"] > 0, weekly["CancelledOrders"] / weekly["Orders"], 0)
        weekly["EstConversion"] = np.where(weekly["Visitors"] > 0, weekly["CompletedOrders"] / weekly["Visitors"], 0)
        weekly["RevenuePerVisitor"] = np.where(weekly["Visitors"] > 0, weekly["CompletedGMV"] / weekly["Visitors"], 0)
        weekly["AOVCompleted"] = np.where(weekly["CompletedOrders"] > 0, weekly["CompletedGMV"] / weekly["CompletedOrders"], 0)
        weekly["WoW Visitors %"] = weekly["Visitors"].pct_change().replace([np.inf, -np.inf], np.nan)
        weekly["WoW Completed GMV %"] = weekly["CompletedGMV"].pct_change().replace([np.inf, -np.inf], np.nan)
        weekly["WeekLabel"] = weekly["StartDate"].dt.strftime("%b %d") + " - " + weekly["EndDate"].dt.strftime("%b %d")

        if orders_campaign.empty:
            orders_by_day = pd.DataFrame(columns=["date"])
        else:
            orders_by_day = (
                orders_campaign.assign(date=orders_campaign["created_at"].dt.floor("D"))
                .groupby("date", as_index=False)
                .agg(
                    Orders=("status", "size"),
                    CompletedOrders=("status", lambda s: int((s == "completed").sum())),
                    PaidOrders=("paymentStatus", lambda s: int((s == "paid").sum())),
                    GMV=("total", "sum"),
                    CompletedGMV=("total", lambda s: float(s[orders_campaign.loc[s.index, "status"].eq("completed")].sum())),
                )
            )

        daily_business = daily.merge(orders_by_day, on="date", how="left").fillna(0)
        for col in ["Orders", "CompletedOrders", "PaidOrders"]:
            daily_business[col] = daily_business[col].astype(int)
        daily_business["CompletedConversion"] = np.where(
            daily_business["visitors"] > 0,
            daily_business["CompletedOrders"] / daily_business["visitors"],
            0,
        )
        daily_business["RevenuePerVisitor"] = np.where(
            daily_business["visitors"] > 0,
            daily_business["CompletedGMV"] / daily_business["visitors"],
            0,
        )

        weekly_alerts = weekly[["week_no", "WeekLabel", "Visitors", "CompletedOrders", "CompletedGMV", "WoW Visitors %", "WoW Completed GMV %"]].copy()
        weekly_alerts["TrafficSignal"] = weekly_alerts["WoW Visitors %"].map(
            lambda x: "Baseline" if pd.isna(x) else ("Increase" if x > 0.2 else "Decrease" if x < -0.2 else "Stable")
        )
        weekly_alerts["RevenueSignal"] = weekly_alerts["WoW Completed GMV %"].map(
            lambda x: "Baseline" if pd.isna(x) else ("Increase" if x > 0.2 else "Decrease" if x < -0.2 else "Stable")
        )

        print("Executive KPI Table")
        display(kpi_df)
        print("Visitor-to-order funnel")
        display(funnel_df)
        print("Weekly business performance")
        display(weekly)
        print("Weekly increase/decrease alerts")
        display(weekly_alerts)
        """,
        "cell-04",
    ),
    code_cell(
        """
        top_countries = add_share(countries, "Visitors", "VisitorShare").head(10).copy()
        top_devices = add_share(devices, "Visitors", "VisitorShare").head(10).copy()
        top_os = add_share(os_table, "Visitors", "VisitorShare").head(10).copy()
        cleaned_referrals = add_share(cleaned_referrals, "Visitors", "VisitorShare")
        channel_table = add_share(channel_table, "Visitors", "VisitorShare")

        if orders_campaign.empty:
            status_table = pd.DataFrame(columns=["status", "Orders", "GMV", "OrderShare", "GMVShare"])
            payment_method_table = pd.DataFrame(columns=["paymentMethod", "Orders", "CompletedOrders", "PaidOrders", "GMV", "CompletionRate", "PaidRate", "AOV"])
            payment_status_table = pd.DataFrame(columns=["paymentStatus", "Orders", "GMV", "OrderShare"])
        else:
            status_table = (
                orders_campaign.groupby("status", as_index=False)
                .agg(Orders=("status", "size"), GMV=("total", "sum"))
                .sort_values("Orders", ascending=False)
            )
            status_table = add_share(status_table, "Orders", "OrderShare")
            status_table = add_share(status_table, "GMV", "GMVShare")

            payment_method_table = (
                orders_campaign.groupby("paymentMethod", as_index=False)
                .agg(
                    Orders=("paymentMethod", "size"),
                    CompletedOrders=("status", lambda s: int((s == "completed").sum())),
                    PaidOrders=("paymentStatus", lambda s: int((s == "paid").sum())),
                    GMV=("total", "sum"),
                )
                .sort_values("Orders", ascending=False)
            )
            payment_method_table["CompletionRate"] = payment_method_table["CompletedOrders"] / payment_method_table["Orders"]
            payment_method_table["PaidRate"] = payment_method_table["PaidOrders"] / payment_method_table["Orders"]
            payment_method_table["AOV"] = payment_method_table["GMV"] / payment_method_table["Orders"]

            payment_status_table = (
                orders_campaign.groupby("paymentStatus", as_index=False)
                .agg(Orders=("paymentStatus", "size"), GMV=("total", "sum"))
                .sort_values("Orders", ascending=False)
            )
            payment_status_table = add_share(payment_status_table, "Orders", "OrderShare")

        if items_campaign.empty:
            product_table = pd.DataFrame(columns=["Product", "Orders", "Quantity", "Revenue", "CompletedQuantity", "CompletedRevenue", "RevenueShare", "CumulativeRevenueShare"])
        else:
            product_table = (
                items_campaign.groupby("Product", as_index=False)
                .agg(
                    Orders=("order_id", "nunique"),
                    Quantity=("Quantity", "sum"),
                    Revenue=("ItemRevenue", "sum"),
                    CompletedQuantity=("Quantity", lambda s: int(s[items_campaign.loc[s.index, "is_completed"]].sum())),
                    CompletedRevenue=("ItemRevenue", lambda s: float(s[items_campaign.loc[s.index, "is_completed"]].sum())),
                    AveragePrice=("Price", "mean"),
                )
                .sort_values("CompletedRevenue", ascending=False)
                .reset_index(drop=True)
            )
            total_completed_revenue = product_table["CompletedRevenue"].sum()
            product_table["RevenueShare"] = np.where(total_completed_revenue > 0, product_table["CompletedRevenue"] / total_completed_revenue, 0)
            product_table["CumulativeRevenueShare"] = product_table["RevenueShare"].cumsum()

        top_traffic_days = daily_business.sort_values(["visitors", "page_views"], ascending=False).head(10)

        print("Cleaned referrals: raw links mapped into canonical sources")
        display(referral_mapping[["Referrer", "CleanReferrer", "Channel", "Visitors", "Total"]])
        print("Cleaned referral sources")
        display(cleaned_referrals)
        print("Acquisition channels")
        display(channel_table)
        print("Audience segments")
        display(top_countries)
        display(top_devices)
        display(top_os)
        print("Commercial diagnostics")
        display(status_table)
        display(payment_method_table)
        display(payment_status_table)
        display(product_table.head(15))
        """,
        "cell-05",
    ),
    code_cell(
        """
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        chart_files = {}


        def save_chart(fig, filename: str, title: str) -> Path:
            path = ASSET_DIR / filename
            fig.savefig(path, dpi=160, bbox_inches="tight")
            chart_files[title] = path
            plt.close(fig)
            return path


        def finish_standalone(fig, ax, filename: str, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
            ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            save_chart(fig, filename, title)


        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(daily_business["date"], daily_business["visitors"], label="Visitors", linewidth=2.2, color="#2563eb")
        ax.plot(daily_business["date"], daily_business["page_views"], label="Page views", linewidth=2.2, color="#16a34a")
        ax.plot(daily_business["date"], daily_business["visitors"].rolling(7, min_periods=1).mean(), label="Visitors 7-day avg", linestyle="--", color="#1e293b")
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
        finish_standalone(fig, ax, "daily_traffic_trend.png", "Daily Traffic Trend", "Date", "Count")

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(daily_business["date"], daily_business["CompletedOrders"], color="#f97316", alpha=0.8, label="Completed orders")
        ax.plot(daily_business["date"], daily_business["Orders"], color="#111827", marker="o", linewidth=1.6, label="All orders")
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
        finish_standalone(fig, ax, "daily_order_volume.png", "Daily Order Volume", "Date", "Orders")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(weekly["week_no"], weekly["Visitors"], color="#2563eb", alpha=0.85, label="Visitors")
        ax.plot(weekly["week_no"], weekly["PageViews"], color="#16a34a", marker="o", linewidth=2.2, label="Page views")
        ax.legend()
        finish_standalone(fig, ax, "weekly_traffic.png", "Weekly Traffic", "Week", "Count")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(weekly["week_no"], weekly["CompletedOrders"], color="#f97316", alpha=0.85, label="Completed orders")
        ax2 = ax.twinx()
        ax2.plot(weekly["week_no"], weekly["EstConversion"], color="#111827", marker="o", linewidth=2.2, label="Visitor -> completed")
        ax2.set_ylabel("Conversion rate")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right")
        finish_standalone(fig, ax, "weekly_conversion.png", "Weekly Conversion", "Week", "Completed orders")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(weekly["week_no"], weekly["CompletedGMV"], color="#0f766e", alpha=0.85, label="Completed GMV")
        ax2 = ax.twinx()
        ax2.plot(weekly["week_no"], weekly["RevenuePerVisitor"], color="#7c3aed", marker="o", linewidth=2.2, label="Revenue / visitor")
        ax2.set_ylabel("Revenue / visitor")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right")
        finish_standalone(fig, ax, "weekly_revenue_quality.png", "Weekly Revenue Quality", "Week", "Completed GMV")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(weekly["week_no"], weekly["ViewsPerVisitor"], color="#db2777", marker="o", linewidth=2.2)
        ax.axhline(1.5, color="#64748b", linestyle="--", linewidth=1, label="1.5 engagement target")
        ax.legend()
        finish_standalone(fig, ax, "weekly_engagement_depth.png", "Weekly Engagement Depth", "Week", "Views per visitor")

        channel_plot = channel_table.sort_values("Visitors", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(channel_plot["Channel"], channel_plot["Visitors"], color="#2563eb")
        finish_standalone(fig, ax, "acquisition_channels_cleaned.png", "Acquisition Channels After Referral Cleaning", "Visitors", None)

        ref_plot = cleaned_referrals.sort_values("Visitors", ascending=True).tail(8)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(ref_plot["CleanReferrer"], ref_plot["Visitors"], color="#16a34a")
        finish_standalone(fig, ax, "cleaned_referral_sources.png", "Cleaned Referral Sources", "Visitors", None)

        country_plot = top_countries.sort_values("Visitors", ascending=True).tail(8)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(country_plot["Country"], country_plot["Visitors"], color="#0f766e")
        finish_standalone(fig, ax, "top_countries.png", "Top Countries by Visitors", "Visitors", None)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(top_devices["Visitors"], labels=top_devices["Device"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Device Mix")
        fig.tight_layout()
        save_chart(fig, "device_mix.png", "Device Mix")

        os_plot = top_os.sort_values("Visitors", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(os_plot["OperatingSystem"], os_plot["Visitors"], color="#7c3aed")
        finish_standalone(fig, ax, "operating_systems.png", "Operating Systems by Visitors", "Visitors", None)

        status_plot = status_table.sort_values("Orders", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(status_plot["status"], status_plot["Orders"], color="#f97316")
        finish_standalone(fig, ax, "order_status_distribution.png", "Order Status Distribution", "Status", "Orders")

        payment_plot = payment_method_table.sort_values("Orders", ascending=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(payment_plot))
        width = 0.35
        ax.bar(x - width / 2, payment_plot["CompletionRate"], width, label="Completion", color="#2563eb")
        ax.bar(x + width / 2, payment_plot["PaidRate"], width, label="Paid", color="#16a34a")
        ax.set_xticks(x)
        ax.set_xticklabels(payment_plot["paymentMethod"])
        ax.legend()
        finish_standalone(fig, ax, "payment_method_quality.png", "Payment Method Quality", "Payment method", "Rate")

        product_revenue_plot = product_table.sort_values("CompletedRevenue", ascending=True).tail(8)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(product_revenue_plot["Product"], product_revenue_plot["CompletedRevenue"], color="#0f766e")
        finish_standalone(fig, ax, "top_products_revenue.png", "Top Products by Completed Revenue", "Completed revenue", None)

        product_qty_plot = product_table.sort_values("Quantity", ascending=True).tail(8)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(product_qty_plot["Product"], product_qty_plot["Quantity"], color="#db2777")
        finish_standalone(fig, ax, "top_products_quantity.png", "Top Products by Quantity Sold", "Quantity", None)

        print("Saved standalone chart assets:")
        display(pd.DataFrame([{"Chart": title, "File": path.as_posix()} for title, path in chart_files.items()]))
        """,
        "cell-06",
    ),
    code_cell(
        """
        def formatted_weekly_table() -> pd.DataFrame:
            out = weekly[[
                "week_no", "WeekLabel", "Visitors", "PageViews", "ViewsPerVisitor", "Orders", "CompletedOrders",
                "PaidOrders", "CompletedGMV", "CompletionRate", "PaidRate", "EstConversion", "RevenuePerVisitor",
                "WoW Visitors %", "WoW Completed GMV %",
            ]].copy()
            out.columns = [
                "Week", "Period", "Visitors", "Page views", "Views/visitor", "Orders", "Completed orders",
                "Paid orders", "Completed GMV", "Completion rate", "Paid rate", "Visitor->completed", "Revenue/visitor",
                "WoW visitors", "WoW completed GMV",
            ]
            for col in ["Visitors", "Page views", "Orders", "Completed orders", "Paid orders"]:
                out[col] = out[col].map(format_int)
            out["Views/visitor"] = out["Views/visitor"].map(lambda x: format_float(x, 2))
            out["Completed GMV"] = out["Completed GMV"].map(format_currency)
            for col in ["Completion rate", "Paid rate", "Visitor->completed", "WoW visitors", "WoW completed GMV"]:
                out[col] = out[col].map(lambda x: format_pct(x, 1) if not pd.isna(x) else "-")
            out["Revenue/visitor"] = out["Revenue/visitor"].map(format_currency)
            return out


        def formatted_segment(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
            out = df.head(limit).copy()
            if "VisitorShare" in out.columns:
                out["VisitorShare"] = out["VisitorShare"].map(lambda x: format_pct(x, 1))
            return out


        def formatted_status_table() -> pd.DataFrame:
            out = status_table.copy()
            if out.empty:
                return out
            out["GMV"] = out["GMV"].map(format_currency)
            out["OrderShare"] = out["OrderShare"].map(lambda x: format_pct(x, 1))
            out["GMVShare"] = out["GMVShare"].map(lambda x: format_pct(x, 1))
            return out


        def formatted_payment_method_table() -> pd.DataFrame:
            out = payment_method_table.copy()
            if out.empty:
                return out
            out["GMV"] = out["GMV"].map(format_currency)
            out["AOV"] = out["AOV"].map(format_currency)
            out["CompletionRate"] = out["CompletionRate"].map(lambda x: format_pct(x, 1))
            out["PaidRate"] = out["PaidRate"].map(lambda x: format_pct(x, 1))
            return out


        def formatted_product_table(limit: int = 12) -> pd.DataFrame:
            out = product_table.head(limit).copy()
            if out.empty:
                return out
            for col in ["Revenue", "CompletedRevenue", "AveragePrice"]:
                out[col] = out[col].map(format_currency)
            for col in ["RevenueShare", "CumulativeRevenueShare"]:
                out[col] = out[col].map(lambda x: format_pct(x, 1))
            return out


        def image_markdown(title: str) -> list[str]:
            rel = chart_files[title].relative_to(REPORT_DIR).as_posix()
            return [f"![{title}]({rel})", ""]


        best_week = weekly.sort_values("Visitors", ascending=False).iloc[0]
        best_revenue_week = weekly.sort_values("CompletedGMV", ascending=False).iloc[0]
        latest_week = weekly.iloc[-1]
        previous_week = weekly.iloc[-2] if len(weekly) > 1 else None
        top_channel = channel_table.iloc[0] if len(channel_table) else None
        top_country = countries.sort_values("Visitors", ascending=False).iloc[0] if len(countries) else None
        mobile_visitors = devices.loc[devices["Device"].str.lower().eq("mobile"), "Visitors"].sum() if len(devices) else 0
        mobile_share = safe_div(mobile_visitors, devices["Visitors"].sum()) if len(devices) else 0

        insights = [
            f"The website generated {format_int(visitors_total)} visitors and {format_int(page_views_total)} page views over {campaign_days} tracked days.",
            f"Engagement depth was {format_float(traffic_kpis['Views / Visitor'], 2)} views per visitor, which means many visitors viewed only one page before leaving.",
            f"The best traffic week was Week {int(best_week['week_no'])} ({best_week['WeekLabel']}) with {format_int(best_week['Visitors'])} visitors.",
            f"The best revenue week was Week {int(best_revenue_week['week_no'])} ({best_revenue_week['WeekLabel']}) with {format_currency(best_revenue_week['CompletedGMV'])} completed GMV.",
            f"Estimated visitor-to-completed-order conversion was {format_pct(order_kpis['Visitor -> Completed Order Rate'], 2)}.",
            f"Only {format_pct(order_kpis['Paid Rate'], 1)} of campaign orders were marked paid, so payment follow-up is a major operational lever.",
        ]

        if previous_week is not None and not pd.isna(latest_week["WoW Visitors %"]):
            direction = "increased" if latest_week["WoW Visitors %"] > 0 else "decreased"
            insights.append(
                f"Latest-week traffic {direction} by {format_pct(abs(latest_week['WoW Visitors %']), 1)} versus the previous week, so weekly monitoring is needed instead of end-period review only."
            )
        if top_channel is not None:
            insights.append(
                f"After cleaning similar referral links, {top_channel['Channel']} is the largest known acquisition channel with {format_pct(top_channel['VisitorShare'], 1)} of referral visitors."
            )
        if top_country is not None:
            insights.append(
                f"The audience is concentrated in {top_country['Country']} with {format_pct(safe_div(top_country['Visitors'], countries['Visitors'].sum()), 1)} of known visitors."
            )
        if mobile_share > 0.5:
            insights.append(f"Mobile is the default user experience because it contributes {format_pct(mobile_share, 1)} of known visitors.")

        action_plan = pd.DataFrame(
            [
                ["Weekly traffic drop >20%", "Review social posting cadence, ad spend, referral links, and product messages within 24 hours."],
                ["Views/visitor below 1.3", "Improve internal links, product recommendations, and clearer calls to action."],
                ["Paid rate below 50%", "Send payment reminders, clarify bank transfer instructions, and separate unpaid COD from bank transfer issues."],
                ["Facebook dominates referrals", "Use UTM links for every Facebook post so the next report can connect posts to visits and orders."],
                ["Mobile share above 50%", "Prioritize mobile page speed, checkout clarity, and first-screen product information."],
                ["Top country concentration above 75%", "Focus copy, shipping, payment information, and customer support for the main market first."],
            ],
            columns=["Trigger", "Action"],
        )

        monitoring_plan = pd.DataFrame(
            [
                ["Every Monday", "Update weekly KPI table", "Visitors, page views, orders, completed GMV, conversion, paid rate"],
                ["Mid-week", "Check acquisition health", "Referral/channel traffic, Facebook link performance, search traffic"],
                ["After each campaign post", "Track campaign effect", "Traffic spike, orders, completed orders, revenue per visitor"],
                ["Before next campaign", "Review product and payment issues", "Top products, cancelled orders, unpaid orders, payment method quality"],
            ],
            columns=["Timing", "Tracking activity", "Metrics"],
        )

        summary_kpi_df = pd.DataFrame(
            [
                ["Campaign period", f"{campaign_start.date()} to {campaign_end.date()}"],
                ["Total visitors", format_int(traffic_kpis["Visitors"])],
                ["Total page views", format_int(traffic_kpis["Page Views"])],
                ["Views per visitor", format_float(traffic_kpis["Views / Visitor"], 2)],
                ["Orders", format_int(order_kpis["Orders"])],
                ["Completed orders", format_int(order_kpis["Completed Orders"])],
                ["Paid orders", format_int(order_kpis["Paid Orders"])],
                ["Visitor -> completed order", format_pct(order_kpis["Visitor -> Completed Order Rate"], 2)],
                ["Completion rate", format_pct(order_kpis["Completion Rate"], 1)],
                ["Paid rate", format_pct(order_kpis["Paid Rate"], 1)],
                ["GMV (all orders)", format_currency(order_kpis["GMV (All Orders)"])],
                ["GMV (completed orders)", format_currency(order_kpis["GMV (Completed Orders)"])],
                ["AOV (completed)", format_currency(order_kpis["AOV (Completed)"])],
                ["Revenue / visitor", format_currency(order_kpis["Revenue / Visitor (Completed GMV)"])],
            ],
            columns=["Metric", "Value"],
        )

        referral_mapping_report = referral_mapping[["Referrer", "CleanReferrer", "Channel", "Visitors", "Total"]].copy()
        cleaned_referral_report = cleaned_referrals[["CleanReferrer", "Channel", "Visitors", "VisitorShare", "Total", "RawLinks"]].copy()
        cleaned_referral_report["VisitorShare"] = cleaned_referral_report["VisitorShare"].map(lambda x: format_pct(x, 1))
        channel_report = channel_table[["Channel", "Visitors", "VisitorShare", "Total", "Sources"]].copy()
        channel_report["VisitorShare"] = channel_report["VisitorShare"].map(lambda x: format_pct(x, 1))

        top_days_report = top_traffic_days[["date", "visitors", "page_views", "Orders", "CompletedOrders", "CompletedGMV", "RevenuePerVisitor"]].copy()
        top_days_report["date"] = top_days_report["date"].dt.date.astype(str)
        top_days_report["CompletedGMV"] = top_days_report["CompletedGMV"].map(format_currency)
        top_days_report["RevenuePerVisitor"] = top_days_report["RevenuePerVisitor"].map(format_currency)

        weekly_alerts_report = weekly_alerts.copy()
        weekly_alerts_report = weekly_alerts_report.rename(
            columns={
                "week_no": "Week",
                "WeekLabel": "Period",
                "CompletedOrders": "Completed orders",
                "CompletedGMV": "Completed GMV",
                "WoW Visitors %": "WoW visitors",
                "WoW Completed GMV %": "WoW completed GMV",
                "TrafficSignal": "Traffic signal",
                "RevenueSignal": "Revenue signal",
            }
        )
        weekly_alerts_report["Completed GMV"] = weekly_alerts_report["Completed GMV"].map(format_currency)
        weekly_alerts_report["WoW visitors"] = weekly_alerts_report["WoW visitors"].map(lambda x: "-" if pd.isna(x) else format_pct(x, 1))
        weekly_alerts_report["WoW completed GMV"] = weekly_alerts_report["WoW completed GMV"].map(lambda x: "-" if pd.isna(x) else format_pct(x, 1))

        payment_status_report = payment_status_table.copy()
        if not payment_status_report.empty:
            payment_status_report["GMV"] = payment_status_report["GMV"].map(format_currency)
            payment_status_report["OrderShare"] = payment_status_report["OrderShare"].map(lambda x: format_pct(x, 1))

        final_lines = [
            "# Bong Lem Campaign Business Analytics Report",
            "",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1) Executive KPI Snapshot",
            "",
            md_table(summary_kpi_df),
            "",
            "## 2) Key Business Insights",
            "",
        ]
        final_lines.extend([f"- {insight}" for insight in insights])

        final_lines.extend(["", "## 3) Weekly Performance Tracking", "", md_table(formatted_weekly_table()), ""])
        for title in ["Weekly Traffic", "Weekly Conversion", "Weekly Revenue Quality", "Weekly Engagement Depth"]:
            final_lines.extend(image_markdown(title))
        final_lines.extend(["### Weekly Increase/Decrease Alerts", "", md_table(weekly_alerts_report), ""])

        final_lines.extend(["", "## 4) Daily Traffic and Order Behavior", ""])
        for title in ["Daily Traffic Trend", "Daily Order Volume"]:
            final_lines.extend(image_markdown(title))
        final_lines.extend(["### Top Traffic Days", "", md_table(top_days_report), ""])

        final_lines.extend(["", "## 5) Cleaned Referral and Acquisition Analysis", ""])
        final_lines.extend([
            "Similar referral links were cleaned into canonical sources. For example, `m.facebook.com`, `l.facebook.com`, `lm.facebook.com`, and `facebook.com` are counted together as `facebook.com`.",
            "",
        ])
        for title in ["Acquisition Channels After Referral Cleaning", "Cleaned Referral Sources"]:
            final_lines.extend(image_markdown(title))
        final_lines.extend(["### Raw Referral Cleanup Mapping", "", md_table(referral_mapping_report), ""])
        final_lines.extend(["### Cleaned Referral Sources", "", md_table(cleaned_referral_report), ""])
        final_lines.extend(["### Acquisition Channels", "", md_table(channel_report), ""])

        final_lines.extend(["", "## 6) Audience Segmentation", ""])
        for title in ["Top Countries by Visitors", "Device Mix", "Operating Systems by Visitors"]:
            final_lines.extend(image_markdown(title))
        final_lines.extend(["### Countries", "", md_table(formatted_segment(top_countries)[["Country", "Visitors", "Total", "VisitorShare"]]), ""])
        final_lines.extend(["### Devices", "", md_table(formatted_segment(top_devices)[["Device", "Visitors", "Total", "VisitorShare"]]), ""])
        final_lines.extend(["### Operating Systems", "", md_table(formatted_segment(top_os)[["OperatingSystem", "Visitors", "Total", "VisitorShare"]]), ""])

        final_lines.extend(["", "## 7) Commercial Performance", ""])
        for title in ["Order Status Distribution", "Payment Method Quality", "Top Products by Completed Revenue", "Top Products by Quantity Sold"]:
            final_lines.extend(image_markdown(title))
        final_lines.extend(["### Order Status", "", md_table(formatted_status_table()), ""])
        final_lines.extend(["### Payment Method Quality", "", md_table(formatted_payment_method_table()), ""])
        final_lines.extend(["### Payment Status", "", md_table(payment_status_report), ""])
        final_lines.extend(["### Product Performance", "", md_table(formatted_product_table()), ""])

        final_lines.extend(["", "## 8) Recommended Action Plan", "", md_table(action_plan), ""])
        final_lines.extend(["", "## 9) Periodic Tracking Plan", "", md_table(monitoring_plan), ""])
        final_lines.extend(
            [
                "", "## 10) Data Notes and Limitations", "",
                f"- Daily traffic source: `{traffic_notes['source_file']}`",
                f"- Base year used for month/day traffic labels: {traffic_notes['base_year']}",
                f"- Visitor fallback rows fixed from the second numeric column: {traffic_notes['fallback_visitor_rows']}",
                f"- Empty future/no-data traffic rows ignored: {traffic_notes['ignored_empty_rows']}",
                "- Referral, country, device, and operating system exports are aggregate-level data, so they explain traffic mix but cannot directly attribute each order to a visitor source.",
                "- Order conversion is estimated by comparing total campaign orders with total campaign visitors, not by user-level tracking.",
            ]
        )

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\\n".join(final_lines) + "\\n", encoding="utf-8")

        print(f"Final Markdown report written to: {REPORT_PATH}")
        print("Standalone chart images included:")
        for title, path in chart_files.items():
            print(f"- {title}: {path}")
        display(Markdown("\\n".join(final_lines[:80])))
        """,
        "cell-07",
    ),
    md_cell(
        """
        ## Notes

        - Use `report/final_campaign_report.md` as the full export for submission. It includes standalone chart images from `report/assets/`.
        - The cleaned referral table prevents Facebook, Messenger, Instagram, search, and payment links from being split across many similar domains.
        - The weekly table should be updated during the project, not only at the end, so increases/decreases can trigger immediate actions.
        - Re-run all cells whenever `total-data/` or `bonglem.orders.json` changes.
        """,
        "cell-08",
    ),
]


def main() -> None:
    if NOTEBOOK_PATH.exists():
        current = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        metadata = current.get("metadata", {})
    else:
        metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }

    notebook = {
        "cells": cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Updated {NOTEBOOK_PATH} with {len(cells)} cells")


if __name__ == "__main__":
    main()
