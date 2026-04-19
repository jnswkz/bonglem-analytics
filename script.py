import pandas as pd
import gspread
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google.oauth2.credentials import Credentials as UserCredentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
from glob import glob
from datetime import datetime
import json

# ====================== CONFIGURATION ======================
CSV_FOLDER = "."                     # Folder containing ALL your historical CSVs
SHEET_ID = "1oLrHCshibYZ5QBTLhX8D7kxY2Xxj9QxYpxxaBibTf0c"   # ←←← CHANGE THIS (from URL: docs.google.com/spreadsheets/d/THIS-PART/edit)

# Mapping: CSV prefix → (Sheet name in Google Sheets, Correct column name)
MAPPINGS = {
    "Top Devices":          ("Device", "Device"),
    "Top Operating Systems": ("Operating System", "Operating System"),
    "Top Countries":        ("Country", "Country"),
    "Top Referrers":        ("Referral", "Referral"),
}

# ===========================================================

def get_all_csvs_for_prefix(prefix: str) -> list:
    """Find ALL CSV files matching the prefix (historical + new)."""
    pattern = os.path.join(CSV_FOLDER, "**", f"*{prefix}*.csv")
    files = glob(pattern, recursive=True)
    if not files:
        print(f"⚠️  No files found for '{prefix}'")
        return []
    # Sort by modification time (newest last)
    files.sort(key=os.path.getmtime)
    print(f"   Found {len(files)} file(s) for {prefix}")
    return files

def aggregate_category(prefix: str, new_column_name: str) -> pd.DataFrame | None:
    """Load ALL CSVs for this category, sum Visitors + Total, return aggregated DF."""
    csv_files = get_all_csvs_for_prefix(prefix)
    if not csv_files:
        return None

    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        if "Page" in df.columns:
            df = df.rename(columns={"Page": new_column_name})
        dfs.append(df)

    # Combine everything
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sum numeric columns by the category key
    agg_df = combined.groupby(new_column_name, as_index=False).agg({
        "Visitors": "sum",
        "Total": "sum"
    })
    
    # Sort by Total descending (most important)
    agg_df = agg_df.sort_values(by="Total", ascending=False).reset_index(drop=True)
    
    print(f"✅ Aggregated {len(csv_files)} files → {len(agg_df)} unique rows")
    return agg_df


def authorize_client(scopes: list[str], credentials_path: str = "credentials.json") -> gspread.Client:
    """Authorize gspread using either service-account or OAuth desktop credentials."""
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(
            f"Missing '{credentials_path}'. Add a service-account key or OAuth client credentials file."
        )

    with open(credentials_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Service account JSON format
    if "client_email" in config and "token_uri" in config:
        creds = ServiceAccountCredentials.from_service_account_file(credentials_path, scopes=scopes)
        return gspread.authorize(creds)

    # OAuth desktop/web client JSON format
    if "installed" in config or "web" in config:
        token_path = "token.json"
        creds = None

        if os.path.exists(token_path):
            creds = UserCredentials.from_authorized_user_file(token_path, scopes=scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes=scopes)
                creds = flow.run_local_server(port=0)

            with open(token_path, "w", encoding="utf-8") as token:
                token.write(creds.to_json())

        return gspread.authorize(creds)

    raise ValueError(
        "Unsupported credentials format in credentials.json. "
        "Expected service-account keys or OAuth client config (installed/web)."
    )

def main():
    # Authenticate with Google
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets"
    ]
    try:
        client = authorize_client(SCOPES)
    except Exception as e:
        print("❌ Authentication failed. Check credentials.json format and API permissions.")
        print(e)
        return
    
    try:
        spreadsheet = client.open_by_key(SHEET_ID)
        print(f"✅ Connected to Google Sheet: {spreadsheet.title}\n")
    except Exception as e:
        print("❌ Could not open sheet. Check SHEET_ID and sharing permissions.")
        print(e)
        return

    updated_count = 0
    for prefix, (sheet_name, column_name) in MAPPINGS.items():
        agg_df = aggregate_category(prefix, column_name)
        if agg_df is None:
            continue

        try:
            worksheet = spreadsheet.worksheet(sheet_name)
            # Clear and write new aggregated data
            worksheet.clear()
            # Write headers + data
            worksheet.update([agg_df.columns.values.tolist()] + agg_df.values.tolist())
            print(f"✅ Updated sheet → '{sheet_name}' ({len(agg_df)} rows)")
            updated_count += 1
        except gspread.exceptions.WorksheetNotFound:
            print(f"⚠️  Sheet '{sheet_name}' not found in your spreadsheet (skipping)")
        except gspread.exceptions.APIError as e:
            print(f"❌ Could not update '{sheet_name}' due to API permission error (skipping)")
            print("   Ensure the authenticated account/service account has Editor access to this spreadsheet.")
            print(f"   Details: {e}")

    print(f"\n🎉 FINISHED! Updated {updated_count} sheets with cumulative data")
    print(f"   🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   📊 All historical CSVs have been summed correctly")

if __name__ == "__main__":
    main()