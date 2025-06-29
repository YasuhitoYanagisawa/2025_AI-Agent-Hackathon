import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from google import genai
from google.genai import types
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import os
import json
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
from datetime import datetime, timedelta
import time
import math
import re

# --- 設定とAPI初期化 ---
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]
try:
    # 環境変数から認証情報を読み込む
    service_account_json_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not service_account_json_str:
        st.error("環境変数 'GOOGLE_SERVICE_ACCOUNT_JSON' が設定されていません。")
        st.stop()
    
    service_account_info = json.loads(service_account_json_str)
    credentials = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    gc = gspread.authorize(credentials)
    drive_service = build('drive', 'v3', credentials=credentials)

    SPREADSHEET_ID = os.getenv("SPREADSHEET_ID")
    DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")

    if not SPREADSHEET_ID or not DRIVE_FOLDER_ID:
        st.error("環境変数 'SPREADSHEET_ID' または 'DRIVE_FOLDER_ID' が設定されていません。")
        st.stop()

except (json.JSONDecodeError, Exception) as e:
    st.error(f"Google APIの認証に失敗しました。環境変数が正しく設定されているか確認してください。エラー: {e}")
    st.stop()

# Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("環境変数 'GEMINI_API_KEY' が設定されていません。")
    st.stop()
# client の初期化は成功した場合のみ行う
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Gemini Clientの初期化に失敗しました: {e}")
    st.stop()

# --- アプリ設定 ---
st.set_page_config(layout="wide")
st.title("お祭り検索＆訪問記録アプリ")
geolocator = Nominatim(user_agent="japanese_festival_app_v11")

# (parse_event_period, get_lat_lonなどのヘルパー関数は変更なし)
# --- Helper Functions ---
def parse_event_period(period_str):
    try:
        year_match = re.search(r'(\d{4})年', period_str)
        month_match = re.search(r'(\d{1,2})月', period_str)
        if not year_match or not month_match: return None, None
        year, month = int(year_match.group(1)), int(month_match.group(1))
        days = [int(d) for d in re.findall(r'(\d{1,2})日', period_str)]
        if not days: return None, None
        start_date = datetime(year, month, days[0])
        end_date = datetime(year, month, days[-1])
        return start_date, end_date + timedelta(days=1)
    except (ValueError, IndexError):
        return None, None

@st.cache_data(ttl=3600)
def get_lat_lon(address):
    if not address or pd.isna(address): return None, None
    try:
        location = geolocator.geocode(address + ", Japan", timeout=10)
        if location: return location.latitude, location.longitude
        time.sleep(1)
    except Exception as e:
        st.warning(f"ジオコーディング失敗: {address} - {e}")
    return None, None

@st.cache_data(ttl=600)
def get_worksheet_data(sheet_name):
    try:
        ws = get_or_create_worksheet(sheet_name)
        return pd.DataFrame(ws.get_all_records())
    except Exception as e:
        st.error(f"'{sheet_name}'シート読込失敗: {e}")
        return pd.DataFrame()

def get_or_create_worksheet(sheet_name):
    spreadsheet = gc.open_by_key(SPREADSHEET_ID)
    try:
        return spreadsheet.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=20)
        headers = {
            "Festival_Data": ["Festival Name", "Location", "Event Period", "Description", "Recommended Highlights", "Latitude", "Longitude", "Added Date"],
            "Visit_Records": ["Festival Name", "Location", "Visit Date", "Photo Path", "Photo Latitude", "Photo Longitude", "Matched", "Distance_km", "Photo Count"]
        }
        if sheet_name in headers:
            ws.append_row(headers[sheet_name])
        return ws

# (get_gemini_recommendations, add_festivals_to_sheetなどは変更なし)
def get_gemini_recommendations(query):
    """Gemini APIでお祭り情報を取得（緯度経度も含む）"""
    try:
        config_with_search = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            max_output_tokens=4096,
            temperature=0.0,
            top_k=40,
            top_p=1.0,
        )

        prompt = f"""
あなたは日本のお祭りを検索する優秀なAIです。
次のリクエスト "{query}" に合うお祭りをウェブで調べて、**必ず**以下のJSON形式のリストのみを返してください。

重要：各お祭りの正確な住所と緯度経度を調べて含めてください。

他の文章・説明・補足文・装飾・コードブロックなどは**一切追加しないでください**。
見つからない場合は [] だけを返してください。

[
  {{
    "Festival Name": "三社祭",
    "Location": "東京都台東区浅草2-3-1 浅草神社",
    "Event Period": "2025年5月16日～18日",
    "Description": "例大祭で神輿渡御が見どころ",
    "Recommended Highlights": "本社神輿の宮出し・宮入り",
    "Latitude": 35.714844,
    "Longitude": 139.796707
  }}
]
"""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=config_with_search
        )
        
        return response.text

    except Exception as e:
        st.error(f"Gemini APIとの通信中にエラーが発生しました: {e}")
        return None


def add_festivals_to_sheet(festivals_df):
    """お祭りデータをシートに追加する関数"""
    ws = get_or_create_worksheet("Festival_Data")
    existing_df = get_worksheet_data("Festival_Data")
    new_rows = []
    
    for _, row in festivals_df.iterrows():
        is_duplicate = False
        if not existing_df.empty:
            is_duplicate = ((existing_df['Festival Name'] == row['Festival Name']) & 
                            (existing_df['Event Period'] == row['Event Period'])).any()
        
        if not is_duplicate:
            lat, lon = row.get('Latitude'), row.get('Longitude')
            if pd.isna(lat) or pd.isna(lon) or not lat or not lon:
                lat, lon = get_lat_lon(row['Location'])
            
            new_rows.append([
                row['Festival Name'], row['Location'], row['Event Period'],
                row['Description'], row.get('Recommended Highlights', ''),
                lat or "", lon or "", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])
            
    if new_rows:
        ws.append_rows(new_rows)
        st.success(f"{len(new_rows)}件の新しいお祭りを追加しました。")
        st.cache_data.clear() # キャッシュをクリア
        st.rerun() # ページを再実行して即時反映
    else:
        st.info("追加する新しいお祭りはありませんでした（重複はスキップ）。")


# (get_exif_data, get_gps_from_exif, haversine_distanceは変更なし)
def get_exif_data(image):
    exif_data = {}
    try:
        info = image.getexif()
        if not info: return {}
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            exif_data[decoded] = value
        gps_info_raw = info.get_ifd(0x8825)
        if gps_info_raw:
            exif_data['GPSInfo'] = {}
            for key, val in gps_info_raw.items():
                decode = GPSTAGS.get(key, key)
                exif_data['GPSInfo'][decode] = val
    except Exception: pass
    return exif_data
def get_gps_from_exif(exif_data):
    try:
        gps_info = exif_data.get('GPSInfo')
        if not gps_info: return None, None
        def convert_to_degrees(value):
            if isinstance(value, (list, tuple)) and len(value) == 3: return float(value[0]) + (float(value[1]) / 60.0) + (float(value[2]) / 3600.0)
            return None
        lat_data, lon_data = gps_info.get('GPSLatitude'), gps_info.get('GPSLongitude')
        lat_ref, lon_ref = gps_info.get('GPSLatitudeRef'), gps_info.get('GPSLongitudeRef')
        if lat_data and lon_data and lat_ref and lon_ref:
            lat, lon = convert_to_degrees(lat_data), convert_to_degrees(lon_data)
            if lat is None or lon is None: return None, None
            if lat_ref == 'S': lat = -lat
            if lon_ref == 'W': lon = -lon
            return lat, lon
    except Exception: pass
    return None, None
def haversine_distance(lat1, lon1, lat2, lon2):
    R, lat1_rad, lon1_rad, lat2_rad, lon2_rad = 6371.0, math.radians(lat1), math.radians(lon1), math.radians(lat2), math.radians(lon2)
    dlon, dlat = lon2_rad - lon1_rad, lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# (find_closest_festivalは変更なし)
def find_closest_festival(photo_lat, photo_lon, photo_date, festival_data_df):
    min_distance = float('inf')
    closest = None
    for _, festival in festival_data_df.iterrows():
        try:
            fest_lat, fest_lon = float(festival.get('Latitude')), float(festival.get('Longitude'))
            distance = haversine_distance(photo_lat, photo_lon, fest_lat, fest_lon)
            if distance < min_distance and distance < 2.0:
                period_str = festival.get('Event Period')
                if period_str:
                    start_date, end_date = parse_event_period(period_str)
                    if start_date and end_date and start_date <= photo_date < end_date:
                        min_distance = distance
                        closest = {'name': festival['Festival Name'],'location': festival['Location'],'distance': round(distance, 2)}
        except (ValueError, TypeError, AttributeError): continue
    return closest

def process_drive_photos():
    """【改善版】写真を訪問グループごとにまとめて記録する"""
    try:
        # 1. 未処理の写真ファイルを特定
        visit_df = get_worksheet_data("Visit_Records")
        processed_photos = set(visit_df['Photo Path']) if not visit_df.empty else set()
        results = drive_service.files().list(q=f"'{DRIVE_FOLDER_ID}' in parents and mimeType contains 'image/'", fields="files(id, name)").execute()
        all_files = results.get('files', [])
        files_to_process = [f for f in all_files if f['name'] not in processed_photos]
        
        if not files_to_process:
            st.info("処理対象の新しい写真はありませんでした。")
            return []

        st.info(f"Google Driveで{len(all_files)}個の写真を発見。うち{len(files_to_process)}個が未処理です。")
        
        # 2. 未処理写真のEXIF情報を一括で抽出
        photo_details = []
        progress_bar = st.progress(0, text="写真の情報をスキャン中...")
        for i, file in enumerate(files_to_process):
            try:
                file_content = drive_service.files().get_media(fileId=file['id']).execute()
                image = Image.open(io.BytesIO(file_content))
                exif_data = get_exif_data(image)
                lat, lon = get_gps_from_exif(exif_data)
                if lat and lon:
                    photo_date_str = exif_data.get('DateTime', datetime.now().strftime("%Y:%m:%d %H:%M:%S"))
                    photo_details.append({
                        'file_name': file['name'],
                        'timestamp': datetime.strptime(photo_date_str, "%Y:%m:%d %H:%M:%S"),
                        'lat': lat, 'lon': lon
                    })
            except Exception as e:
                st.warning(f"ファイル解析エラー ({file['name']}): {e}")
            progress_bar.progress((i + 1) / len(files_to_process))
        
        if not photo_details:
            st.info("GPS情報付きの新しい写真が見つかりませんでした。")
            return []

        # 3. 写真を「訪問グループ」に分類
        visit_groups = {}
        festival_df = get_worksheet_data("Festival_Data")
        for photo in photo_details:
            visit_date_str = photo['timestamp'].strftime("%Y-%m-%d")
            closest_festival = find_closest_festival(photo['lat'], photo['lon'], photo['timestamp'], festival_df)
            
            group_key = (closest_festival['name'], visit_date_str) if closest_festival else ('GPS記録のみ', visit_date_str)
            
            if group_key not in visit_groups:
                visit_groups[group_key] = {'festival_info': closest_festival, 'photos': []}
            visit_groups[group_key]['photos'].append(photo)

        # 4. 各グループから代表記録を作成
        new_visit_records = []
        for group_key, group_data in visit_groups.items():
            photos_in_group = group_data['photos']
            festival_info = group_data['festival_info']
            
            # 最も早い写真を代表として選択
            earliest_photo = min(photos_in_group, key=lambda p: p['timestamp'])
            
            if festival_info:
                record = {
                    'Festival Name': festival_info['name'], 'Location': festival_info['location'],
                    'Visit Date': group_key[1], 'Photo Path': earliest_photo['file_name'],
                    'Photo Latitude': earliest_photo['lat'], 'Photo Longitude': earliest_photo['lon'],
                    'Matched': 'Yes', 'Distance_km': festival_info['distance'],
                    'Photo Count': len(photos_in_group)
                }
            else: # GPS記録のみの場合
                record = {
                    'Festival Name': 'GPS記録のみ', 'Location': f"緯度{earliest_photo['lat']:.4f}, 経度{earliest_photo['lon']:.4f}",
                    'Visit Date': group_key[1], 'Photo Path': earliest_photo['file_name'],
                    'Photo Latitude': earliest_photo['lat'], 'Photo Longitude': earliest_photo['lon'],
                    'Matched': 'No', 'Distance_km': None,
                    'Photo Count': len(photos_in_group)
                }
            new_visit_records.append(record)

        # 5. 新しい訪問記録をシートに一括保存
        if new_visit_records:
            visit_ws = get_or_create_worksheet("Visit_Records")
            headers = visit_ws.row_values(1)
            rows_to_append = [[r.get(h, '') for h in headers] for r in new_visit_records]
            visit_ws.append_rows(rows_to_append)
            total_photos_processed = sum(r['Photo Count'] for r in new_visit_records)
            st.success(f"{len(new_visit_records)}件の新しい訪問（計{total_photos_processed}枚の写真）を記録しました。")
            st.cache_data.clear() # キャッシュクリア
        else:
            st.info("記録する新しい訪問はありませんでした。")
            
        return new_visit_records

    except Exception as e:
        st.error(f"Google Driveの写真処理中にエラー: {e}")
        return []

# (地図と統計情報の表示部分は変更なし)
# --- 地図と統計情報の表示 ---
def create_map(df, lat_col, lon_col, popup_cols, tooltip_col, color):
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.dropna(subset=[lat_col, lon_col])
    if df.empty: return None
    center = [df[lat_col].mean(), df[lon_col].mean()]
    m = folium.Map(location=center, zoom_start=10)
    for _, row in df.iterrows():
        popup_html = "<br>".join([f"<b>{col}:</b> {row[col]}" for col in popup_cols if col in row])
        folium.Marker([row[lat_col], row[lon_col]], popup=popup_html, tooltip=row[tooltip_col], icon=folium.Icon(color=color, icon='star')).add_to(m)
    return m

def create_visit_heatmap(visit_df):
    visit_df = visit_df.dropna(subset=['Photo Latitude', 'Photo Longitude']).copy()
    visit_df['Photo Latitude'] = pd.to_numeric(visit_df['Photo Latitude'], errors='coerce')
    visit_df['Photo Longitude'] = pd.to_numeric(visit_df['Photo Longitude'], errors='coerce')
    visit_df = visit_df.dropna(subset=['Photo Latitude', 'Photo Longitude'])
    if visit_df.empty: return None, 0
    heat_data = [[row['Photo Latitude'], row['Photo Longitude']] for _, row in visit_df.iterrows()]
    center = [visit_df['Photo Latitude'].mean(), visit_df['Photo Longitude'].mean()]
    m = folium.Map(location=center, zoom_start=10)
    HeatMap(heat_data, radius=15).add_to(m)
    return m, len(heat_data)

def display_visit_statistics(festival_df, visit_df):
    """【修正】訪問状況の統計情報を4列で表示し、訪問率を復活"""
    st.subheader("📊 訪問状況サマリー")
    total_festivals = len(festival_df)
    if visit_df.empty:
        visited_count, total_photos = 0, 0
    else:
        visited_festivals = visit_df[visit_df['Matched'] == 'Yes']
        visited_count = visited_festivals['Festival Name'].nunique() if not visited_festivals.empty else 0
        total_photos = pd.to_numeric(visit_df['Photo Count'], errors='coerce').sum()
    visit_rate = (visited_count / total_festivals * 100) if total_festivals > 0 else 0
    
    # ★修正点: 列を4つにし、訪問率のメトリックを再追加
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("総お祭り登録数", f"{total_festivals} 件")
    col2.metric("訪問済みお祭り数", f"{visited_count} 件", help="マッチングに成功したユニークなお祭りの数")
    col3.metric("お祭り訪問率", f"{visit_rate:.1f} %")
    col4.metric("総撮影枚数", f"{int(total_photos)} 枚", help="訪問記録にある写真の合計枚数")


# --- Streamlit UI ---
st.header("📋 登録済みお祭り一覧")
festival_df = get_worksheet_data("Festival_Data")
if not festival_df.empty:
    st.dataframe(festival_df, use_container_width=True)
else:
    st.info("まだお祭りが登録されていません。")

st.header("🎌 お祭りを探す")
query = st.text_input("キーワードを入力（例：東京 7月、神田祭）")
if st.button("🔍 検索"):
    if query:
        with st.spinner("Geminiがお祭りを検索中..."):
            raw_response = get_gemini_recommendations(query)
            try:
                json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
                if json_match:
                    festivals = json.loads(json_match.group(0))
                    if festivals:
                        add_festivals_to_sheet(pd.DataFrame(festivals))
                    else:
                        st.info("該当するお祭りが見つかりませんでした。")
                else:
                    st.error("検索結果の形式が不正です。")
            except json.JSONDecodeError:
                st.error("検索結果の解析に失敗しました。")
                st.code(raw_response)
    else:
        st.warning("検索キーワードを入力してください。")

st.header("🗺️ お祭りマップ")
if not festival_df.empty:
    festival_map = create_map(festival_df, 'Latitude', 'Longitude', ['Festival Name', 'Location', 'Event Period'], 'Festival Name', 'red')
    if festival_map:
        st.components.v1.html(festival_map.get_root().render(), height=500)
    else:
        st.warning("地図に表示できるお祭りがありません。")
else:
    st.info("表示するお祭りデータがありません。")

st.header("📝 訪問記録")
if st.button("📷 Google Driveの写真を処理して訪問記録を更新"):
    process_drive_photos()
    st.rerun() # 処理後にページを再読み込みして最新の記録と統計を表示

visit_df = get_worksheet_data("Visit_Records")
if not visit_df.empty:
    # ★★★ エラー対策: Distance_km列を強制的に数値に変換 ★★★
    # 空文字''など、数値にできない値はNaN（Not a Number）に変換され、エラーを防ぎます。
    visit_df['Distance_km'] = pd.to_numeric(visit_df['Distance_km'], errors='coerce')

    st.dataframe(visit_df.style.format({
        'Photo Latitude': '{:.6f}',
        'Photo Longitude': '{:.6f}',
        'Distance_km': '{:.2f}' # 距離も小数点2桁にフォーマット
    }), use_container_width=True)
else:
    st.info("まだ訪問記録がありません。")

st.header("🔥 訪問済みエリアのヒートマップ")
display_visit_statistics(festival_df, visit_df)
hmap, count = create_visit_heatmap(visit_df)
if hmap:
    st.success(f"全 {count} 地点の訪問記録からヒートマップを生成しました。")
    st.components.v1.html(hmap.get_root().render(), height=500)
else:
    st.info("ヒートマップを生成するための訪問記録がありません。")