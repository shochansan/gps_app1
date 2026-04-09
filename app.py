import io
import re
import zipfile
import tempfile
from pathlib import Path
import datetime
import unicodedata
import math
from typing import Optional, List, Dict, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import streamlit as st
from numbers_parser import Document

APP_VERSION = "2026-04-09_session_position_boxbooklet_v5"

POSITION_ORDER = ["GK", "CB", "SB", "MF", "SH", "FW"]

POINT_ALPHA = 0.25
LABEL_ALPHA = 0.55
POINT_SIZE = 22
LABEL_FONTSIZE = 7
MEDIAN_FONTSIZE = 8
JITTER_WIDTH = 0.16
SHOW_VALUE_IN_LABEL = False

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".numbers"}
STREAMLIT_FILE_TYPES = ["csv", "xlsx", "xls", "numbers"]

_ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff"]


def configure_matplotlib_japanese_font():
    try:
        import matplotlib
        import matplotlib.font_manager as fm
        import subprocess

        # Linux環境でfclistから日本語フォントパスを取得
        try:
            result = subprocess.run(
                ["fc-list", ":lang=ja", "--format=%{file}\n"],
                capture_output=True, text=True, timeout=10
            )
            font_files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if font_files:
                fm.fontManager.addfont(font_files[0])
                prop = fm.FontProperties(fname=font_files[0])
                matplotlib.rcParams["font.family"] = prop.get_name()
                matplotlib.rcParams["axes.unicode_minus"] = False
                matplotlib.rcParams["pdf.fonttype"] = 42
                matplotlib.rcParams["ps.fonttype"] = 42
                return
        except Exception:
            pass

        # フォールバック：フォント名で検索
        preferred = [
            "IPAexGothic", "IPAGothic",
            "Noto Sans CJK JP", "Noto Sans JP",
            "Hiragino Sans", "Hiragino Kaku Gothic ProN",
            "Yu Gothic", "YuGothic",
            "MS Gothic", "Meiryo",
            "TakaoGothic",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        for name in preferred:
            if name in available:
                matplotlib.rcParams["font.family"] = name
                break

        matplotlib.rcParams["axes.unicode_minus"] = False
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
    except Exception:
        pass


configure_matplotlib_japanese_font()




# =========================
# Utilities
# =========================
def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(s))


def norm_colname(s: str) -> str:
    s = str(s)

    for zw in _ZERO_WIDTH:
        s = s.replace(zw, "")

    s = s.replace("\u00A0", " ").replace("　", " ").strip()
    s = s.replace("（", "(").replace("）", ")").replace("％", "%")
    s = s.replace("－", "-").replace("ー", "-")
    s = unicodedata.normalize("NFKC", s)

    s = s.replace("/", "").replace("\\", "")
    s = s.replace("_", "").replace("-", "")
    s = s.replace(">", "").replace("{", "").replace("}", "")
    s = re.sub(r"\s+", "", s)

    return s.lower()


def file_ext_from_name(name: str) -> str:
    return Path(str(name)).suffix.lower()


def validate_file_name(name: str, label: str):
    ext = file_ext_from_name(name)
    if ext not in ALLOWED_EXTENSIONS:
        st.error(f"❌ {label} のファイル形式が未対応です: {name}")
        st.info("対応形式: csv / xlsx / xls / numbers")
        st.stop()


def validate_uploaded_file(uploaded_file, label: str):
    if uploaded_file is None:
        return
    validate_file_name(uploaded_file.name, label)


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    norm_to_actual = {}

    for c in cols:
        nc = norm_colname(c)
        if nc not in norm_to_actual:
            norm_to_actual[nc] = c

    for cand in candidates:
        if cand in df.columns:
            return cand

        nc = norm_colname(cand)
        if nc in norm_to_actual:
            return norm_to_actual[nc]

    return None


def require_cols(df: pd.DataFrame, needed: Dict[str, List[str]], label: str) -> Dict[str, str]:
    mapping = {}
    missing = []

    for logical, cands in needed.items():
        found = find_col(df, cands)
        if not found:
            missing.append((logical, cands))
        else:
            mapping[logical] = found

    if missing:
        msg = [f"❌ {label} に必要な列が見つかりません。"]
        for logical, cands in missing:
            msg.append(f"- 必要項目: {logical} / 候補列名: {cands}")
        st.error("\n".join(msg))
        st.stop()

    return mapping


def normalize_session_name(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass

    s = str(x)
    s = s.replace("\u00A0", " ").replace("　", " ")
    for zw in _ZERO_WIDTH:
        s = s.replace(zw, "")
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("＋", "+").replace("−", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_position(x) -> str:
    if x is None:
        return "Unknown"
    try:
        if isinstance(x, float) and np.isnan(x):
            return "Unknown"
    except Exception:
        pass

    s = str(x).replace("\u00A0", " ").replace("　", " ").strip()
    for zw in _ZERO_WIDTH:
        s = s.replace(zw, "")
    if s == "" or s.lower() == "nan":
        return "Unknown"
    s = unicodedata.normalize("NFKC", s).strip()
    return s.upper()


def normalize_person_key(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass

    s = str(x)
    for zw in _ZERO_WIDTH:
        s = s.replace(zw, "")
    s = s.replace("\u00A0", "").replace("　", "").replace(" ", "")
    s = unicodedata.normalize("NFKC", s)
    return s.strip()


def sort_positions_fixed(pos_list: List[str]) -> List[str]:
    seen = []
    for p in pos_list:
        p2 = normalize_position(p)
        if p2 not in seen:
            seen.append(p2)

    order_set = set(POSITION_ORDER)
    in_order = [p for p in POSITION_ORDER if p in seen]
    rest = [p for p in seen if p not in order_set and p != "Unknown"]
    rest_sorted = sorted(rest)

    if "Unknown" in seen:
        return in_order + rest_sorted + ["Unknown"]
    return in_order + rest_sorted


def get_normalized_position_series(df: pd.DataFrame, col_name: str = "Position") -> pd.Series:
    if col_name in df.columns:
        s = df[col_name]
        if isinstance(s, pd.DataFrame):
            if s.shape[1] == 0:
                return pd.Series(["Unknown"] * len(df), index=df.index)
            s = s.iloc[:, 0]
        return s.map(normalize_position)
    return pd.Series(["Unknown"] * len(df), index=df.index)


def get_text_series(
    df: pd.DataFrame,
    primary: str,
    fallbacks: Optional[List[str]] = None,
    default: str = ""
) -> pd.Series:
    fallbacks = fallbacks or []
    candidates = [primary] + fallbacks
    for c in candidates:
        if c in df.columns:
            s = df[c]
            if isinstance(s, pd.DataFrame):
                if s.shape[1] == 0:
                    continue
                s = s.iloc[:, 0]
            return s.fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype="object")


# =========================
# 重複列対策
# =========================
def force_series(obj, index=None) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj

    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(np.nan, index=obj.index if index is None else index)
        return obj.bfill(axis=1).iloc[:, 0]

    if obj is None:
        return pd.Series(np.nan, index=index)

    try:
        return pd.Series(obj, index=index)
    except Exception:
        if index is not None:
            return pd.Series([obj] * len(index), index=index)
        return pd.Series([obj])


def force_numeric_series(obj, index=None) -> pd.Series:
    s = force_series(obj, index=index)
    return pd.to_numeric(s, errors="coerce")


def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    done = set()

    for col in df.columns:
        if col in done:
            continue

        same = df.loc[:, df.columns == col]

        if same.shape[1] == 1:
            out[col] = same.iloc[:, 0]
        else:
            tmp = same.bfill(axis=1)
            out[col] = tmp.iloc[:, 0]

        done.add(col)

    return out


# =========================
# Y軸レンジ制御
# =========================
def _nice_step(value: float) -> float:
    if value <= 0:
        return 10.0
    if value <= 100:
        return 10.0
    if value <= 200:
        return 20.0
    if value <= 500:
        return 50.0
    return 100.0


def _compute_dynamic_ylim(y_values: Union[pd.Series, np.ndarray, list]) -> Tuple[float, float]:
    y = pd.to_numeric(pd.Series(y_values), errors="coerce").dropna()
    if y.empty:
        return (0.0, 100.0)

    y_min = float(y.min())
    y_max = float(y.max())

    lower = 0.0
    upper = 100.0

    if y_min < 0:
        pad_low = max(1.0, abs(y_min) * 0.05)
        lower = y_min - pad_low
        step_low = _nice_step(abs(lower))
        lower = math.floor(lower / step_low) * step_low

    if y_max > 100:
        pad_high = max(1.0, abs(y_max) * 0.05)
        upper = y_max + pad_high
        step_high = _nice_step(upper)
        upper = math.ceil(upper / step_high) * step_high

    if upper <= lower:
        upper = lower + 10.0

    return (lower, upper)


# =========================
# Duration parsing
# =========================
def _normalize_time_text(x: str) -> str:
    if x is None:
        return ""
    x = str(x).replace("\u00A0", " ").replace("　", " ").strip()

    trans = str.maketrans({
        "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
        "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
        "：": ":"
    })
    x = x.translate(trans).strip()
    x = re.sub(r"^\d{4}[-/]\d{2}[-/]\d{2}\s+", "", x)
    return x


def _cell_to_minutes(x) -> float:
    if x is None:
        return np.nan
    try:
        if isinstance(x, float) and np.isnan(x):
            return np.nan
    except Exception:
        pass

    if isinstance(x, datetime.timedelta):
        return x.total_seconds() / 60.0
    if isinstance(x, datetime.time):
        return (x.hour * 3600 + x.minute * 60 + x.second) / 60.0
    if isinstance(x, datetime.datetime):
        return (x.hour * 3600 + x.minute * 60 + x.second) / 60.0
    if isinstance(x, pd.Timestamp):
        return (x.hour * 3600 + x.minute * 60 + x.second) / 60.0

    try:
        if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
            v = float(x)
            if 0 < v < 1.5:
                return v * 24 * 60
            if v >= 300:
                return v / 60.0
            return v
    except Exception:
        pass

    s = _normalize_time_text(x)
    if s == "" or s.lower() == "nan":
        return np.nan

    td = pd.to_timedelta(s, errors="coerce")
    if pd.notna(td):
        return td.total_seconds() / 60.0

    parts = s.split(":")
    try:
        if len(parts) == 3:
            h, m, sec = parts
            return (int(h) * 3600 + int(m) * 60 + int(sec)) / 60.0
        if len(parts) == 2:
            a, b = parts
            return (int(a) * 60 + int(b)) / 60.0
    except Exception:
        return np.nan

    return np.nan


def parse_duration_to_min(series: pd.Series) -> pd.Series:
    out = series.map(_cell_to_minutes)
    return pd.to_numeric(out, errors="coerce")


# =========================
# Numbers 読み込み
# =========================
def _numbers_bytes_to_matrix(data_bytes: bytes):
    tmp_path = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".numbers") as tmp:
        tmp.write(data_bytes)
        tmp_path = tmp.name

    try:
        doc = Document(tmp_path)
        sheet = doc.sheets[0]
        table = sheet.tables[0]

        def cell_to_value(cell):
            v = getattr(cell, "value", None)
            if v is None or v == "":
                fv = getattr(cell, "formatted_value", None)
                if fv is not None and str(fv).strip() != "":
                    return fv
            return v

        data = [[cell_to_value(cell) for cell in row] for row in table.rows()]
        return data
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def read_numbers_bytes_to_df(data_bytes: bytes, header_row: Optional[int] = 0) -> pd.DataFrame:
    data = _numbers_bytes_to_matrix(data_bytes)
    if not data:
        return pd.DataFrame()

    if header_row is None:
        return pd.DataFrame(data)

    if header_row >= len(data):
        return pd.DataFrame()

    header = [str(x).strip() if x is not None else "" for x in data[header_row]]
    rows = data[header_row + 1:]
    df = pd.DataFrame(rows, columns=header)
    df.columns = [str(c).strip() for c in df.columns]
    keep_cols = [c for c in df.columns if c != ""]
    return df.loc[:, keep_cols]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_any_table_from_bytes(file_name: str, data_bytes: bytes) -> pd.DataFrame:
    ext = file_ext_from_name(file_name)

    if ext == ".csv":
        last_error = None
        for enc in ["utf-8-sig", "utf-8", "cp932"]:
            try:
                return _clean_columns(pd.read_csv(io.BytesIO(data_bytes), encoding=enc))
            except Exception as e:
                last_error = e
        raise last_error

    if ext in {".xlsx", ".xls"}:
        return _clean_columns(pd.read_excel(io.BytesIO(data_bytes)))

    if ext == ".numbers":
        return _clean_columns(read_numbers_bytes_to_df(data_bytes, header_row=0))

    raise ValueError(f"未対応のファイル形式: {file_name}")


# =========================
# 試合平均：ヘッダー行を自動検出して読む
# =========================
def detect_header_row(df_raw: pd.DataFrame, required_tokens: List[str], scan_rows: int = 80) -> Optional[int]:
    n = min(len(df_raw), scan_rows)
    tokens_norm = [norm_colname(t) for t in required_tokens]

    best_row = None
    best_hit = 0
    for r in range(n):
        row_vals = df_raw.iloc[r].tolist()
        row_texts = [norm_colname(v) for v in row_vals if v is not None and str(v).strip() != ""]
        hit = sum(1 for t in tokens_norm if t in row_texts)
        if hit > best_hit:
            best_hit = hit
            best_row = r

    if best_row is not None and best_hit >= max(2, int(len(tokens_norm) * 0.3)):
        return best_row
    return None


def _game_required_tokens() -> List[str]:
    return [
        "Name", "名前",
        "Distance", "Distance_90",
        "SI_D", "SI D", "SID", "SI_D_90",
        "HI_D", "HI D", "HID", "HI_D_90",
        "SPD_Max", "SPD MX", "SPD_MX", "Max Speed",
        "Accel", "Accel_Total", "Accel_Total_90",
        "Decel", "Decel_Total", "Decel_Total_90",
        "High Agility", "HighAgility", "High Agility_90", "HighAgility_90", "High Agility90", "HighAgility90",
        "Distance_per_min", "Distance per min", "dis(m)/min",
        "HighAgility_per_min", "High Agility_per_min", "High Agility per min", "High Agility/min(n/min)",
    ]


def read_game_table_any_header_from_bytes(file_name: str, data_bytes: bytes) -> pd.DataFrame:
    ext = file_ext_from_name(file_name)
    required = _game_required_tokens()

    if ext == ".csv":
        last_error = None
        for enc in ["utf-8-sig", "utf-8", "cp932"]:
            try:
                raw = pd.read_csv(io.BytesIO(data_bytes), header=None, encoding=enc)
                h = detect_header_row(raw, required_tokens=required)
                if h is None:
                    return _clean_columns(pd.read_csv(io.BytesIO(data_bytes), encoding=enc))
                return _clean_columns(pd.read_csv(io.BytesIO(data_bytes), header=h, encoding=enc))
            except Exception as e:
                last_error = e
        raise last_error

    if ext in {".xlsx", ".xls"}:
        raw = pd.read_excel(io.BytesIO(data_bytes), header=None)
        h = detect_header_row(raw, required_tokens=required)
        if h is None:
            return _clean_columns(pd.read_excel(io.BytesIO(data_bytes)))
        return _clean_columns(pd.read_excel(io.BytesIO(data_bytes), header=h))

    if ext == ".numbers":
        raw = read_numbers_bytes_to_df(data_bytes, header_row=None)
        h = detect_header_row(raw, required_tokens=required)
        if h is None:
            h = 0
        return _clean_columns(read_numbers_bytes_to_df(data_bytes, header_row=h))

    raise ValueError(f"未対応のファイル形式: {file_name}")


# =========================
# Core calculations
# =========================
def build_session_metrics(train_df: pd.DataFrame) -> pd.DataFrame:
    needed_train = {
        "Name": ["Name", "名前"],
        "Session": ["Session", "セッション"],
        "Duration_TF": ["Duration_TF", "Duration", "時間", "Duration (TF)"],
        "Distance": ["Distance", "距離"],
        "SI_D": ["SI_D", "SI D", "SID"],
        "HI_D": ["HI_D", "HI D", "HID"],
        "Sprint": ["Sprint", "スプリント"],
        "Accel_Z2": ["Accel_Z2", "Accel Z2"],
        "Accel_Z3": ["Accel_Z3", "Accel Z3"],
        "Decel_Z2": ["Decel_Z2", "Decel Z2"],
        "Decel_Z3": ["Decel_Z3", "Decel Z3"],
        "SPD_MX": ["SPD MX", "SPD_MX", "SPD_Max", "Max Speed", "最高速度"],
    }
    m = require_cols(train_df, needed_train, "練習データ")

    pos_col = find_col(train_df, ["Position", "Pos", "ポジション", "POS", "position", "P"])

    df = train_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    df[m["Name"]] = df[m["Name"]].astype(str)
    df[m["Session"]] = df[m["Session"]].map(normalize_session_name)

    if pos_col:
        df[pos_col] = df[pos_col].map(normalize_position)
    else:
        df["Position__tmp"] = "Unknown"
        pos_col = "Position__tmp"

    df["duration_min"] = parse_duration_to_min(df[m["Duration_TF"]])
    if df["duration_min"].dropna().empty:
        st.error("Duration_TF を分に変換できませんでした。Duration_TF列の形式を確認してください。")
        st.stop()

    for key in ["Distance", "SI_D", "HI_D", "Sprint", "Accel_Z2", "Accel_Z3", "Decel_Z2", "Decel_Z3", "SPD_MX"]:
        df[m[key]] = pd.to_numeric(df[m[key]], errors="coerce")

    def sum_min1(s: pd.Series) -> float:
        s2 = pd.to_numeric(s, errors="coerce")
        return s2.sum() if s2.notna().any() else np.nan

    def first_nonnull_pos(x: pd.Series):
        x2 = x.dropna()
        if len(x2) == 0:
            return "Unknown"
        return normalize_position(x2.iloc[0])

    g = df.groupby([m["Name"], m["Session"]], as_index=False)

    session = g.agg(**{
        "Position": (pos_col, first_nonnull_pos),
        "duration_min": ("duration_min", sum_min1),
        "Distance": (m["Distance"], "sum"),
        "SI_D": (m["SI_D"], "sum"),
        "HI_D": (m["HI_D"], "sum"),
        "Sprint": (m["Sprint"], "sum"),
        "Accel_Z2": (m["Accel_Z2"], "sum"),
        "Accel_Z3": (m["Accel_Z3"], "sum"),
        "Decel_Z2": (m["Decel_Z2"], "sum"),
        "Decel_Z3": (m["Decel_Z3"], "sum"),
        "SPD MX": (m["SPD_MX"], "max"),
    })

    session = session.rename(columns={m["Name"]: "Name", m["Session"]: "Session"})
    session["Name"] = session["Name"].astype(str)
    session["Session"] = session["Session"].map(normalize_session_name)
    session["Name_key"] = session["Name"].map(normalize_person_key)
    session["Position"] = session["Position"].map(normalize_position)

    dur = pd.to_numeric(session["duration_min"], errors="coerce").replace(0, np.nan)
    dist = pd.to_numeric(session["Distance"], errors="coerce").replace(0, np.nan)

    session["dis(m)/min"] = session["Distance"] / dur
    session["High Agility"] = session["Accel_Z3"] + session["Decel_Z3"]
    session["Accel  >2m/s/s(n)"] = session["Accel_Z2"] + session["Accel_Z3"]
    session["Decel>2m/s/s(n)"] = session["Decel_Z2"] + session["Decel_Z3"]

    session["High Agility/min(n/min)"] = session["High Agility"] / dur
    session["{Accel>2m/s/s}/min(n/min)"] = session["Accel  >2m/s/s(n)"] / dur
    session["{Decel>2m/s/s}/min(n/min)"] = session["Decel>2m/s/s(n)"] / dur

    session["SI D/min(m/min)"] = session["SI_D"] / dur
    session["HI D/min(m/min)"] = session["HI_D"] / dur

    session["SI D/T D(%)"] = (session["SI_D"] / dist) * 100
    session["HI D/T D(%)"] = (session["HI_D"] / dist) * 100
    session["Sprint(n)/min"] = session["Sprint"] / dur

    return session


def normalize_game_df_columns(game_df: pd.DataFrame) -> pd.DataFrame:
    g = game_df.copy()
    g.columns = [str(c).strip() for c in g.columns]

    name_col = find_col(g, ["Name", "名前"])
    if name_col is None:
        name_col = g.columns[0]

    g = g.rename(columns={name_col: "Name"})
    g["Name"] = g["Name"].astype(str)
    g["Name_key"] = g["Name"].map(normalize_person_key)

    game_pos_col = find_col(g, ["Position", "Pos", "ポジション", "POS", "position", "P"])
    if game_pos_col is not None and game_pos_col != "Name":
        g = g.rename(columns={game_pos_col: "Position__game"})
        g["Position__game"] = g["Position__game"].map(normalize_position)

    metric_alias_map = {
        norm_colname("Distance"): "Distance",
        norm_colname("Distance_90"): "Distance",

        norm_colname("SI_D"): "SI_D",
        norm_colname("SI D"): "SI_D",
        norm_colname("SID"): "SI_D",
        norm_colname("SI_D_90"): "SI_D",

        norm_colname("HI_D"): "HI_D",
        norm_colname("HI D"): "HI_D",
        norm_colname("HID"): "HI_D",
        norm_colname("HI_D_90"): "HI_D",

        norm_colname("SPD MX"): "SPD MX",
        norm_colname("SPD_MX"): "SPD MX",
        norm_colname("SPD_Max"): "SPD MX",
        norm_colname("Max Speed"): "SPD MX",

        norm_colname("Accel"): "Accel  >2m/s/s(n)",
        norm_colname("Accel_Total"): "Accel  >2m/s/s(n)",
        norm_colname("Accel_Total_90"): "Accel  >2m/s/s(n)",

        norm_colname("Decel"): "Decel>2m/s/s(n)",
        norm_colname("Decel_Total"): "Decel>2m/s/s(n)",
        norm_colname("Decel_Total_90"): "Decel>2m/s/s(n)",
        norm_colname("Decel_T"): "Decel>2m/s/s(n)",

        norm_colname("High Agility"): "High Agility",
        norm_colname("HighAgility"): "High Agility",
        norm_colname("High Agility_90"): "High Agility",
        norm_colname("HighAgility_90"): "High Agility",
        norm_colname("High Agility90"): "High Agility",
        norm_colname("HighAgility90"): "High Agility",

        norm_colname("dis(m)/min"): "dis(m)/min",
        norm_colname("Distance_per_min"): "dis(m)/min",
        norm_colname("Distance per min"): "dis(m)/min",
        norm_colname("Dist/min"): "dis(m)/min",
        norm_colname("DIST/min"): "dis(m)/min",
        norm_colname("Disper_min"): "dis(m)/min",
        norm_colname("Dis per_min"): "dis(m)/min",

        norm_colname("High Agility/min(n/min)"): "High Agility/min(n/min)",
        norm_colname("HighAgility_per_min"): "High Agility/min(n/min)",
        norm_colname("High Agility_per_min"): "High Agility/min(n/min)",
        norm_colname("High Agility per min"): "High Agility/min(n/min)",

        norm_colname("{Accel>2m/s/s}/min(n/min)"): "{Accel>2m/s/s}/min(n/min)",
        norm_colname("Accel_per_min"): "{Accel>2m/s/s}/min(n/min)",
        norm_colname("Accel Total per min"): "{Accel>2m/s/s}/min(n/min)",

        norm_colname("{Decel>2m/s/s}/min(n/min)"): "{Decel>2m/s/s}/min(n/min)",
        norm_colname("Decel_per_min"): "{Decel>2m/s/s}/min(n/min)",
        norm_colname("Decel Total per min"): "{Decel>2m/s/s}/min(n/min)",

        norm_colname("SI D/min(m/min)"): "SI D/min(m/min)",
        norm_colname("SI_D_per_min"): "SI D/min(m/min)",

        norm_colname("HI D/min(m/min)"): "HI D/min(m/min)",
        norm_colname("HI_D_per_min"): "HI D/min(m/min)",

        norm_colname("SI D/T D(%)"): "SI D/T D(%)",
        norm_colname("HI D/T D(%)"): "HI D/T D(%)",

        norm_colname("Sprint(n)/min"): "Sprint(n)/min",
        norm_colname("Sprint_per_min"): "Sprint(n)/min",
    }

    rename_map = {}
    for c in g.columns:
        if c in ["Name", "Name_key", "Position__game"]:
            continue

        nc = norm_colname(c)
        if nc in metric_alias_map:
            rename_map[c] = f"{metric_alias_map[nc]}__game"

    g = g.rename(columns=rename_map)

    if "High Agility__game" not in g.columns:
        for c in g.columns:
            if c in ["Name", "Name_key", "Position__game"]:
                continue
            nc = norm_colname(c)
            if nc in {
                norm_colname("HighAgility_90"),
                norm_colname("High Agility_90"),
                norm_colname("HighAgility90"),
                norm_colname("High Agility90"),
            }:
                g = g.rename(columns={c: "High Agility__game"})
                break

    return g


def build_ratio_df(session_df: pd.DataFrame, game_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Distance", "SPD MX", "SI_D", "HI_D",
        "dis(m)/min", "High Agility", "Accel  >2m/s/s(n)", "Decel>2m/s/s(n)",
        "High Agility/min(n/min)", "{Accel>2m/s/s}/min(n/min)", "{Decel>2m/s/s}/min(n/min)",
        "SI D/min(m/min)", "HI D/min(m/min)", "SI D/T D(%)", "HI D/T D(%)", "Sprint(n)/min",
    ]

    s = session_df.copy()
    s["Name"] = get_text_series(s, "Name")
    s["Session"] = get_text_series(s, "Session").map(normalize_session_name)
    s["Position"] = get_normalized_position_series(s, "Position")
    s["Name_key"] = get_text_series(s, "Name_key")

    g2 = normalize_game_df_columns(game_df)

    merged = s.merge(g2, on="Name_key", how="left", suffixes=("_x", "_y"))
    merged = collapse_duplicate_columns(merged)

    merged["Name"] = get_text_series(merged, "Name", ["Name_x", "Name_y"])
    merged["Session"] = get_text_series(merged, "Session", ["Session_x", "Session_y"]).map(normalize_session_name)

    pos_train = get_text_series(merged, "Position", ["Position_x"], default="")
    pos_game = get_text_series(merged, "Position__game", ["Position_y"], default="")
    pos_final = pos_train.where(pos_train.astype(str).str.strip().ne(""), pos_game)
    merged["Position"] = pos_final.map(normalize_position)

    for m in metrics:
        gcol = f"{m}__game"

        if m in merged.columns:
            session_series = force_numeric_series(merged[m], index=merged.index)
        else:
            session_series = pd.Series(np.nan, index=merged.index)

        if gcol in merged.columns:
            game_series = force_numeric_series(merged[gcol], index=merged.index).replace(0, np.nan)
            merged[f"{m}_game"] = game_series
            merged[f"{m}_vsGame_pct"] = (session_series / game_series) * 100
        else:
            merged[f"{m}_game"] = np.nan
            merged[f"{m}_vsGame_pct"] = np.nan

    drop_mid = [c for c in merged.columns if c.endswith("__game")]
    merged = merged.drop(columns=drop_mid, errors="ignore")

    keep_first = ["Name", "Name_key", "Session", "Position", "duration_min"]
    ordered_cols = [c for c in keep_first if c in merged.columns]
    other_cols = [c for c in merged.columns if c not in ordered_cols and not c.endswith("_x") and not c.endswith("_y")]
    merged = merged.loc[:, ordered_cols + other_cols]

    return merged


# =========================
# Boxplots出力
# =========================
def _get_session_order(df: pd.DataFrame) -> List[str]:
    return df["Session"].dropna().astype(str).map(normalize_session_name).unique().tolist()


def _is_round_all_session(session_name: str) -> bool:
    return norm_colname(session_name) == norm_colname("ROUND ALL")


def _get_booklet_session_order(df: pd.DataFrame) -> List[str]:
    sessions = _get_session_order(df)
    return [s for s in sessions if not _is_round_all_session(s)]


def _get_box_metric_order(ratio_df: pd.DataFrame) -> List[str]:
    preferred = [
        "Distance",
        "SI_D",
        "HI_D",
        "SPD MX",
        "dis(m)/min",
        "High Agility",
        "Accel  >2m/s/s(n)",
        "Decel>2m/s/s(n)",
        "High Agility/min(n/min)",
        "{Accel>2m/s/s}/min(n/min)",
        "{Decel>2m/s/s}/min(n/min)",
        "SI D/min(m/min)",
        "HI D/min(m/min)",
        "SI D/T D(%)",
        "HI D/T D(%)",
        "Sprint(n)/min",
    ]
    existing = [c.replace("_vsGame_pct", "") for c in ratio_df.columns if c.endswith("_vsGame_pct")]
    ordered = [m for m in preferred if m in existing]
    ordered += [m for m in existing if m not in ordered]
    return ordered


def make_boxplots_png_zip_only(ratio_df: pd.DataFrame):
    ratio_df = ratio_df.copy()
    ratio_df["Name"] = get_text_series(ratio_df, "Name")
    ratio_df["Session"] = get_text_series(ratio_df, "Session").map(normalize_session_name)
    ratio_df["Position"] = get_normalized_position_series(ratio_df, "Position")

    session_order = _get_session_order(ratio_df)
    box_metrics = _get_box_metric_order(ratio_df)

    zip_buf = io.BytesIO()
    zf = zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED)

    rng = np.random.default_rng(0)

    def export_group(df_group: pd.DataFrame, group_label: str, zip_folder: str):
        all_labels = session_order
        all_pos = {s: i + 1 for i, s in enumerate(all_labels)}

        for metric in box_metrics:
            col = f"{metric}_vsGame_pct"
            if col not in df_group.columns:
                continue

            data = []
            positions = []
            for s in all_labels:
                vals = pd.to_numeric(df_group.loc[df_group["Session"] == s, col], errors="coerce").dropna().values
                if len(vals) > 0:
                    data.append(vals)
                    positions.append(all_pos[s])

            fig = plt.figure(figsize=(11, 6))
            ax = plt.gca()

            if len(data) > 0:
                bp = ax.boxplot(
                    data,
                    positions=positions,
                    showfliers=False,
                    patch_artist=True,
                    widths=0.55,
                    boxprops=dict(linewidth=2.2, edgecolor="black"),
                    whiskerprops=dict(linewidth=2.0, color="black"),
                    capprops=dict(linewidth=2.0, color="black"),
                    medianprops=dict(linewidth=2.6, color="black"),
                )
                for b in bp["boxes"]:
                    b.set_facecolor("white")
                    b.set_alpha(1.0)

            for s in all_labels:
                x0 = all_pos[s]
                sub = df_group[df_group["Session"] == s].copy()
                if sub.empty:
                    continue

                yvals = pd.to_numeric(sub[col], errors="coerce")
                names = sub["Name"].fillna("").astype(str).tolist()

                for yv, nm in zip(yvals, names):
                    if pd.isna(yv):
                        continue
                    xj = x0 + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH)
                    ax.scatter([xj], [yv], alpha=POINT_ALPHA, s=POINT_SIZE)
                    label_text = nm
                    if SHOW_VALUE_IN_LABEL:
                        label_text = f"{nm} ({yv:.0f})"
                    ax.text(
                        xj + 0.02, yv, label_text,
                        fontsize=LABEL_FONTSIZE,
                        alpha=LABEL_ALPHA,
                        va="center"
                    )

                med = pd.to_numeric(sub[col], errors="coerce").dropna().median()
                if pd.notna(med):
                    ax.text(
                        x0, med, f"{med:.1f}",
                        fontsize=MEDIAN_FONTSIZE,
                        ha="center", va="bottom"
                    )

            ymin, ymax = _compute_dynamic_ylim(pd.to_numeric(df_group[col], errors="coerce"))
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(range(1, len(all_labels) + 1))
            ax.set_xticklabels(all_labels, rotation=30, ha="right")
            ax.set_ylabel(f"{metric} (% of Game Avg)")
            ax.set_title(f"{group_label} - {metric}")
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            png_name = f"{zip_folder}/{safe_name(group_label)}__{safe_name(metric)}.png"
            zf.writestr(png_name, img_buf.getvalue())

    export_group(ratio_df, "ALL", "ALL")

    positions = sort_positions_fixed(ratio_df["Position"].dropna().astype(str).tolist())
    for pos in positions:
        sub = ratio_df[ratio_df["Position"] == pos].copy()
        if sub.empty:
            continue
        export_group(sub, pos, f"Position_{safe_name(pos)}")

    zf.close()
    zip_buf.seek(0)
    return zip_buf


def make_session_boxplot_booklet_pdf(ratio_df: pd.DataFrame) -> io.BytesIO:
    ratio_df = ratio_df.copy()
    ratio_df["Name"] = get_text_series(ratio_df, "Name")
    ratio_df["Session"] = get_text_series(ratio_df, "Session").map(normalize_session_name)
    ratio_df["Position"] = get_normalized_position_series(ratio_df, "Position")

    ratio_df = ratio_df[~ratio_df["Session"].map(_is_round_all_session)].copy()

    session_order = _get_booklet_session_order(ratio_df)
    box_metrics = _get_box_metric_order(ratio_df)
    positions = sort_positions_fixed(ratio_df["Position"].dropna().astype(str).tolist())

    page_specs = []
    for session_name in session_order:
        sub = ratio_df[ratio_df["Session"] == session_name].copy()
        if not sub.empty:
            page_specs.append({
                "kind": "all_graph",
                "session": session_name,
                "position": None,
                "title": f"Session: {session_name} / ALL",
                "toc_title": f"Session: {session_name} / ALL（グラフ）",
            })
            page_specs.append({
                "kind": "all_stats",
                "session": session_name,
                "position": None,
                "title": f"Session: {session_name} / ALL / Statistics",
                "toc_title": f"Session: {session_name} / ALL（統計表）",
            })

    for session_name in session_order:
        sub = ratio_df[ratio_df["Session"] == session_name].copy()
        for pos in positions:
            sub_pos = sub[sub["Position"] == pos].copy()
            if sub_pos.empty:
                continue
            page_specs.append({
                "kind": "position",
                "session": session_name,
                "position": pos,
                "title": f"Session: {session_name} / {pos}",
                "toc_title": f"Session: {session_name} / {pos}",
            })

    BOOKLET_FIGSIZE = (13.0, 7.8)

    def _draw_page_number(fig, page_no: int):
        fig.text(0.5, 0.018, str(page_no), ha="center", va="center", fontsize=10)

    def _box_stats(series: pd.Series) -> Dict[str, Union[int, float, str]]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return {
                "n": 0,
                "median": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "whis_low": np.nan,
                "whis_high": np.nan,
            }

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        within = s[(s >= lower_fence) & (s <= upper_fence)]
        if within.empty:
            whis_low = float(s.min())
            whis_high = float(s.max())
        else:
            whis_low = float(within.min())
            whis_high = float(within.max())

        return {
            "n": int(s.shape[0]),
            "median": float(s.median()),
            "q1": q1,
            "q3": q3,
            "whis_low": whis_low,
            "whis_high": whis_high,
        }

    def _format_stat_cell(series: pd.Series) -> str:
        stt = _box_stats(series)
        if stt["n"] == 0:
            return "-"
        return (
            f"N={stt['n']}\n"
            f"Med {stt['median']:.1f}\n"
            f"Q1-Q3 {stt['q1']:.1f}-{stt['q3']:.1f}\n"
            f"Whis {stt['whis_low']:.1f}-{stt['whis_high']:.1f}"
        )

    def _build_position_stats_table(df_page: pd.DataFrame, valid_metrics: List[str]):
        pos_cols = ["ALL"] + [p for p in positions if not df_page[df_page["Position"] == p].empty]
        col_labels = ["Metric"] + pos_cols
        cell_text = []
        for metric in valid_metrics:
            col = f"{metric}_vsGame_pct"
            row = [metric]
            for pos in pos_cols:
                sub = df_page if pos == "ALL" else df_page[df_page["Position"] == pos]
                row.append(_format_stat_cell(sub[col] if col in sub.columns else pd.Series(dtype=float)))
            cell_text.append(row)
        return col_labels, cell_text

    def _collect_valid_metrics(df_page: pd.DataFrame):
        valid_metrics = []
        data = []
        page_values = []
        for metric in box_metrics:
            col = f"{metric}_vsGame_pct"
            if col not in df_page.columns:
                continue
            vals = pd.to_numeric(df_page[col], errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            valid_metrics.append(metric)
            data.append(vals)
            page_values.extend([float(v) for v in vals])
        return valid_metrics, data, page_values

    def _draw_boxplot_page(df_page: pd.DataFrame, title_text: str, page_no: int, rng) -> plt.Figure:
        valid_metrics, data, page_values = _collect_valid_metrics(df_page)

        fig = plt.figure(figsize=BOOKLET_FIGSIZE)
        gs = fig.add_gridspec(nrows=1, ncols=1)
        ax = fig.add_subplot(gs[0, 0])

        if data:
            x_positions = list(range(1, len(valid_metrics) + 1))
            bp = ax.boxplot(
                data,
                positions=x_positions,
                showfliers=False,
                patch_artist=True,
                widths=0.55,
                boxprops=dict(linewidth=2.2, edgecolor="black"),
                whiskerprops=dict(linewidth=2.0, color="black"),
                capprops=dict(linewidth=2.0, color="black"),
                medianprops=dict(linewidth=2.6, color="black"),
            )
            for b in bp["boxes"]:
                b.set_facecolor("white")
                b.set_alpha(1.0)

            for x0, metric in zip(x_positions, valid_metrics):
                col = f"{metric}_vsGame_pct"
                yvals = pd.to_numeric(df_page[col], errors="coerce")
                names = df_page["Name"].fillna("").astype(str).tolist()

                for yv, nm in zip(yvals, names):
                    if pd.isna(yv):
                        continue
                    xj = x0 + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH)
                    ax.scatter([xj], [yv], alpha=POINT_ALPHA, s=POINT_SIZE)
                    label_text = nm
                    if SHOW_VALUE_IN_LABEL:
                        label_text = f"{nm} ({yv:.0f})"
                    ax.text(
                        xj + 0.02, yv, label_text,
                        fontsize=LABEL_FONTSIZE,
                        alpha=LABEL_ALPHA,
                        va="center"
                    )

                med = pd.to_numeric(df_page[col], errors="coerce").dropna().median()
                if pd.notna(med):
                    ax.text(
                        x0, med, f"{med:.1f}",
                        fontsize=MEDIAN_FONTSIZE,
                        ha="center", va="bottom"
                    )

            ymin, ymax = _compute_dynamic_ylim(page_values)
            ax.set_ylim(ymin, ymax)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(valid_metrics, rotation=30, ha="right")
            ax.tick_params(axis="x", labelsize=10, pad=6)
            ax.set_ylabel("% of Game Avg")
            ax.set_title(title_text)
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "表示できるデータがありません", ha="center", va="center", fontsize=14)
            ax.set_title(title_text)

        _draw_page_number(fig, page_no)
        fig.subplots_adjust(left=0.05, right=0.985, top=0.93, bottom=0.18)
        return fig

    def _draw_stats_table_page(df_page: pd.DataFrame, title_text: str, page_no: int) -> plt.Figure:
        valid_metrics, _, _ = _collect_valid_metrics(df_page)
        fig = plt.figure(figsize=BOOKLET_FIGSIZE)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(title_text, fontsize=15, pad=12)

        if valid_metrics:
            col_labels, cell_text = _build_position_stats_table(df_page, valid_metrics)
            tbl = ax.table(
                cellText=cell_text,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
                rowLoc="center",
                bbox=[0.01, 0.06, 0.98, 0.82],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7.0)
            tbl.scale(1.0, 1.62)
            try:
                ncols = len(col_labels)
                if ncols >= 2:
                    first_w = 0.19
                    other_w = (0.98 - first_w) / (ncols - 1)
                    for (r, c), cell in tbl.get_celld().items():
                        if c == 0:
                            cell.set_width(first_w)
                            cell.get_text().set_ha("left")
                            cell.PAD = 0.02
                        else:
                            cell.set_width(other_w)
            except Exception:
                pass
            ax.text(0.5, 0.92, "各ポジションの箱ひげ図統計値（N / Median / Q1-Q3 / Whisker Low-High）", ha="center", va="center", fontsize=11)
        else:
            ax.text(0.5, 0.5, "表示できる統計表がありません", ha="center", va="center", fontsize=14)

        _draw_page_number(fig, page_no)
        fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08)
        return fig

    def _draw_cover_page(total_pages: int) -> plt.Figure:
        fig = plt.figure(figsize=BOOKLET_FIGSIZE)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        fig.text(0.5, 0.72, "GPS セッション別箱ひげ図 冊子", ha="center", va="center", fontsize=24)
        fig.text(0.5, 0.58, "ALL → セッション別ポジション順", ha="center", va="center", fontsize=18)
        fig.text(0.5, 0.47, "※ ROUND ALL は掲載対象外", ha="center", va="center", fontsize=13)
        fig.text(0.5, 0.38, f"総ページ数: {total_pages}", ha="center", va="center", fontsize=14)
        _draw_page_number(fig, 1)
        return fig

    def _draw_toc_page(entries: List[Tuple[int, str]], toc_page_no: int, toc_page_idx: int, toc_page_count: int) -> plt.Figure:
        fig = plt.figure(figsize=BOOKLET_FIGSIZE)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        suffix = "" if toc_page_count == 1 else f" ({toc_page_idx}/{toc_page_count})"
        fig.text(0.06, 0.94, f"目次{suffix}", fontsize=22, ha="left", va="top")
        start_y = 0.88
        line_h = 0.042
        for i, (page_num, title) in enumerate(entries):
            y = start_y - i * line_h
            fig.text(0.07, y, f"{page_num:>3}  {title}", fontsize=11.5, ha="left", va="top")
        _draw_page_number(fig, toc_page_no)
        return fig

    pdf_buf = io.BytesIO()
    rng = np.random.default_rng(0)

    toc_chunk_size = 18
    toc_chunks = [page_specs[i:i + toc_chunk_size] for i in range(0, len(page_specs), toc_chunk_size)] or [[]]
    toc_page_count = len(toc_chunks)
    first_content_page = 2 + toc_page_count

    with PdfPages(pdf_buf) as pdf:
        cover_fig = _draw_cover_page(total_pages=1 + toc_page_count + len(page_specs))
        pdf.savefig(cover_fig)
        plt.close(cover_fig)

        for toc_idx, chunk in enumerate(toc_chunks, start=1):
            entries = []
            for local_idx, spec in enumerate(chunk):
                global_idx = (toc_idx - 1) * toc_chunk_size + local_idx
                page_no = first_content_page + global_idx
                entries.append((page_no, spec.get("toc_title", spec["title"])))
            toc_fig = _draw_toc_page(entries, toc_page_no=1 + toc_idx, toc_page_idx=toc_idx, toc_page_count=toc_page_count)
            pdf.savefig(toc_fig)
            plt.close(toc_fig)

        for page_no, spec in enumerate(page_specs, start=first_content_page):
            df_page = ratio_df[ratio_df["Session"] == spec["session"]].copy()
            if spec["position"] is not None:
                df_page = df_page[df_page["Position"] == spec["position"]].copy()

            if spec["kind"] == "all_stats":
                fig = _draw_stats_table_page(df_page, spec["title"], page_no)
            else:
                fig = _draw_boxplot_page(df_page, spec["title"], page_no, rng)
            pdf.savefig(fig)
            plt.close(fig)

    pdf_buf.seek(0)
    return pdf_buf

# =========================
# Session state
# =========================
def init_state():
    defaults = {
        "train_blob_name": None,
        "train_blob_bytes": None,
        "game_blob_name": None,
        "game_blob_bytes": None,
        "uploader_nonce": 0,
        "last_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def bump_uploader_nonce():
    st.session_state["uploader_nonce"] = int(st.session_state.get("uploader_nonce", 0)) + 1


def save_uploaded_files(train_upload, game_upload):
    if train_upload is None or game_upload is None:
        st.warning("練習データと試合平均の両方を選択してから保存してください。")
        return False

    validate_uploaded_file(train_upload, "練習データ")
    validate_uploaded_file(game_upload, "試合平均")

    st.session_state["train_blob_name"] = train_upload.name
    st.session_state["train_blob_bytes"] = bytes(train_upload.getvalue())
    st.session_state["game_blob_name"] = game_upload.name
    st.session_state["game_blob_bytes"] = bytes(game_upload.getvalue())
    st.session_state["last_error"] = None
    return True


def clear_uploaded_files():
    st.session_state["train_blob_name"] = None
    st.session_state["train_blob_bytes"] = None
    st.session_state["game_blob_name"] = None
    st.session_state["game_blob_bytes"] = None
    st.session_state["last_error"] = None
    bump_uploader_nonce()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="GPSデータ自動集計アプリ", layout="wide")
init_state()

st.title("GPSデータ自動集計アプリ（CSV + Excel + Numbers）")
st.caption(f"VERSION: {APP_VERSION} / FILE: app.py")

st.info(
    f"Position順：{' → '.join(POSITION_ORDER)}（それ以外/Unknownは最後）\n"
    "対応形式：csv / xlsx / xls / numbers\n"
    "High Agility / HighAgility_90 系の列名ゆれ対応を追加済みです。\n"
    "アップロードの 403 を避けるため、form を外し、uploader key を毎回更新できる安定版に変更しました。"
)

nonce = st.session_state.get("uploader_nonce", 0)
col1, col2 = st.columns(2)

with col1:
    train_upload = st.file_uploader(
        "① 練習データ（Position列ありでもOK）",
        type=STREAMLIT_FILE_TYPES,
        key=f"train_file_uploader_stable_v2_{nonce}",
        accept_multiple_files=False,
        help="csv / xlsx / xls / numbers"
    )

with col2:
    game_upload = st.file_uploader(
        "② 試合平均（ヘッダーがどの行でもOK）",
        type=STREAMLIT_FILE_TYPES,
        key=f"game_file_uploader_stable_v2_{nonce}",
        accept_multiple_files=False,
        help="csv / xlsx / xls / numbers"
    )

save_c1, save_c2 = st.columns([3, 1])
save_clicked = save_c1.button("ファイルを保存", use_container_width=True, type="secondary")
reset_clicked = save_c2.button("リセット", use_container_width=True)

if reset_clicked:
    clear_uploaded_files()
    st.rerun()

if save_clicked:
    if save_uploaded_files(train_upload, game_upload):
        st.success("ファイルを保存しました。下の『集計を実行』を押してください。")

saved_train_name = st.session_state.get("train_blob_name")
saved_game_name = st.session_state.get("game_blob_name")

if saved_train_name or saved_game_name:
    st.write("### 現在保存中のファイル")
    left, right = st.columns(2)
    with left:
        st.write(f"練習データ: {saved_train_name if saved_train_name else '未保存'}")
    with right:
        st.write(f"試合平均: {saved_game_name if saved_game_name else '未保存'}")

run_clicked = st.button("集計を実行", use_container_width=True, type="primary")

if run_clicked:
    train_blob_name = st.session_state.get("train_blob_name")
    train_blob_bytes = st.session_state.get("train_blob_bytes")
    game_blob_name = st.session_state.get("game_blob_name")
    game_blob_bytes = st.session_state.get("game_blob_bytes")

    if not train_blob_name or train_blob_bytes is None or not game_blob_name or game_blob_bytes is None:
        st.warning("先に『ファイルを保存』を押して、練習データと試合平均の両方を保存してください。")
        st.stop()

    try:
        with st.spinner("データを処理しています..."):
            train_df = read_any_table_from_bytes(train_blob_name, train_blob_bytes)
            game_df = read_game_table_any_header_from_bytes(game_blob_name, game_blob_bytes)

            session_df = build_session_metrics(train_df)
            ratio_df = build_ratio_df(session_df, game_df)

            st.success("集計が完了しました。")

            st.write("## セッション集計表")
            st.dataframe(session_df, use_container_width=True)

            st.write("## 試合平均比（%）")
            st.dataframe(ratio_df, use_container_width=True)

            csv_bytes = ratio_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "試合平均比テーブルをCSVダウンロード",
                data=csv_bytes,
                file_name="gps_vs_game_avg.csv",
                mime="text/csv",
                use_container_width=True
            )

            pdf_buf = make_session_boxplot_booklet_pdf(ratio_df)
            st.download_button(
                "セッション別箱ひげ図冊子PDFをダウンロード",
                data=pdf_buf.getvalue(),
                file_name="gps_session_boxplot_booklet.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            zip_buf = make_boxplots_png_zip_only(ratio_df)
            st.download_button(
                "既存のPNGグラフZIPをダウンロード",
                data=zip_buf.getvalue(),
                file_name="gps_boxplots_png.zip",
                mime="application/zip",
                use_container_width=True
            )

    except Exception as e:
        st.session_state["last_error"] = str(e)
        st.error("処理中にエラーが発生しました。")
        st.exception(e)

if st.session_state.get("last_error"):
    st.write("### 最後のエラー")
    st.code(st.session_state["last_error"])
