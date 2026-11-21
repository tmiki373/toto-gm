# -*- coding: utf-8 -*-
"""
toto 予想エンジン   v6.0（オッズ市場モデル・安全フォールバック）
  - データ取得 : くじ対象試合・販売締切 / 投票率 / くじ結果（スポーツくじ公式）、過去結果（Ｊリーグデータサイト）
  - 表示する回 : くじ種別ごとに『まだ試合が残っている回のうち一番古い回』（開催当日は当日の回）
  - モデル     : ① Dixon-Coles ポアソン（J1/J2/J3 の階層補正・ルヴァン杯の重み付け）
                 ② Elo + 順序ロジット（シーズン間の平均回帰・ルヴァン杯の重み付け）
                 → 過去データの答え合わせで設定値を自動調整し、2つを最適な比率で混合
                 ③ 学習補正（Stacker）: 振り返りで採用された「検討要素」で確率を補正
                 ④ 投票率ブレンド: 過去回の投票率と結果から混ぜる比率を学習
"""
import math
import re
import time
import unicodedata
from datetime import datetime, date
from itertools import product
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

JST = ZoneInfo("Asia/Tokyo")
BASE = "https://store.toto-dream.com/dcs/subos/screen"
TOTO_INFO_URL = f"{BASE}/pi01/spin000/PGSPIN00001DisptotoLotInfo.form"
VOTE_URL = f"{BASE}/pi09/spin003/PGSPIN00301InitVoteRate.form"
RESULT_URL = f"{BASE}/pi04/spin011/PGSPIN01101LnkHoldCntLotResultLsttoto.form"
JDATA_FORM_URL = "https://data.j-league.or.jp/SFMS01/"
JDATA_URL = "https://data.j-league.or.jp/SFMS01/search"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal toto analysis)"}
REQUEST_INTERVAL = 2.0
KINDS = ["toto", "mini toto-A組", "mini toto-B組", "totoGOAL3"]
COMMODITY = {"toto": "01", "totoGOAL3": "02", "mini toto-A組": "01", "mini toto-B組": "01"}
SEASONS = ["2025", "2026特別", "2026/27"]
CACHE_VER = "v2"
MAX_GOALS = 10
OUT3 = ["1", "0", "2"]
OUT4 = ["0", "1", "2", "3"]
KEEP_COMP = "J1|J2|J3|百年構想|ルヴァン|昇格|入れ替え|参入"
DROP_COMP = "U-21|ユース|エリート|育成|サテライト|オールスター|ドリーム|ワールドチャレンジ|国際"
CUP_RE = "ルヴァン"
TIER = {"J1": 0, "J2": 1, "J3": 2}
CARD_COLS = ["round", "kind", "no", "date", "time", "stadium", "home", "away", "deadline"]

# ---- モデル設定（自動調整の候補と初期値）
HP_VER = 1
HP_RETUNE = 30
POI_DEFAULT = {"half_life": 180, "l2": 0.05, "cup_w": 1.0, "tier": False}
ELO_DEFAULT = {"k": 24, "hfa": 55, "regress": 0.0, "cup_w": 1.0}
POI_GRID = [dict(half_life=h, l2=l, cup_w=c, tier=t)
            for h, l, c, t in product([90, 180, 365], [0.05, 0.15], [0.5, 1.0], [True, False])]
ELO_GRID = [dict(k=k, hfa=a, regress=r, cup_w=c)
            for k, a, r, c in product([16, 24, 32], [40, 55, 75], [0.0, 0.33], [0.5, 1.0])]

TEAM_ALIAS = {
    "北海道コンサドーレ札幌": "札幌", "ヴァンラーレ八戸": "八戸", "いわてグルージャ盛岡": "岩手",
    "ベガルタ仙台": "仙台", "ブラウブリッツ秋田": "秋田", "モンテディオ山形": "山形",
    "福島ユナイテッドFC": "福島", "いわきFC": "いわき", "鹿島アントラーズ": "鹿島",
    "水戸ホーリーホック": "水戸", "栃木SC": "栃木SC", "栃木シティ": "栃木C", "栃木シティFC": "栃木C",
    "ザスパ群馬": "群馬", "ザスパクサツ群馬": "群馬", "浦和レッズ": "浦和",
    "RB大宮アルディージャ": "大宮", "大宮アルディージャ": "大宮", "ジェフユナイテッド千葉": "千葉",
    "ジェフユナイテッド市原・千葉": "千葉", "柏レイソル": "柏", "FC東京": "FC東京",
    "東京ヴェルディ": "東京V", "FC町田ゼルビア": "町田", "川崎フロンターレ": "川崎F",
    "横浜F・マリノス": "横浜FM", "横浜FC": "横浜FC", "Y.S.C.C.横浜": "YS横浜",
    "湘南ベルマーレ": "湘南", "SC相模原": "相模原", "ヴァンフォーレ甲府": "甲府",
    "松本山雅FC": "松本", "AC長野パルセイロ": "長野", "アルビレックス新潟": "新潟",
    "カターレ富山": "富山", "ツエーゲン金沢": "金沢", "清水エスパルス": "清水",
    "ジュビロ磐田": "磐田", "藤枝MYFC": "藤枝", "アスルクラロ沼津": "沼津",
    "名古屋グランパス": "名古屋", "FC岐阜": "岐阜", "レイラック滋賀FC": "滋賀", "レイラック滋賀": "滋賀",
    "京都サンガF.C.": "京都", "ガンバ大阪": "G大阪", "セレッソ大阪": "C大阪", "FC大阪": "FC大阪",
    "ヴィッセル神戸": "神戸", "奈良クラブ": "奈良", "ガイナーレ鳥取": "鳥取",
    "ファジアーノ岡山": "岡山", "サンフレッチェ広島": "広島", "レノファ山口FC": "山口",
    "カマタマーレ讃岐": "讃岐", "徳島ヴォルティス": "徳島", "愛媛FC": "愛媛", "FC今治": "今治",
    "高知ユナイテッドSC": "高知", "アビスパ福岡": "福岡", "ギラヴァンツ北九州": "北九州",
    "サガン鳥栖": "鳥栖", "V・ファーレン長崎": "長崎", "ロアッソ熊本": "熊本",
    "大分トリニータ": "大分", "テゲバジャーロ宮崎": "宮崎", "鹿児島ユナイテッドFC": "鹿児島",
    "FC琉球": "琉球",
}


def norm(s) -> str:
    return unicodedata.normalize("NFKC", str(s)).strip().replace(" ", "").replace("\u3000", "")


ALIAS_N = {norm(k): norm(v) for k, v in TEAM_ALIAS.items()}


def to_short(name) -> str:
    n = norm(name)
    return ALIAS_N.get(n, n)


def now_jst():
    return datetime.now(JST)


# =============================================================================
# Web 取得
# =============================================================================
def http_get(url, params=None):
    import requests
    time.sleep(REQUEST_INTERVAL)
    r = requests.get(url, params=params, headers=HEADERS, timeout=40)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def html_text(html):
    if "<" in html:
        from bs4 import BeautifulSoup
        html = BeautifulSoup(html, "lxml").get_text(" ")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", html))


LOTTERY_RE = re.compile(r"第\s*(\d+)\s*回\s*(toto|mini\s*toto-A組|mini\s*toto-B組|totoGOAL\d?)\s*くじ情報")
MATCH_RE = re.compile(r"(?<!\d)(\d{1,2})\s+(\d{2}/\d{2})\s+(\d{1,2}:\d{2})\s+(\S+)\s+(\S+)\s+VS\s+(\S+)")
DEADLINE_RE = re.compile(r"販売終了日\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日.*?ネット決済\)\s*(\d{1,2}):(\d{2})")


def _match_date(mmdd, today):
    m, d = map(int, mmdd.split("/"))
    y = today.year + (1 if m < today.month - 6 else 0) - (1 if m > today.month + 6 else 0)
    return date(y, m, d)


def parse_toto_card(html_or_text, today=None) -> pd.DataFrame:
    today = today or now_jst().date()
    text = html_text(html_or_text)
    heads = list(LOTTERY_RE.finditer(text))
    rows = []
    for i, h in enumerate(heads):
        chunk = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        kind = re.sub(r"\s+", " ", h.group(2))
        if kind.startswith("totoGOAL"):
            kind = "totoGOAL3"
        dm = DEADLINE_RE.search(chunk[:600])
        deadline = datetime(*map(int, dm.groups()), tzinfo=JST).isoformat() if dm else None
        for m in MATCH_RE.finditer(chunk):
            rows.append({"round": int(h.group(1)), "kind": kind, "no": int(m.group(1)),
                         "date": _match_date(m.group(2), today), "time": m.group(3),
                         "stadium": m.group(4), "home": m.group(5), "away": m.group(6),
                         "deadline": deadline})
    return pd.DataFrame(rows, columns=CARD_COLS).drop_duplicates(["round", "kind", "no"])


def current_rounds(card: pd.DataFrame, today=None) -> pd.DataFrame:
    """
    くじ種別ごとに『まだ試合が残っている回（最終試合日が今日以降）のうち、一番古い回』を選ぶ。
    → 開催当日は当日の回を表示し、その回の試合がすべて終わった翌日に次の回へ切り替わる。
    （v4 までは『一番新しい回』を選んでいたため、開催当日に次の回へ切り替わっていた）
    """
    today = today or now_jst().date()
    if card.empty:
        return card
    last = card.groupby(["round", "kind"])["date"].transform("max")
    alive = card[last >= today]
    c = alive if not alive.empty else card
    pick = c.groupby("kind")["round"].transform("min" if not alive.empty else "max") == c["round"]
    return c[pick].reset_index(drop=True)


latest_rounds = current_rounds      # 旧名との互換


def _in_progress_from_results(kind, rnd, today, log=print):
    """販売が終わって『くじ情報』に載っていない開催中の回を『くじ結果』ページから探す"""
    found = None
    for r in range(rnd - 1, rnd - 4, -1):
        try:
            res = fetch_round_result(r)
        except Exception as e:  # noqa
            log(f"第{r}回のくじ結果を取得できず: {e}")
            continue
        g = res[(res["kind"] == kind) & (res["round"] == r)]
        if g.empty:
            continue
        if max(g["date"]) >= today:
            found = g
            continue            # さらに古い回も開催中か確認
        break                   # 終わった回に到達したら打ち切り
    if found is None:
        return None
    out = found.copy()
    out["time"] = ""
    out["deadline"] = None
    return out[CARD_COLS]


def fetch_card(round_no=None, log=print):
    html = http_get(TOTO_INFO_URL, {"holdCntId": round_no} if round_no else None)
    card = parse_toto_card(html)
    if card.empty:
        raise RuntimeError("くじ情報ページから試合を読み取れませんでした（ページ構成変更の可能性）")
    if round_no:
        return card[card["round"] == round_no].reset_index(drop=True)
    today = now_jst().date()
    card = current_rounds(card, today)
    parts = []
    for kind, g in card.groupby("kind"):
        rnd = int(g["round"].iloc[0])
        older = _in_progress_from_results(kind, rnd, today, log)
        if older is not None and int(older["round"].iloc[0]) < rnd:
            log(f"{kind}: 開催中の第{int(older['round'].iloc[0])}回を表示（販売中の第{rnd}回は試合終了後に表示）")
            parts.append(older)
        else:
            parts.append(g)
    return pd.concat(parts, ignore_index=True)


PCT_RE = re.compile(r"[\d,]+\s*[（(]\s*([\d.]+)\s*%\s*[）)]")


def parse_vote_rates(html_or_text, n_slots, n_out):
    text = html_text(html_or_text).split("合計")[0]
    pcts = [float(x) / 100 for x in PCT_RE.findall(text)]
    if len(pcts) < n_slots * n_out:
        return None
    pcts = pcts[: n_slots * n_out]
    return [pcts[i * n_out:(i + 1) * n_out] for i in range(n_slots)]


def fetch_vote_rates(round_no, kind, n_slots):
    n_out = 4 if kind == "totoGOAL3" else 3
    html = http_get(VOTE_URL, {"holdCntId": round_no, "commodityId": COMMODITY[kind]})
    return parse_vote_rates(html, n_slots, n_out)


# ---------------------------------------------------------------- 公式「くじ結果」ページ
RES_HEAD_RE = re.compile(r"第\s*(\d+)\s*回\s*(toto|mini\s*toto-A組|mini\s*toto-B組|totoGOAL\d?)\s*くじ結果")
RES3_RE = re.compile(r"(\d{2}/\d{2})\s+(\S+)\s+(\d{1,2})\s+(\S+)\s+(?:(\d+)\s*-\s*(\d+)\s+|-\s+)?(\S+)(?:\s+([012])(?=\s|$))?")
RESG_RE = re.compile(r"(\d{2}/\d{2})\s+(\S+)\s+(\d{1,2})\s+(\S+)\s+(?:(\d+)\s+([0-3])\s+)?(\d{1,2})\s+(\S+)(?:\s+(\d+)\s+([0-3])(?=\s|$))?")


def parse_round_result(html_or_text, today=None):
    """公式『くじ結果』ページ → くじ種別ごとの対象試合とスコア（未確定は None）"""
    today = today or now_jst().date()
    text = html_text(html_or_text)
    heads = list(RES_HEAD_RE.finditer(text))
    rows = []
    for i, h in enumerate(heads):
        chunk = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)].split("ご注意")[0]
        kind = re.sub(r"\s+", " ", h.group(2))
        rnd = int(h.group(1))
        if kind.startswith("totoGOAL"):
            for k, m in enumerate(RESG_RE.finditer(chunk), 1):
                rows.append({"round": rnd, "kind": "totoGOAL3", "no": k, "date": _match_date(m.group(1), today),
                             "time": "", "stadium": m.group(2), "home": m.group(4), "away": m.group(8),
                             "hg": int(m.group(5)) if m.group(5) else None,
                             "ag": int(m.group(9)) if m.group(9) else None})
        else:
            for m in RES3_RE.finditer(chunk):
                rows.append({"round": rnd, "kind": kind, "no": int(m.group(3)), "date": _match_date(m.group(1), today),
                             "time": "", "stadium": m.group(2), "home": m.group(4), "away": m.group(7),
                             "hg": int(m.group(5)) if m.group(5) else None,
                             "ag": int(m.group(6)) if m.group(6) else None})
    cols = ["round", "kind", "no", "date", "time", "stadium", "home", "away", "hg", "ag"]
    return pd.DataFrame(rows, columns=cols).drop_duplicates(["round", "kind", "no"])


def fetch_round_result(rnd):
    return parse_round_result(http_get(RESULT_URL, {"holdCntId": rnd}))


# ---------------------------------------------------------------- Ｊリーグデータサイト
def parse_form_options(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    out = {}
    for sel in soup.find_all("select"):
        name = sel.get("name")
        if name:
            out.setdefault(name, []).extend(
                (o.get("value", o.get_text(strip=True)), norm(o.get_text(strip=True))) for o in sel.find_all("option"))
    for inp in soup.find_all("input", attrs={"type": re.compile("checkbox|radio", re.I)}):
        name = inp.get("name")
        if not name:
            continue
        label = ""
        if inp.get("id"):
            lab = soup.find("label", attrs={"for": inp["id"]})
            label = lab.get_text(strip=True) if lab else ""
        if not label and inp.parent is not None and inp.parent.name == "label":
            label = inp.parent.get_text(strip=True)
        if not label and inp.next_sibling:
            label = str(inp.next_sibling).strip()
        out.setdefault(name, []).append((inp.get("value", ""), norm(label)))
    return out


def resolve_search_params(form):
    season_vals = {}
    years = form.get("competition_years", [])
    for s in SEASONS:
        hit = [v for v, lab in years if lab == norm(s)] or [v for v, lab in years if v == s]
        season_vals[s] = hit[0] if hit else s
    frames = [(v, lab) for v, lab in form.get("competition_frame_ids", [])
              if v and re.search(KEEP_COMP, lab) and not re.search(DROP_COMP, lab)]
    return season_vals, frames


def parse_jdata_table(html) -> pd.DataFrame:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="search-table") or soup.find("table")
    if table is None:
        return pd.DataFrame()
    head = [th.get_text(strip=True) for th in table.find_all("th")]
    data = [[td.get_text(" ", strip=True) for td in tr.find_all("td")] for tr in table.find_all("tr")]
    data = [d for d in data if len(d) >= 8]
    if not data or len(head) < 8:
        return pd.DataFrame()
    w = min(len(head), len(data[0]))
    return pd.DataFrame([d[:w] for d in data], columns=head[:w])


def clean_results(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["comp", "date", "home", "away", "hg", "ag"])
    raw = raw.copy()
    raw.columns = [norm(c) for c in raw.columns]
    out = pd.DataFrame({
        "comp": raw["大会"].map(norm),
        "date": pd.to_datetime(raw["試合日"].str.extract(r"(\d{2}/\d{2}/\d{2})")[0],
                               format="%y/%m/%d", errors="coerce"),
        "home": raw["ホーム"].map(to_short),
        "away": raw["アウェイ"].map(to_short),
        "score": raw["スコア"].map(norm),
    })
    sc = out["score"].str.extract(r"^(\d+)-(\d+)")
    out["hg"], out["ag"] = pd.to_numeric(sc[0]), pd.to_numeric(sc[1])
    out = out.dropna(subset=["date", "hg", "ag"])
    keep = out["comp"].str.contains(KEEP_COMP, regex=True)
    drop = out["comp"].str.contains(DROP_COMP, regex=True)
    out = out[keep & ~drop].astype({"hg": int, "ag": int})
    return out.drop(columns="score").reset_index(drop=True)


def fetch_results(cache_dir, refresh=False, log=print) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        season_vals, frames = resolve_search_params(parse_form_options(http_get(JDATA_FORM_URL)))
        log(f"検索条件: シーズン {season_vals} / 大会 {len(frames)}種")
    except Exception as e:  # noqa
        log(f"検索フォームの解析に失敗（既定値で続行）: {e}")
        season_vals, frames = {s: s for s in SEASONS}, []

    out = []
    for s in SEASONS:
        cache = cache_dir / f"results_{CACHE_VER}_{re.sub(r'[^0-9A-Za-z]', '_', s)}_{len(s)}.csv"
        if cache.exists() and not refresh and s != SEASONS[-1]:
            out.append(pd.read_csv(cache, parse_dates=["date"]))
            continue
        parts, failed = [], False
        for fid, lab in (frames or [(None, "全大会")]):
            params = {"competition_years": season_vals[s], "tv_relay_station_name": ""}
            if fid:
                params["competition_frame_ids"] = fid
            try:
                raw = parse_jdata_table(http_get(JDATA_URL, params))
                if len(raw) >= 1500:
                    log(f"[注意] {s} {lab}: 1,500件の上限に達しました（一部欠落の可能性）")
                parts.append(clean_results(raw))
            except Exception as e:  # noqa
                failed = True
                log(f"{s} {lab} の取得に失敗: {e}")
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if (df.empty or failed) and cache.exists():
            old = pd.read_csv(cache, parse_dates=["date"])
            df = pd.concat([old, df], ignore_index=True) if not df.empty else old
            log(f"シーズン {s}: 保存済みデータで補完")
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"].astype(str).str.slice(0, 10), errors="coerce")
            df = df.drop_duplicates(["date", "home", "away"])
            df.to_csv(cache, index=False, encoding="utf-8-sig")
            log(f"シーズン {s}: {len(df)} 試合（最新 {df['date'].max():%Y/%m/%d}）")
        out.append(df)
    out = [f for f in out if not f.empty]
    if not out:
        raise RuntimeError("過去の試合結果を取得できませんでした")
    return pd.concat(out, ignore_index=True)


def clean_res(res: pd.DataFrame, log=print) -> pd.DataFrame:
    res = res.copy()
    res["date"] = pd.to_datetime(res["date"].astype(str).str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    res["hg"] = pd.to_numeric(res["hg"], errors="coerce")
    res["ag"] = pd.to_numeric(res["ag"], errors="coerce")
    bad = int(res[["date", "home", "away", "hg", "ag"]].isna().any(axis=1).sum())
    if bad:
        log(f"不正な行を除外: {bad}件")
    res = res.dropna(subset=["date", "home", "away", "hg", "ag"]).drop_duplicates(["date", "home", "away"])
    res["comp"] = res["comp"].fillna("").astype(str)
    res["home"], res["away"] = res["home"].astype(str), res["away"].astype(str)
    res[["hg", "ag"]] = res[["hg", "ag"]].astype(int)
    return res.sort_values("date").reset_index(drop=True)


def team_leagues(res):
    lg = res[~res["comp"].str.contains("ルヴァン|昇格|入れ替え|参入")].copy()
    lg["lv"] = lg["comp"].str.extract(r"J([123])")[0]
    lg.loc[lg["lv"].isna() & lg["comp"].str.contains("百年構想"), "lv"] = "1"
    lg = lg.dropna(subset=["lv"])
    if lg.empty:
        return {}
    long = pd.concat([lg[["date", "home", "lv"]].rename(columns={"home": "team"}),
                      lg[["date", "away", "lv"]].rename(columns={"away": "team"})])
    return long.sort_values("date").groupby("team")["lv"].last().map(lambda x: f"J{x}").to_dict()


def _cup_mask(df):
    return df["comp"].str.contains(CUP_RE, regex=True).to_numpy() if "comp" in df else np.zeros(len(df), bool)


# =============================================================================
# モデル① Dixon-Coles ポアソン
# =============================================================================
def _tau(x, y, lh, la, rho):
    t = np.ones_like(lh, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


class PoissonModel:
    def __init__(self, half_life=180, l2=0.05, cup_w=1.0, tier=True):
        self.half_life, self.l2, self.cup_w, self.use_tier = half_life, l2, cup_w, tier

    def fit(self, df, ref_date, tiers=None):
        teams = sorted(set(df["home"]) | set(df["away"]))
        self.teams = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        tiers = (tiers or {}) if self.use_tier else {}
        tier = np.array([TIER.get(tiers.get(t), 1) for t in teams])
        h = df["home"].map(self.teams).to_numpy()
        a = df["away"].map(self.teams).to_numpy()
        hg, ag = df["hg"].to_numpy(float), df["ag"].to_numpy(float)
        w = 0.5 ** ((pd.Timestamp(ref_date) - df["date"]).dt.days.clip(lower=0).to_numpy() / self.half_life)
        w = w * np.where(_cup_mask(df), self.cup_w, 1.0)
        l2, eps = self.l2, 1e-3

        def nll(p):
            att, de = p[:n], p[n:2 * n]
            ta, td = p[2 * n:2 * n + 3], p[2 * n + 3:2 * n + 6]
            home, mu = p[2 * n + 6], p[2 * n + 7]
            A, D = att + ta[tier], de + td[tier]
            lh = np.exp(mu + home + A[h] + D[a]); la = np.exp(mu + A[a] + D[h])
            f = np.sum(w * (lh - hg * np.log(lh) + la - ag * np.log(la)))
            f += l2 * (att @ att + de @ de) + eps * (ta @ ta + td @ td)
            rh, ra = w * (lh - hg), w * (la - ag)
            gA = np.bincount(h, rh, n) + np.bincount(a, ra, n)
            gD = np.bincount(a, rh, n) + np.bincount(h, ra, n)
            g = np.concatenate([gA + 2 * l2 * att, gD + 2 * l2 * de,
                                np.bincount(tier, gA, 3) + 2 * eps * ta, np.bincount(tier, gD, 3) + 2 * eps * td,
                                [rh.sum()], [rh.sum() + ra.sum()]])
            return f, g

        x0 = np.zeros(2 * n + 8); x0[-2:] = 0.2
        p = minimize(nll, x0, jac=True, method="L-BFGS-B").x
        ta, td = p[2 * n:2 * n + 3], p[2 * n + 3:2 * n + 6]
        self.att, self.de = p[:n] + ta[tier], p[n:2 * n] + td[tier]
        self.home_adv, self.mu = p[2 * n + 6], p[2 * n + 7]
        lh = np.exp(self.mu + self.home_adv + self.att[h] + self.de[a])
        la = np.exp(self.mu + self.att[a] + self.de[h])
        best = (-np.inf, 0.0)
        for rho in np.arange(-0.25, 0.101, 0.01):
            t = _tau(hg, ag, lh, la, rho)
            if np.all(t > 0):
                best = max(best, (np.sum(w * np.log(t)), rho))
        self.rho = best[1]
        return self

    def known(self, t):
        return t in self.teams

    def matrix(self, home, away):
        i, j = self.teams[home], self.teams[away]
        lh = math.exp(self.mu + self.home_adv + self.att[i] + self.de[j])
        la = math.exp(self.mu + self.att[j] + self.de[i])
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        x, y = np.meshgrid(g, g, indexing="ij")
        m *= _tau(x, y, np.full_like(m, lh), np.full_like(m, la), self.rho)
        return m / m.sum(), lh, la

    def probs(self, home, away):
        m, lh, la = self.matrix(home, away)
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]), lh, la

    def top_scores(self, home, away, k=3):
        m = self.matrix(home, away)[0]
        idx = np.dstack(np.unravel_index(np.argsort(-m.ravel())[:k], m.shape))[0]
        return [(int(i), int(j), float(m[i, j])) for i, j in idx]

    def ratings(self):
        inv = {i: t for t, i in self.teams.items()}
        return pd.DataFrame({"チーム": [inv[i] for i in range(len(inv))],
                             "攻撃力": np.exp(self.att - self.att.mean()),
                             "守備力(失点倍率)": np.exp(self.de - self.de.mean())})


# =============================================================================
# モデル② Elo + 順序ロジット
# =============================================================================
def _sig(x):
    return 1 / (1 + np.exp(-x))


class EloModel:
    def __init__(self, k=24, hfa=55, regress=0.0, cup_w=1.0):
        self.k, self.hfa, self.regress, self.cup_w = k, hfa, regress, cup_w

    def run(self, df):
        self.r, self.init = {}, {}
        diffs = np.empty(len(df))
        last = None
        cup = _cup_mask(df)
        for idx, (comp, d, h, a, hg, ag) in enumerate(
                df[["comp", "date", "home", "away", "hg", "ag"]].itertuples(index=False)):
            if last is not None and self.regress and (d - last).days > 40:
                for t in self.r:
                    self.r[t] -= self.regress * (self.r[t] - self.init[t])
            last = d
            init = 1600 if "J1" in comp or "百年構想" in comp else 1400 if "J3" in comp else 1500
            for t in (h, a):
                if t not in self.r:
                    self.r[t] = self.init[t] = init
            rh, ra = self.r[h], self.r[a]
            dd = rh + self.hfa - ra
            diffs[idx] = dd
            exp = 1 / (1 + 10 ** (-dd / 400))
            res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            k = self.k * (self.cup_w if cup[idx] else 1.0)
            delta = k * (math.log(abs(hg - ag) + 1) + 1) * (res - exp)
            self.r[h], self.r[a] = rh + delta, ra - delta
        return diffs

    def fit_logit(self, diffs, outcome, w=None):
        w = np.ones_like(diffs) if w is None else w
        x = diffs / 100

        def nll(p):
            b, c1, gap = p
            c2 = c1 + math.exp(gap)
            pa = _sig(c1 - b * x); pna = _sig(c2 - b * x)
            pr = np.choose(outcome, [1 - pna, pna - pa, pa])
            return -np.sum(w * np.log(np.clip(pr, 1e-9, 1)))

        self.b, self.c1, gap = minimize(nll, [0.5, -0.9, -0.5], method="Nelder-Mead").x
        self.c2 = self.c1 + math.exp(gap)
        return self

    def probs_from_diff(self, d):
        x = np.asarray(d) / 100
        pa = _sig(self.c1 - self.b * x); pna = _sig(self.c2 - self.b * x)
        return np.stack([1 - pna, pna - pa, pa], axis=-1)

    def known(self, t):
        return t in self.r

    def probs(self, home, away):
        return self.probs_from_diff(self.r[home] + self.hfa - self.r[away])


def outcome_code(df):
    return np.where(df["hg"] > df["ag"], 0, np.where(df["hg"] == df["ag"], 1, 2))


# =============================================================================
# v6: オッズ市場モデル（オッズ未取得時は完全に従来モデルへフォールバック）
# =============================================================================
def odds_to_prob(odds):
    """10進オッズ [home, draw, away] から控除率を除いた確率を返す。無効なら None。"""
    if odds is None:
        return None
    try:
        o = np.asarray(odds, float)
    except Exception:
        return None
    if o.shape != (3,) or not np.isfinite(o).all() or np.any(o <= 1.0):
        return None
    q = 1.0 / o
    return q / q.sum()


def log_pool(p_model, p_market, weight):
    """対数加重平均。weight=0 はモデルのみ、1 は市場のみ。"""
    p = np.clip(np.asarray(p_model, float), 1e-9, 1)
    if p_market is None or weight <= 0:
        return p / p.sum()
    m = np.clip(np.asarray(p_market, float), 1e-9, 1)
    w = float(np.clip(weight, 0, 1))
    q = np.exp((1 - w) * np.log(p) + w * np.log(m))
    return q / q.sum()


def js_divergence(p, q):
    """Jensen-Shannon divergence（0=一致、約0.693=最大）。"""
    if q is None:
        return None
    p, q = np.clip(np.asarray(p,float),1e-9,1), np.clip(np.asarray(q,float),1e-9,1)
    p, q = p/p.sum(), q/q.sum(); m=(p+q)/2
    return float(.5*np.sum(p*np.log(p/m)) + .5*np.sum(q*np.log(q/m)))


def load_odds(card, path):
    """
    任意CSVを読む。列: round,kind,no,home_odds,draw_odds,away_odds[,captured_at]
    kind は省略可（toto扱い）。欠損・古い・不正な行は無視する。
    戻り値: {(round,kind,no): {odds,p_market,captured_at}}
    """
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    need={'round','no','home_odds','draw_odds','away_odds'}
    if not need.issubset(df.columns):
        return {}
    if 'kind' not in df: df['kind']='toto'
    ans={}
    for _,r in df.iterrows():
        pm=odds_to_prob([r.home_odds,r.draw_odds,r.away_odds])
        if pm is None: continue
        key=(int(r['round']),str(r['kind']),int(r['no']))
        ans[key]={'odds':[float(r.home_odds),float(r.draw_odds),float(r.away_odds)],
                  'p_market':pm,'captured_at':str(r.get('captured_at',''))}
    return ans


def metrics(P, y):
    P = np.clip(np.asarray(P, float), 1e-9, 1)
    onehot = np.eye(3)[y]
    cp, co = np.cumsum(P, axis=1)[:, :2], np.cumsum(onehot, axis=1)[:, :2]
    conf=P.max(1); correct=(P.argmax(1)==y).astype(float)
    ece=0.0
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        z=(conf>=lo)&(conf<(hi if hi<1 else 1.01))
        if z.any(): ece += z.mean()*abs(conf[z].mean()-correct[z].mean())
    return {"ll": float(-np.mean(np.log(P[np.arange(len(y)), y]))),
            "brier": float(np.mean(np.sum((P - onehot) ** 2, axis=1))),
            "acc": float(np.mean(correct)),
            "rps": float(np.mean(np.sum((cp-co)**2,axis=1)/2)),
            "ece": float(ece),
            "draw_pred": float(P[:,1].mean()), "draw_act": float((y==1).mean()),
            "sharpness": float(np.mean(np.std(P,axis=1)))}


# =============================================================================
# 検討要素と学習補正
# =============================================================================
def match_features(res):
    r = res.reset_index(drop=True)
    ph = np.where(r["hg"] > r["ag"], 3, np.where(r["hg"] == r["ag"], 1, 0))
    pa = np.where(r["ag"] > r["hg"], 3, np.where(r["hg"] == r["ag"], 1, 0))
    long = pd.concat([
        pd.DataFrame({"i": r.index, "side": "h", "team": r["home"], "date": r["date"], "pts": ph}),
        pd.DataFrame({"i": r.index, "side": "a", "team": r["away"], "date": r["date"], "pts": pa}),
    ]).sort_values(["team", "date", "i"]).reset_index(drop=True)
    g = long.groupby("team", sort=False)
    long["form"] = g["pts"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum()).fillna(7.0)
    long["rest"] = (long["date"] - g["date"].shift(1)).dt.days.fillna(14).clip(0, 21)
    h = long[long["side"] == "h"].set_index("i").sort_index()
    a = long[long["side"] == "a"].set_index("i").sort_index()
    return pd.DataFrame({"rest_h": h["rest"].values, "rest_a": a["rest"].values,
                         "form_h": h["form"].values, "form_a": a["form"].values})


FEAT_INFO = {
    "temperature": ("確率の強弱補正", "予想が全体的に強気すぎ・弱気すぎる傾向を補正"),
    "draw_bias": ("引き分け補正", "引き分けの起きやすさを全体的に補正"),
    "closeness": ("接戦度→引き分け", "実力が拮抗した試合ほど引き分けを多めに見る"),
    "goal_env": ("ロースコア傾向", "両チームの期待得点が低い試合ほど引き分けを多めに見る"),
    "rest_diff": ("休養日差", "前の試合からの間隔の差（連戦による疲労）"),
    "form_diff": ("直近の勢い", "直近5試合の勝点の差"),
    "model_gap": ("モデル間の不一致", "2つのモデルの見立てが割れる試合を引き分け寄りに"),
    "vote": ("投票率（世間の予想）", "一般の投票率を予想に混ぜる（過去回の投票率と結果から比率を学習）"),
}
DRAW_FEATS = ["closeness", "goal_env", "model_gap"]
SIDE_FEATS = ["rest_diff", "form_diff"]
STACK_FEATS = ["temperature", "draw_bias"] + DRAW_FEATS + SIDE_FEATS


def feat_values(P, F):
    lam = np.nan_to_num(np.asarray(F["lam_h"], float) + np.asarray(F["lam_a"], float), nan=2.6)
    return {
        "closeness": 1 - np.abs(P[:, 0] - P[:, 2]),
        "goal_env": 2.6 - lam,
        "model_gap": np.nan_to_num(np.asarray(F["gap"], float)),
        "rest_diff": np.clip(np.asarray(F["rest_h"], float) - np.asarray(F["rest_a"], float), -7, 7) / 7,
        "form_diff": (np.asarray(F["form_h"], float) - np.asarray(F["form_a"], float)) / 15,
    }


class Stacker:
    def __init__(self, active=(), theta=None):
        self.active = [a for a in STACK_FEATS if a in active]
        self.theta = None if theta is None else np.asarray(theta, float)

    def names(self):
        n = []
        if "temperature" in self.active:
            n.append("temperature")
        if "draw_bias" in self.active:
            n.append("draw_bias")
        n += [f for f in DRAW_FEATS if f in self.active]
        n += [f for f in SIDE_FEATS if f in self.active]
        return n

    def _theta0(self):
        return np.array([1.0 if n == "temperature" else 0.0 for n in self.names()])

    def _logits(self, th, P, X):
        p = dict(zip(self.names(), th))
        L = p.get("temperature", 1.0) * np.log(np.clip(P, 1e-9, 1))
        L = L.copy()
        L[:, 1] += p.get("draw_bias", 0.0)
        for f in DRAW_FEATS:
            if f in p:
                L[:, 1] += p[f] * X[f]
        for f in SIDE_FEATS:
            if f in p:
                L[:, 0] += p[f] * X[f]
                L[:, 2] -= p[f] * X[f]
        L -= L.max(1, keepdims=True)
        Ex = np.exp(L)
        return Ex / Ex.sum(1, keepdims=True)

    def fit(self, P, F, y, reg=0.01):
        if not self.active:
            self.theta = np.array([])
            return self
        X = feat_values(P, F)
        th0 = self._theta0()

        def loss(th):
            Q = self._logits(th, P, X)
            return -np.mean(np.log(np.clip(Q[np.arange(len(y)), y], 1e-9, 1))) + reg * np.sum((th - th0) ** 2)

        self.theta = minimize(loss, th0, method="L-BFGS-B").x
        return self

    def apply(self, P, F):
        P = np.atleast_2d(np.asarray(P, float))
        if not self.active or self.theta is None or len(self.theta) == 0:
            return P
        return self._logits(self.theta, P, feat_values(P, F))

    def params(self):
        return dict(zip(self.names(), [float(x) for x in (self.theta if self.theta is not None else [])]))

    def to_dict(self):
        return {"active": self.active, "theta": [float(x) for x in (self.theta if self.theta is not None else [])]}

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls(d.get("active", []), d.get("theta"))


def subF(F, idx):
    return {k: np.asarray(v)[idx] for k, v in F.items()}


def cv_score(active, P, F, y, k=5):
    idx = np.arange(len(y))
    chunks = np.array_split(idx, k)
    Q, te_all = [], []
    for j in range(1, k):
        tr, te = np.concatenate(chunks[:j]), chunks[j]
        st = Stacker(active).fit(P[tr], subF(F, tr), y[tr])
        Q.append(st.apply(P[te], subF(F, te)))
        te_all.append(te)
    Q, te_all = np.vstack(Q), np.concatenate(te_all)
    return metrics(Q, y[te_all])["ll"], Q, te_all


# =============================================================================
# 学習：設定値の自動調整
# =============================================================================
def _wf_split(res, frac=0.4):
    s = int(len(res) * (1 - frac))
    t0 = res["date"].iloc[s]
    return int((res["date"] < t0).sum()), t0


def elo_oos(res, s_idx, y, cfg):
    elo = EloModel(**cfg)
    diffs = elo.run(res)
    t0 = res["date"].iloc[s_idx]
    w = 0.5 ** ((t0 - res["date"].iloc[:s_idx]).dt.days.to_numpy() / 365)
    elo.fit_logit(diffs[:s_idx], y[:s_idx], w)
    return elo.probs_from_diff(diffs[s_idx:])


def poisson_oos(res, s_idx, t0, configs, base, step=42):
    chunks, t, end = [], t0, res["date"].max()
    while t <= end:
        te = np.where((res["date"] >= t) & (res["date"] < t + pd.Timedelta(days=step)))[0]
        if len(te):
            tr = res[res["date"] < t]
            chunks.append((t, tr, team_leagues(tr), te))
        t += pd.Timedelta(days=step)
    m = len(res) - s_idx
    out = []
    for cfg in configs:
        P = np.tile(base, (m, 1)); LAM = np.full((m, 2), np.nan)
        for t, tr, tiers, te in chunks:
            pm = PoissonModel(**cfg).fit(tr, t, tiers)
            for i in te:
                h, a = res.at[i, "home"], res.at[i, "away"]
                if pm.known(h) and pm.known(a):
                    P[i - s_idx], lh, la = pm.probs(h, a)
                    LAM[i - s_idx] = (lh, la)
        out.append((cfg, P, LAM))
    return out


def train(res: pd.DataFrame, today=None, hp=None, log=print):
    today = pd.Timestamp(today or now_jst().date())
    res = res.sort_values("date").reset_index(drop=True)
    y = outcome_code(res)
    s_idx, t0 = _wf_split(res)
    yt = y[s_idx:]
    base = np.bincount(y[:s_idx], minlength=3) / max(s_idx, 1)
    retune = (not hp or hp.get("ver") != HP_VER or len(res) - hp.get("n", 0) >= HP_RETUNE)

    report = {}
    if retune:
        log(f"モデル設定を調整中（Elo {len(ELO_GRID)}通り・ポアソン {len(POI_GRID)}通り）")
        sc = [(metrics(elo_oos(res, s_idx, y, c), yt)["ll"], i) for i, c in enumerate(ELO_GRID)]
        ecfg = ELO_GRID[min(sc)[1]]
        report["elo"] = {"best": min(sc)[0], "default": metrics(elo_oos(res, s_idx, y, ELO_DEFAULT), yt)["ll"]}
        cands = poisson_oos(res, s_idx, t0, POI_GRID, base)
        lls = [metrics(P, yt)["ll"] for _, P, _ in cands]
        pcfg, P_poi, LAM = cands[int(np.argmin(lls))]
        dflt = [l for (c, _, _), l in zip(cands, lls) if c == POI_DEFAULT]
        report["poi"] = {"best": min(lls), "default": dflt[0] if dflt else None}
    else:
        ecfg, pcfg = hp["elo"], hp["poi"]
        _, P_poi, LAM = poisson_oos(res, s_idx, t0, [pcfg], base)[0]
    P_elo = elo_oos(res, s_idx, y, ecfg)

    best_w, best_ll = 0.5, np.inf
    for wgt in np.linspace(0, 1, 11):
        ll = metrics(wgt * P_poi + (1 - wgt) * P_elo, yt)["ll"]
        if ll < best_ll:
            best_w, best_ll = wgt, ll

    mf = match_features(res).iloc[s_idx:].reset_index(drop=True)
    F = {"rest_h": mf["rest_h"].values, "rest_a": mf["rest_a"].values,
         "form_h": mf["form_h"].values, "form_a": mf["form_a"].values,
         "lam_h": LAM[:, 0], "lam_a": LAM[:, 1], "gap": 0.5 * np.abs(P_poi - P_elo).sum(1)}
    oos = {"P_poi": P_poi, "P_elo": P_elo, "P_base": best_w * P_poi + (1 - best_w) * P_elo,
           "y": yt, "F": F, "base": base, "dates": res["date"].iloc[s_idx:].reset_index(drop=True)}

    pred = Predictor.fit(res, today, w=best_w, hp={"elo": ecfg, "poi": pcfg})
    new_hp = {"ver": HP_VER, "n": len(res) if retune else hp["n"], "elo": ecfg, "poi": pcfg}
    return {"poisson": pred.poi, "elo": pred.elo, "w": best_w, "oos": oos, "hp": new_hp, "retuned": retune,
            "hp_report": report, "test_period": (oos["dates"].min(), oos["dates"].max(), len(yt)),
            "base": np.bincount(y, minlength=3) / len(y)}


# =============================================================================
# 予想
# =============================================================================
def team_form(res, team, before, n=5):
    g = res[((res["home"] == team) | (res["away"] == team)) & (res["date"] < before)].tail(n)
    marks, gf, ga, pts = [], 0, 0, 0
    for _, r in g.iterrows():
        f, a = (r["hg"], r["ag"]) if r["home"] == team else (r["ag"], r["hg"])
        gf += f; ga += a
        marks.append("W" if f > a else "D" if f == a else "L")
        pts += 3 if f > a else 1 if f == a else 0
    last = g["date"].max() if len(g) else None
    return {"marks": "".join(marks), "gf": int(gf), "ga": int(ga), "pts": int(pts) if len(g) else 7, "last": last}


def head_to_head(res, h, a, n=5, before=None):
    g = res[((res["home"] == h) & (res["away"] == a)) | ((res["home"] == a) & (res["away"] == h))]
    if before is not None:
        g = g[g["date"] < before]
    g = g.tail(n)
    w = d = l = 0
    for _, r in g.iterrows():
        f, x = (r["hg"], r["ag"]) if r["home"] == h else (r["ag"], r["hg"])
        w += f > x; d += f == x; l += f < x
    return f"{w}勝{d}分{l}敗" if len(g) else "対戦なし"


class Predictor:
    def __init__(self, res, poi, elo, w, base, stacker=None, vote_b=0.0, odds_w=0.0):
        self.res, self.poi, self.elo, self.w, self.base = res, poi, elo, w, base
        self.stacker = stacker or Stacker()
        self.vote_b = vote_b
        self.odds_w = float(np.clip(odds_w, 0, 1))

    @classmethod
    def fit(cls, res, today, w=0.5, stacker=None, vote_b=0.0, hp=None, odds_w=0.0):
        hp = hp or {}
        res = res.sort_values("date").reset_index(drop=True)
        y = outcome_code(res)
        elo = EloModel(**(hp.get("elo") or ELO_DEFAULT))
        d = elo.run(res)
        elo.fit_logit(d, y, 0.5 ** ((res["date"].max() - res["date"]).dt.days.to_numpy() / 365))
        poi = PoissonModel(**(hp.get("poi") or POI_DEFAULT)).fit(res, today, team_leagues(res))
        return cls(res, poi, elo, w, np.bincount(y, minlength=3) / len(y), stacker, vote_b, odds_w)

    def predict(self, home, away, mdate):
        h, a = to_short(home), to_short(away)
        md = pd.Timestamp(mdate)
        poi, elo = self.poi, self.elo
        ok_p, ok_e = poi.known(h) and poi.known(a), elo.known(h) and elo.known(a)
        p_poi = poi.probs(h, a)[0] if ok_p else None
        p_elo = elo.probs(h, a) if ok_e else None
        if ok_p and ok_e:
            pb = self.w * p_poi + (1 - self.w) * p_elo; note = ""
        elif ok_p or ok_e:
            pb = p_poi if ok_p else p_elo; note = "片方のモデルのみ"
        else:
            pb = self.base.copy(); note = "データなし（平均値）"
        lam = poi.matrix(h, a)[1:] if ok_p else (None, None)
        fh, fa = team_form(self.res, h, md), team_form(self.res, a, md)
        rest = lambda f: (md - f["last"]).days if f["last"] is not None else None  # noqa
        rh, ra = rest(fh), rest(fa)
        F = {"rest_h": [min(rh if rh is not None else 14, 21)], "rest_a": [min(ra if ra is not None else 14, 21)],
             "form_h": [fh["pts"]], "form_a": [fa["pts"]],
             "lam_h": [lam[0] if lam[0] is not None else np.nan], "lam_a": [lam[1] if lam[1] is not None else np.nan],
             "gap": [0.5 * np.abs(p_poi - p_elo).sum() if ok_p and ok_e else 0.0]}
        p = self.stacker.apply(pb[None, :], F)[0] if (ok_p or ok_e) else pb
        out = {"hs": h, "as": a, "p": p, "p_base": pb, "p_poi": p_poi, "p_elo": p_elo, "lam": lam, "note": note,
               "known": ok_p and ok_e, "scores": poi.top_scores(h, a) if ok_p else [],
               "elo_h": elo.r.get(h), "elo_a": elo.r.get(a),
               "form_h": fh, "form_a": fa, "rest_h": rh, "rest_a": ra,
               "h2h": head_to_head(self.res, h, a, before=md)}
        if ok_p:
            m = poi.matrix(h, a)[0]
            bh, ba = m.sum(1), m.sum(0)
            out["g_h"] = np.array([bh[0], bh[1], bh[2], bh[3:].sum()])
            out["g_a"] = np.array([ba[0], ba[1], ba[2], ba[3:].sum()])
        else:
            out["g_h"] = out["g_a"] = np.array([.25, .35, .24, .16])
        return out

    def blend_vote(self, p, v):
        if not self.vote_b or v is None:
            return p
        q = np.asarray(p) * np.clip(np.asarray(v), 1e-3, 1) ** self.vote_b
        return q / q.sum()


def analyze_card(card, pred: Predictor, votes, odds=None):
    odds = odds or {}
    out = {}
    for kind, g in card.groupby("kind"):
        rows = []
        for _, r in g.sort_values("no").iterrows():
            d = pred.predict(r["home"], r["away"], r["date"])
            d.update({"no": int(r["no"]), "date": r["date"], "time": r.get("time", ""),
                      "home": r["home"], "away": r["away"]})
            rows.append(d)
        v = votes.get(kind)
        if v:
            if kind == "totoGOAL3":
                for i, row in enumerate(rows):
                    row["v_h"], row["v_a"] = np.array(v[2 * i]), np.array(v[2 * i + 1])
            else:
                for row, vv in zip(rows, v):
                    row["v"] = np.array(vv)
                    row["p_model"] = row["p"]
                    row["p"] = pred.blend_vote(row["p"], row["v"])
        # v6市場モデル: toto/miniのみ。市場データがなければ確率は一切変更しない。
        if kind != "totoGOAL3":
            rnd = int(g["round"].iloc[0])
            for row in rows:
                od = odds.get((rnd, kind, int(row["no"])))
                if od is None and kind.startswith("mini"):
                    od = odds.get((rnd, "toto", int(row["no"])))
                row["p_pre_odds"] = np.asarray(row["p"]).copy()
                row["odds"] = od.get("odds") if od else None
                row["p_market"] = od.get("p_market") if od else None
                row["odds_captured_at"] = od.get("captured_at", "") if od else ""
                row["divergence"] = js_divergence(row["p_pre_odds"], row["p_market"])
                row["market_alert"] = bool(row["divergence"] is not None and row["divergence"] >= 0.025)
                row["p"] = log_pool(row["p_pre_odds"], row["p_market"], pred.odds_w)
        out[kind] = rows
    return out


# =============================================================================
# 買い目
# =============================================================================
def slots_of(kind, rows):
    if kind == "totoGOAL3":
        s = []
        for r in rows:
            s.append((np.asarray(r["g_h"]), r.get("v_h"), OUT4))
            s.append((np.asarray(r["g_a"]), r.get("v_a"), OUT4))
        return s
    return [(np.asarray(r["p"]), r.get("v"), OUT3) for r in rows]


def build_ticket(slots, budget_units, alpha=0.0):
    def score(p, v, idxs):
        P = p[idxs].sum()
        if alpha and v is not None:
            return math.log(P) + alpha * math.log(P / max(np.asarray(v)[idxs].sum(), 1e-3))
        return math.log(P)

    picks = [[max(range(len(p)), key=lambda k: score(p, v, [k]))] for p, v, _ in slots]
    while True:
        cost = int(np.prod([len(s) for s in picks]))
        best, best_gain = None, 0.0
        for i, (p, v, _) in enumerate(slots):
            k = len(picks[i])
            if k == len(p) or cost // k * (k + 1) > budget_units:
                continue
            cur = score(p, v, picks[i])
            for c in range(len(p)):
                if c in picks[i]:
                    continue
                gain = (score(p, v, picks[i] + [c]) - cur) / math.log((k + 1) / k)
                if gain > best_gain:
                    best, best_gain = (i, c), gain
        if best is None:
            break
        picks[best[0]].append(best[1])
    return [sorted(s) for s in picks]
