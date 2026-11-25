# -*- coding: utf-8 -*-
"""
toto 予想エンジン   v7（外れから学ぶモデル ＋ 直前情報）
  - データ取得 : くじ対象試合・販売締切 / 投票率 / くじ結果（スポーツくじ公式）、過去結果（Ｊリーグデータサイト）
                 v7: WINNER（1試合予想くじ）のスコア別オッズ ＝ 試合開始10分前まで動く『市場の予想』
                 v7: Ｊリーグ公式の出場停止選手の一覧
  - 基本モデル : ① Dixon-Coles ポアソン（J1/J2/J3 別の得点水準・ホーム有利・引き分け）
                 ② Elo + 順序ロジット（カテゴリ別の引き分け幅・シーズン間の平均回帰）
  - 補正モデル : 外れの診断で見つかった偏りを直す『補正』（Stacker）。v7: 出場停止の差を追加
  - 市場       : WINNERのオッズを確率に直し、予想と重み付きで平均（対数）。重みは記録した試合の
                 答え合わせで学習し、効果が確認できるまでは 0（表示と比較のみ）
  - 投票率     : 予想と toto 投票率を重み付きで平均（市場を使った試合では使わない）
"""
import math
import re
import time
import unicodedata
from datetime import datetime, date, timedelta
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
WINNER_ODDS_URL = f"{BASE}/pl37/spsl023/PGSPSL02301InitGetWinnerOdds.form"
JDATA_FORM_URL = "https://data.j-league.or.jp/SFMS01/"
JDATA_URL = "https://data.j-league.or.jp/SFMS01/search"
SUSP_URL = "https://data.j-league.or.jp/SFIN03/"
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

HP_VER = 2
HP_RETUNE = 30
POI_DEFAULT = {"half_life": 180, "l2": 0.05, "cup_w": 1.0, "tier": False}
ELO_DEFAULT = {"k": 24, "hfa": 55, "regress": 0.0, "cup_w": 1.0}
POI_GRID = [dict(half_life=h, l2=l, cup_w=c, tier=t)
            for h, l, c, t in product([90, 180, 365], [0.05, 0.15], [0.5, 1.0], [True, False])]
ELO_GRID = [dict(k=k, hfa=a, regress=r, cup_w=c)
            for k, a, r, c in product([16, 24, 32], [40, 55, 75], [0.0, 0.33], [0.5, 1.0])]

RESID_DECAY = 0.85
RESID_PRIOR = 3.0
DIVERGE_FLAG = 0.15       # 予想と市場の食い違い（全変動距離）がこれ以上で「荒れ注意」

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
    today = today or now_jst().date()
    if card.empty:
        return card
    last = card.groupby(["round", "kind"])["date"].transform("max")
    alive = card[last >= today]
    c = alive if not alive.empty else card
    pick = c.groupby("kind")["round"].transform("min" if not alive.empty else "max") == c["round"]
    return c[pick].reset_index(drop=True)


latest_rounds = current_rounds


# ---------------------------------------------------------------- 公式「くじ結果」ページ
RES_HEAD_RE = re.compile(r"第\s*(\d+)\s*回\s*(toto|mini\s*toto-A組|mini\s*toto-B組|totoGOAL\d?)\s*くじ結果")
RES3_RE = re.compile(r"(\d{2}/\d{2})\s+(\S+)\s+(\d{1,2})\s+(\S+)\s+(?:(\d+)\s*-\s*(\d+)\s+|-\s+)?(\S+)(?:\s+([012])(?=\s|$))?")
RESG_RE = re.compile(r"(\d{2}/\d{2})\s+(\S+)\s+(\d{1,2})\s+(\S+)\s+(?:(\d+)\s+([0-3])\s+)?(\d{1,2})\s+(\S+)(?:\s+(\d+)\s+([0-3])(?=\s|$))?")
_RESULT_CACHE = {}


def parse_round_result(html_or_text, today=None):
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
    if rnd not in _RESULT_CACHE:
        _RESULT_CACHE[rnd] = parse_round_result(http_get(RESULT_URL, {"holdCntId": rnd}))
    return _RESULT_CACHE[rnd]


def round_finished(g, today=None):
    today = today or now_jst().date()
    if g.empty:
        return False
    if g["hg"].notna().all() and g["ag"].notna().all():
        return True
    return max(g["date"]) < today


def _in_progress_from_results(kind, rnd, today, log=print):
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
        if not round_finished(g, today):
            found = g
            continue
        break
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
            log(f"{kind}: 開催中の第{int(older['round'].iloc[0])}回を表示（販売中の第{rnd}回は結果が出そろってから表示）")
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


# ---------------------------------------------------------------- v7: WINNER のスコア別オッズ
WIN_HEAD_RE = re.compile(r"(\S+)\s*\(ホーム\)\s*試合開始予定\s*(\d{1,2}/\d{1,2})\s*\(\S\)\s*(\d{1,2}:\d{2})\s*(\S+)\s+(\S+)\s*\(アウェイ\)")
WIN_SCORE_RE = re.compile(r"(?<![\d.:/])(\d)\s+-\s+(\d)\s+(\d+(?:\.\d+)?)")
WIN_4UP_RE = re.compile(r"4点以上\s+(\d+(?:\.\d+)?)")
WIN_D3_RE = re.compile(r"3点以上\s+(\d+(?:\.\d+)?)")
WIN_COUNT_RE = re.compile(r"絞り込み結果\s*:?\s*(\d+)\s*件")


def winner_probs(scores, h4, a4, d3):
    """
    18通りのオッズ → 確率（払い戻し率が一定なので 1/オッズ に比例）。
    返り値: p3=[ホーム勝ち,引き分け,アウェイ勝ち], g_h/g_a=各チームの得点 [0,1,2,3以上]
    """
    items = [((h, a), 1 / o) for (h, a), o in scores] + [("H4", 1 / h4), ("A4", 1 / a4), ("D3", 1 / d3)]
    tot = sum(w for _, w in items)
    p3 = np.zeros(3); gh = np.zeros(4); ga = np.zeros(4)
    for k, w in items:
        p = w / tot
        if k == "H4":
            p3[0] += p; gh[3] += p; ga[:3] += p / 3            # 相手の得点は0〜2点に均等配分（近似）
        elif k == "A4":
            p3[2] += p; ga[3] += p; gh[:3] += p / 3
        elif k == "D3":
            p3[1] += p; gh[3] += p; ga[3] += p
        else:
            h, a = k
            p3[0 if h > a else 1 if h == a else 2] += p
            gh[min(h, 3)] += p; ga[min(a, 3)] += p
    return p3, gh, ga


def parse_winner_odds(html_or_text, today=None):
    """WINNER『オッズから選ぶ』ページ → サッカーの各試合の確率"""
    today = today or now_jst().date()
    text = html_text(html_or_text)
    out = []
    for blk in text.split("この試合を購入する"):
        m = WIN_HEAD_RE.search(blk)
        if not m:
            continue
        rest = blk[m.end():]
        sc = WIN_SCORE_RE.findall(rest)
        f4 = WIN_4UP_RE.findall(rest)
        d3 = WIN_D3_RE.findall(rest)
        if len(sc) != 15 or len(f4) != 2 or len(d3) != 1:
            continue
        try:
            scores = [((int(h), int(a)), float(o)) for h, a, o in sc]
            h4, a4, dd = float(f4[0]), float(f4[1]), float(d3[0])
            if min([o for _, o in scores] + [h4, a4, dd]) <= 1.0:
                continue
        except ValueError:
            continue
        d = _match_date(m.group(2), today)
        hh, mm = map(int, m.group(3).split(":"))
        p3, gh, ga = winner_probs(scores, h4, a4, dd)
        out.append({"date": d.isoformat(), "ko": datetime(d.year, d.month, d.day, hh, mm, tzinfo=JST).isoformat(),
                    "home": m.group(1), "away": m.group(5), "hs": to_short(m.group(1)), "as": to_short(m.group(5)),
                    "p3": [round(float(x), 5) for x in p3], "g_h": [round(float(x), 5) for x in gh],
                    "g_a": [round(float(x), 5) for x in ga]})
    return out


def fetch_winner_odds(log=print):
    html = http_get(WINNER_ODDS_URL)
    rows = parse_winner_odds(html)
    cm = WIN_COUNT_RE.search(html_text(html))
    log(f"WINNERオッズ: サッカー {len(rows)}試合を読み取り（ページ表示 {cm.group(1) if cm else '?'}件・バスケ含む）")
    return rows


# ---------------------------------------------------------------- v7: 出場停止選手
SUSP_RE = re.compile(r"(\S+)\s+(J1|J2|J3|C|U-21)\s+\(([a-hz*])\)\s+[^()]*?\((\d{1,2}/\d{1,2})\)")


def parse_suspensions(html_or_text, today=None):
    """Ｊリーグデータサイト『出場停止』ページ → [{player, team, comp, kind, date}]（U-21は除外）"""
    today = today or now_jst().date()
    text = html_text(html_or_text)
    if "出場停止試合" in text:
        text = text.split("出場停止試合", 1)[1]
    text = text.split("・大会")[0]
    out, prev = [], 0
    for m in SUSP_RE.finditer(text):
        player = text[prev:m.start()].strip()
        prev = m.end()
        if m.group(2) == "U-21":
            continue
        out.append({"player": player[-20:], "team": to_short(m.group(1)), "comp": m.group(2),
                    "kind": m.group(3), "date": _match_date(m.group(4), today).isoformat()})
    return out


def fetch_suspensions(log=print):
    rows = parse_suspensions(http_get(SUSP_URL))
    log(f"出場停止: {len(rows)}件を読み取り")
    return rows


def susp_map(hist):
    """[{team, date}] → {"team|YYYY-MM-DD": 人数}"""
    m = {}
    for x in hist or []:
        k = f"{x['team']}|{x['date']}"
        m[k] = m.get(k, 0) + 1
    return m


def susp_count(smap, team, d):
    d = pd.Timestamp(d).date()
    for off in (0, -1, 1):
        v = smap.get(f"{team}|{(d + timedelta(days=off)).isoformat()}")
        if v:
            return v
    return 0


# ---------------------------------------------------------------- Ｊリーグデータサイト（試合結果）
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


def _comp_tier(comp):
    if "百年構想" in comp or "J1" in comp:
        return 0
    if "J2" in comp:
        return 1
    if "J3" in comp:
        return 2
    return None


def match_tiers(df):
    last, out = {}, np.empty(len(df), dtype=int)
    for i, (comp, h, a) in enumerate(df[["comp", "home", "away"]].itertuples(index=False)):
        t = _comp_tier(comp)
        if t is None:
            t = last.get(h, last.get(a, 1))
        else:
            last[h] = last[a] = t
        out[i] = t
    return out


def _cup_mask(df):
    return df["comp"].str.contains(CUP_RE, regex=True).to_numpy() if "comp" in df else np.zeros(len(df), bool)


# =============================================================================
# 順位と懸かっているもの
# =============================================================================
def _league_key(comp):
    if re.search(r"ルヴァン|昇格|入れ替え|参入", comp):
        return None
    t = _comp_tier(comp)
    if t is None:
        return None
    return ("百年構想" if "百年構想" in comp else f"J{t + 1}")


def _stake(pts, played, team, n_teams):
    total = max(2 * (n_teams - 1), 1)
    prog = float(np.mean(list(played.values()))) / total if played else 0.0
    if prog < 0.5 or n_teams < 6:
        return 1.0, prog
    table = sorted(pts.values(), reverse=True)
    p = pts.get(team, 0)
    rem = max(total - played.get(team, 0), 0)
    reach = 3 * rem
    top_line = table[min(5, len(table) - 1)]
    safe_line = table[max(len(table) - 3, 0)]
    rank = 1 + sum(1 for x in table if x > p)
    alive_top = p + reach >= top_line
    danger = rank >= n_teams - 5 and (p - safe_line) <= reach
    return (1.0 if (alive_top or danger) else 0.0), prog


def standings_features(res):
    r = res.reset_index(drop=True)
    sh = np.ones(len(r)); sa = np.ones(len(r)); pg = np.zeros(len(r))
    seasons, team_season = {}, {}
    keys = [_league_key(c) for c in r["comp"]]
    sid, last_d = {}, {}
    season_id = [None] * len(r)
    for i, (k, d) in enumerate(zip(keys, r["date"])):
        if k is None:
            continue
        if k not in sid or (d - last_d[k]).days > 40:
            sid[k] = sid.get(k, -1) + 1
        last_d[k] = d
        season_id[i] = (k, sid[k])
    teams_in = {}
    for i, s in enumerate(season_id):
        if s is not None:
            teams_in.setdefault(s, set()).update([r.at[i, "home"], r.at[i, "away"]])
    for i, s in enumerate(season_id):
        h, a = r.at[i, "home"], r.at[i, "away"]
        if s is None:
            for side, t in (("h", h), ("a", a)):
                ss = team_season.get(t)
                if ss is not None:
                    st = seasons[ss]
                    v, p = _stake(st["pts"], st["played"], t, len(teams_in[ss]))
                    (sh if side == "h" else sa)[i] = v
                    pg[i] = max(pg[i], p)
            continue
        st = seasons.setdefault(s, {"pts": {}, "played": {}})
        n = len(teams_in[s])
        vh, p = _stake(st["pts"], st["played"], h, n)
        va, _ = _stake(st["pts"], st["played"], a, n)
        sh[i], sa[i], pg[i] = vh, va, p
        hg, ag = r.at[i, "hg"], r.at[i, "ag"]
        st["pts"][h] = st["pts"].get(h, 0) + (3 if hg > ag else 1 if hg == ag else 0)
        st["pts"][a] = st["pts"].get(a, 0) + (3 if ag > hg else 1 if hg == ag else 0)
        st["played"][h] = st["played"].get(h, 0) + 1
        st["played"][a] = st["played"].get(a, 0) + 1
        team_season[h] = team_season[a] = s
    now = {}
    for t, s in team_season.items():
        st = seasons[s]
        v, p = _stake(st["pts"], st["played"], t, len(teams_in[s]))
        rank = 1 + sum(1 for x in st["pts"].values() if x > st["pts"].get(t, 0))
        now[t] = {"stake": v, "prog": p, "rank": rank, "n": len(teams_in[s]), "pts": st["pts"].get(t, 0)}
    return pd.DataFrame({"stake_h": sh, "stake_a": sa, "prog": pg}), now


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
    T_EPS = 0.5

    def __init__(self, half_life=180, l2=0.05, cup_w=1.0, tier=True):
        self.half_life, self.l2, self.cup_w, self.use_tier = half_life, l2, cup_w, tier

    def fit(self, df, ref_date, tiers=None):
        teams = sorted(set(df["home"]) | set(df["away"]))
        self.teams = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        self.team_tier = {t: TIER.get(v, 1) for t, v in (tiers or {}).items()}
        tt = np.array([self.team_tier.get(t, 1) for t in teams]) if self.use_tier else np.ones(n, int)
        mt = match_tiers(df)
        h = df["home"].map(self.teams).to_numpy()
        a = df["away"].map(self.teams).to_numpy()
        hg, ag = df["hg"].to_numpy(float), df["ag"].to_numpy(float)
        w = 0.5 ** ((pd.Timestamp(ref_date) - df["date"]).dt.days.clip(lower=0).to_numpy() / self.half_life)
        w = w * np.where(_cup_mask(df), self.cup_w, 1.0)
        l2, eps, te = self.l2, 1e-3, self.T_EPS
        o = 2 * n

        def unpack(p):
            return (p[:n], p[n:o], p[o:o + 3], p[o + 3:o + 6],
                    p[o + 6], p[o + 7:o + 10], p[o + 10], p[o + 11:o + 14])

        def nll(p):
            att, de, ta, td, home0, dh, mu0, dm = unpack(p)
            A, D = att + ta[tt], de + td[tt]
            base = mu0 + dm[mt]
            lh = np.exp(base + home0 + dh[mt] + A[h] + D[a]); la = np.exp(base + A[a] + D[h])
            f = np.sum(w * (lh - hg * np.log(lh) + la - ag * np.log(la)))
            f += l2 * (att @ att + de @ de) + eps * (ta @ ta + td @ td) + te * (dh @ dh + dm @ dm)
            rh, ra = w * (lh - hg), w * (la - ag)
            gA = np.bincount(h, rh, n) + np.bincount(a, ra, n)
            gD = np.bincount(a, rh, n) + np.bincount(h, ra, n)
            g = np.concatenate([gA + 2 * l2 * att, gD + 2 * l2 * de,
                                np.bincount(tt, gA, 3) + 2 * eps * ta, np.bincount(tt, gD, 3) + 2 * eps * td,
                                [rh.sum()], np.bincount(mt, rh, 3) + 2 * te * dh,
                                [rh.sum() + ra.sum()], np.bincount(mt, rh + ra, 3) + 2 * te * dm])
            return f, g

        x0 = np.zeros(o + 14); x0[o + 6] = 0.2; x0[o + 10] = 0.2
        p = minimize(nll, x0, jac=True, method="L-BFGS-B").x
        att, de, ta, td, home0, dh, mu0, dm = unpack(p)
        self.att, self.de = att + ta[tt], de + td[tt]
        self.home_t, self.mu_t = home0 + dh, mu0 + dm
        self.home_adv, self.mu = float(home0 + dh.mean()), float(mu0 + dm.mean())

        lh = np.exp(self.mu_t[mt] + self.home_t[mt] + self.att[h] + self.de[a])
        la = np.exp(self.mu_t[mt] + self.att[a] + self.de[h])

        def best_rho(mask):
            best = (-np.inf, 0.0)
            for rho in np.arange(-0.25, 0.201, 0.01):
                t = _tau(hg[mask], ag[mask], lh[mask], la[mask], rho)
                if np.all(t > 0):
                    best = max(best, (np.sum(w[mask] * np.log(t)), rho))
            return best[1]

        g_rho = best_rho(np.ones(len(df), bool))
        self.rho_t = np.array([best_rho(mt == k) if (mt == k).sum() >= 150 else g_rho for k in range(3)])
        self.rho = g_rho
        return self

    def known(self, t):
        return t in self.teams

    def tier_of(self, home, away):
        return self.team_tier.get(home, self.team_tier.get(away, 1))

    def matrix(self, home, away):
        i, j = self.teams[home], self.teams[away]
        k = self.tier_of(home, away)
        lh = math.exp(self.mu_t[k] + self.home_t[k] + self.att[i] + self.de[j])
        la = math.exp(self.mu_t[k] + self.att[j] + self.de[i])
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        x, y = np.meshgrid(g, g, indexing="ij")
        m *= _tau(x, y, np.full_like(m, lh), np.full_like(m, la), self.rho_t[k])
        m = np.clip(m, 0, None)
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
    T_EPS = 2.0

    def __init__(self, k=24, hfa=55, regress=0.0, cup_w=1.0):
        self.k, self.hfa, self.regress, self.cup_w = k, hfa, regress, cup_w

    def run(self, df):
        self.r, self.init = {}, {}
        diffs = np.empty(len(df))
        last = None
        cup = _cup_mask(df)
        self.mt = match_tiers(df)
        self.team_tier = {}
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
            ct = _comp_tier(comp)
            if ct is not None:
                self.team_tier[h] = self.team_tier[a] = ct
            rh, ra = self.r[h], self.r[a]
            dd = rh + self.hfa - ra
            diffs[idx] = dd
            exp = 1 / (1 + 10 ** (-dd / 400))
            res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            k = self.k * (self.cup_w if cup[idx] else 1.0)
            delta = k * (math.log(abs(hg - ag) + 1) + 1) * (res - exp)
            self.r[h], self.r[a] = rh + delta, ra - delta
        return diffs

    def _cuts(self, p, t):
        c1 = p[1] + p[3:6][t]
        c2 = c1 + np.exp(p[2] + p[6:9][t])
        return c1, c2

    def fit_logit(self, diffs, outcome, w=None, mt=None):
        w = np.ones_like(diffs) if w is None else w
        mt = np.ones(len(diffs), int) if mt is None else np.asarray(mt)
        x = diffs / 100
        te = self.T_EPS

        def nll(p):
            c1, c2 = self._cuts(p, mt)
            pa = _sig(c1 - p[0] * x); pna = _sig(c2 - p[0] * x)
            pr = np.choose(outcome, [1 - pna, pna - pa, pa])
            return -np.sum(w * np.log(np.clip(pr, 1e-9, 1))) / max(w.sum(), 1) * len(w) \
                + te * (p[3:9] @ p[3:9])

        p0 = np.array([0.5, -0.9, -0.5, 0, 0, 0, 0, 0, 0], float)
        self.p = minimize(nll, p0, method="L-BFGS-B").x
        self.b = self.p[0]
        return self

    def probs_from_diff(self, d, t=1):
        x = np.asarray(d, float) / 100
        t = np.broadcast_to(np.asarray(t), x.shape) if np.ndim(x) else int(t)
        c1, c2 = self._cuts(self.p, t)
        pa = _sig(c1 - self.b * x); pna = _sig(c2 - self.b * x)
        return np.stack([1 - pna, pna - pa, pa], axis=-1)

    def known(self, t):
        return t in self.r

    def probs(self, home, away):
        t = self.team_tier.get(home, self.team_tier.get(away, 1))
        return self.probs_from_diff(self.r[home] + self.hfa - self.r[away], t)


def outcome_code(df):
    return np.where(df["hg"] > df["ag"], 0, np.where(df["hg"] == df["ag"], 1, 2))


def metrics(P, y, w=None):
    P = np.clip(np.asarray(P, float), 1e-9, 1)
    y = np.asarray(y)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    ws = max(w.sum(), 1e-9)
    onehot = np.eye(3)[y]
    cp, co = np.cumsum(P, 1)[:, :2], np.cumsum(onehot, 1)[:, :2]
    return {"ll": float(np.sum(w * -np.log(P[np.arange(len(y)), y])) / ws),
            "brier": float(np.sum(w * np.sum((P - onehot) ** 2, axis=1)) / ws),
            "acc": float(np.sum(w * (P.argmax(1) == y)) / ws),
            "rps": float(np.sum(w * np.sum((cp - co) ** 2, axis=1) / 2) / ws),
            "draw_pred": float(np.sum(w * P[:, 1]) / ws), "draw_act": float(np.sum(w * (y == 1)) / ws)}


def log_pool(P, M, a):
    """P^(1-a) × M^a を正規化（行ごと）。P, M は (n,k) または (k,)"""
    P = np.clip(np.asarray(P, float), 1e-9, 1)
    M = np.clip(np.asarray(M, float), 1e-9, 1)
    L = (1 - a) * np.log(P) + a * np.log(M)
    L = L - L.max(-1, keepdims=True)
    Q = np.exp(L)
    return Q / Q.sum(-1, keepdims=True)


def divergence(p, m):
    """予想と市場の食い違い（全変動距離 0〜1）"""
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(m)).sum())


# =============================================================================
# 補正モデル
# =============================================================================
def match_features(res):
    r = res.reset_index(drop=True)
    ph = np.where(r["hg"] > r["ag"], 3, np.where(r["hg"] == r["ag"], 1, 0))
    pa = np.where(r["ag"] > r["hg"], 3, np.where(r["hg"] == r["ag"], 1, 0))
    dh = (r["hg"] == r["ag"]).astype(float).to_numpy()
    long = pd.concat([
        pd.DataFrame({"i": r.index, "side": "h", "team": r["home"], "date": r["date"], "pts": ph, "drw": dh}),
        pd.DataFrame({"i": r.index, "side": "a", "team": r["away"], "date": r["date"], "pts": pa, "drw": dh}),
    ]).sort_values(["team", "date", "i"]).reset_index(drop=True)
    g = long.groupby("team", sort=False)
    long["form"] = g["pts"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum()).fillna(7.0)
    long["drate"] = g["drw"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean()).fillna(0.27)
    long["rest"] = (long["date"] - g["date"].shift(1)).dt.days.fillna(14).clip(0, 21)
    h = long[long["side"] == "h"].set_index("i").sort_index()
    a = long[long["side"] == "a"].set_index("i").sort_index()
    sf, _ = standings_features(r)
    return pd.DataFrame({"rest_h": h["rest"].values, "rest_a": a["rest"].values,
                         "form_h": h["form"].values, "form_a": a["form"].values,
                         "drate_h": h["drate"].values, "drate_a": a["drate"].values,
                         "stake_h": sf["stake_h"].values, "stake_a": sf["stake_a"].values, "prog": sf["prog"].values,
                         "tier": match_tiers(r)})


def team_residuals(df, P):
    S, W = {}, {}
    rh = np.zeros(len(df)); ra = np.zeros(len(df))
    val = lambda t: S.get(t, 0.0) / (W.get(t, 0.0) + RESID_PRIOR)  # noqa
    for i, (h, a, hg, ag) in enumerate(df[["home", "away", "hg", "ag"]].itertuples(index=False)):
        rh[i], ra[i] = val(h), val(a)
        p = P[i]
        exh, exa = 3 * p[0] + p[1], 3 * p[2] + p[1]
        ah = 3 if hg > ag else 1 if hg == ag else 0
        aa = 3 if ag > hg else 1 if hg == ag else 0
        for t, d in ((h, ah - exh), (a, aa - exa)):
            S[t] = RESID_DECAY * S.get(t, 0.0) + d
            W[t] = RESID_DECAY * W.get(t, 0.0) + 1.0
    return rh, ra, {t: val(t) for t in S}


FEAT_INFO = {
    "temperature": ("確率の強弱補正", "予想が全体的に強気すぎ・弱気すぎる傾向を補正"),
    "draw_bias": ("引き分け補正", "引き分けの起きやすさを全体的に補正"),
    "closeness": ("接戦度→引き分け", "実力が拮抗した試合ほど引き分けを多めに見る"),
    "goal_env": ("ロースコア傾向", "両チームの期待得点が低い試合ほど引き分けを多めに見る"),
    "draw_form": ("引き分け癖", "直近10試合で引き分けが多いチーム同士ほど引き分けを多めに見る"),
    "model_gap": ("モデル間の不一致", "2つのモデルの見立てが割れる試合を引き分け寄りに"),
    "draw_j2": ("J2の引き分け補正", "J2の試合の引き分けの出やすさを補正"),
    "draw_j3": ("J3の引き分け補正", "J3の試合の引き分けの出やすさを補正"),
    "dead_rubber": ("消化試合", "両チームとも懸かっているものがない終盤の試合の傾向を補正"),
    "rest_diff": ("休養日差", "前の試合からの間隔の差（連戦による疲労）"),
    "form_diff": ("直近の勢い", "直近5試合の勝点の差"),
    "stake_diff": ("懸かっているものの差", "終盤に優勝・昇格・残留が懸かっている側を有利に見る"),
    "resid_diff": ("調子の過小評価", "最近『予想より勝点を取っている』チームを有利に見る"),
    "home_j2": ("J2のホーム補正", "J2のホーム有利の強さを補正"),
    "home_j3": ("J3のホーム補正", "J3のホーム有利の強さを補正"),
    "susp_diff": ("出場停止", "公式の出場停止選手（警告累積・退場など）が多い側を不利に見る"),
    "market": ("WINNERのオッズ（直前の市場）", "試合開始10分前まで動くWINNERのスコア別オッズを予想に反映"),
    "vote": ("toto投票率（世間の予想）", "toto投票率と『人気の偏り』を予想に反映（市場を使えない試合のみ）"),
}
DRAW_FEATS = ["closeness", "goal_env", "draw_form", "model_gap", "draw_j2", "draw_j3", "dead_rubber"]
SIDE_FEATS = ["rest_diff", "form_diff", "stake_diff", "resid_diff", "home_j2", "home_j3", "susp_diff"]
STACK_FEATS = ["temperature", "draw_bias"] + DRAW_FEATS + SIDE_FEATS
F_KEYS = ["rest_h", "rest_a", "form_h", "form_a", "drate_h", "drate_a", "lam_h", "lam_a", "gap",
          "stake_h", "stake_a", "prog", "tier", "resid_h", "resid_a", "susp_h", "susp_a"]
F_DEFAULT = {"rest_h": 14, "rest_a": 14, "form_h": 7, "form_a": 7, "drate_h": .27, "drate_a": .27,
             "lam_h": np.nan, "lam_a": np.nan, "gap": 0.0, "stake_h": 1, "stake_a": 1, "prog": 0.0,
             "tier": 1, "resid_h": 0.0, "resid_a": 0.0, "susp_h": 0.0, "susp_a": 0.0}


def _fcol(F, k, n):
    v = F.get(k)
    if v is None:
        return np.full(n, F_DEFAULT[k], float)
    return np.asarray(v, float)


def feat_values(P, F):
    n = len(P)
    c = lambda k: _fcol(F, k, n)  # noqa
    lam = np.nan_to_num(c("lam_h") + c("lam_a"), nan=2.6)
    tier = c("tier")
    late = (c("prog") >= 0.5).astype(float)
    return {
        "closeness": 1 - np.abs(P[:, 0] - P[:, 2]),
        "goal_env": 2.6 - lam,
        "draw_form": ((c("drate_h") + c("drate_a")) / 2 - 0.27) * 4,
        "model_gap": np.nan_to_num(c("gap")),
        "draw_j2": (tier == 1).astype(float),
        "draw_j3": (tier == 2).astype(float),
        "dead_rubber": late * (1 - c("stake_h")) * (1 - c("stake_a")),
        "rest_diff": np.clip(c("rest_h") - c("rest_a"), -7, 7) / 7,
        "form_diff": (c("form_h") - c("form_a")) / 15,
        "stake_diff": late * (c("stake_h") - c("stake_a")),
        "resid_diff": np.clip(c("resid_h") - c("resid_a"), -2, 2),
        "home_j2": (tier == 1).astype(float),
        "home_j3": (tier == 2).astype(float),
        "susp_diff": np.clip(c("susp_a") - c("susp_h"), -3, 3) / 2,
    }


class Stacker:
    def __init__(self, active=(), theta=None):
        self.active = [a for a in STACK_FEATS if a in active]
        self.theta = None if theta is None else np.asarray(theta, float)
        if self.theta is not None and len(self.theta) != len(self.names()):
            self.theta = None

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

    def _logits(self, th, P, X, drop=None):
        p = dict(zip(self.names(), th))
        if drop is not None:
            p[drop] = 1.0 if drop == "temperature" else 0.0
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

    def fit(self, P, F, y, w=None, reg=0.01):
        if not self.active:
            self.theta = np.array([])
            return self
        X = feat_values(P, F)
        th0 = self._theta0()
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        ws = w.sum()

        def loss(th):
            Q = self._logits(th, P, X)
            return -np.sum(w * np.log(np.clip(Q[np.arange(len(y)), y], 1e-9, 1))) / ws + reg * np.sum((th - th0) ** 2)

        self.theta = minimize(loss, th0, method="L-BFGS-B").x
        return self

    def apply(self, P, F):
        P = np.atleast_2d(np.asarray(P, float))
        if not self.active or self.theta is None or len(self.theta) == 0:
            return P
        return self._logits(self.theta, P, feat_values(P, F))

    def explain(self, P, F):
        P = np.atleast_2d(np.asarray(P, float))
        if not self.active or self.theta is None or len(self.theta) == 0:
            return []
        X = feat_values(P, F)
        full = self._logits(self.theta, P, X)[0]
        out = []
        for f in self.names():
            wo = self._logits(self.theta, P, X, drop=f)[0]
            d = full - wo
            if np.abs(d).max() >= 0.005:
                out.append((f, d))
        return sorted(out, key=lambda x: -np.abs(x[1]).max())

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


def cv_score(active, P, F, y, w=None, k=5):
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    idx = np.arange(len(y))
    chunks = np.array_split(idx, k)
    Q, te_all = [], []
    for j in range(1, k):
        tr, te = np.concatenate(chunks[:j]), chunks[j]
        st = Stacker(active).fit(P[tr], subF(F, tr), y[tr], w[tr])
        Q.append(st.apply(P[te], subF(F, te)))
        te_all.append(te)
    Q, te_all = np.vstack(Q), np.concatenate(te_all)
    return metrics(Q, y[te_all], w[te_all]), Q, te_all


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
    elo.fit_logit(diffs[:s_idx], y[:s_idx], w, elo.mt[:s_idx])
    return elo.probs_from_diff(diffs[s_idx:], elo.mt[s_idx:])


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
    P_base = best_w * P_poi + (1 - best_w) * P_elo

    mf_all = match_features(res)
    mf = mf_all.iloc[s_idx:].reset_index(drop=True)
    oos_df = res.iloc[s_idx:].reset_index(drop=True)
    rh, ra, resid_now = team_residuals(oos_df, P_base)
    F = {k: mf[k].values for k in ["rest_h", "rest_a", "form_h", "form_a", "drate_h", "drate_a",
                                   "stake_h", "stake_a", "prog", "tier"]}
    F.update({"lam_h": LAM[:, 0], "lam_a": LAM[:, 1], "gap": 0.5 * np.abs(P_poi - P_elo).sum(1),
              "resid_h": rh, "resid_a": ra})
    oos = {"P_poi": P_poi, "P_elo": P_elo, "P_base": P_base, "y": yt, "F": F, "base": base,
           "dates": oos_df["date"], "home": oos_df["home"].values, "away": oos_df["away"].values,
           "tier": mf["tier"].values}

    pred = Predictor.fit(res, today, w=best_w, hp={"elo": ecfg, "poi": pcfg}, resid=resid_now)
    new_hp = {"ver": HP_VER, "n": len(res) if retune else hp["n"], "elo": ecfg, "poi": pcfg}
    return {"poisson": pred.poi, "elo": pred.elo, "w": best_w, "oos": oos, "hp": new_hp, "retuned": retune,
            "hp_report": report, "test_period": (oos["dates"].min(), oos["dates"].max(), len(yt)),
            "base": np.bincount(y, minlength=3) / len(y), "resid": resid_now, "standing": pred.standing}


# =============================================================================
# 予想
# =============================================================================
def team_form(res, team, before, n=5):
    g = res[((res["home"] == team) | (res["away"] == team)) & (res["date"] < before)]
    g10 = g.tail(10)
    g = g.tail(n)
    marks, gf, ga, pts = [], 0, 0, 0
    for _, r in g.iterrows():
        f, a = (r["hg"], r["ag"]) if r["home"] == team else (r["ag"], r["hg"])
        gf += f; ga += a
        marks.append("W" if f > a else "D" if f == a else "L")
        pts += 3 if f > a else 1 if f == a else 0
    last = g["date"].max() if len(g) else None
    drate = float((g10["hg"] == g10["ag"]).mean()) if len(g10) >= 3 else 0.27
    return {"marks": "".join(marks), "gf": int(gf), "ga": int(ga), "pts": int(pts) if len(g) else 7,
            "last": last, "drate": drate}


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


def blend_vote(p, v, b=0.0, c=0.0):
    if (not b and not c) or v is None:
        return np.asarray(p)
    q = np.clip(np.asarray(p, float), 1e-9, 1) ** (1 - b) * np.clip(np.asarray(v, float), 1e-3, 1) ** b
    q = q * np.array([1.0, math.exp(c), 1.0])
    return q / q.sum()


class Predictor:
    def __init__(self, res, poi, elo, w, base, stacker=None, vote_b=0.0, vote_c=0.0, resid=None, standing=None,
                 susp=None, market_a=0.0, market_ag=0.0):
        self.res, self.poi, self.elo, self.w, self.base = res, poi, elo, w, base
        self.stacker = stacker or Stacker()
        self.vote_b, self.vote_c = vote_b, vote_c
        self.resid = resid or {}
        self.standing = standing or {}
        self.susp = susp or {}
        self.market_a, self.market_ag = market_a, market_ag

    @classmethod
    def fit(cls, res, today, w=0.5, stacker=None, vote_b=0.0, hp=None, vote_c=0.0, resid=None):
        hp = hp or {}
        res = res.sort_values("date").reset_index(drop=True)
        y = outcome_code(res)
        elo = EloModel(**(hp.get("elo") or ELO_DEFAULT))
        d = elo.run(res)
        elo.fit_logit(d, y, 0.5 ** ((res["date"].max() - res["date"]).dt.days.to_numpy() / 365), elo.mt)
        poi = PoissonModel(**(hp.get("poi") or POI_DEFAULT)).fit(res, today, team_leagues(res))
        _, standing = standings_features(res)
        return cls(res, poi, elo, w, np.bincount(y, minlength=3) / len(y), stacker, vote_b, vote_c, resid, standing)

    def features(self, h, a, md, lam, p_poi, p_elo, fh, fa, rh, ra):
        sh, sa = self.standing.get(h, {}), self.standing.get(a, {})
        t = self.poi.tier_of(h, a) if hasattr(self.poi, "team_tier") else 1
        return {"rest_h": min(rh if rh is not None else 14, 21), "rest_a": min(ra if ra is not None else 14, 21),
                "form_h": fh["pts"], "form_a": fa["pts"], "drate_h": fh["drate"], "drate_a": fa["drate"],
                "lam_h": lam[0] if lam[0] is not None else np.nan, "lam_a": lam[1] if lam[1] is not None else np.nan,
                "gap": 0.5 * np.abs(p_poi - p_elo).sum() if p_poi is not None and p_elo is not None else 0.0,
                "stake_h": sh.get("stake", 1.0), "stake_a": sa.get("stake", 1.0),
                "prog": max(sh.get("prog", 0.0), sa.get("prog", 0.0)), "tier": t,
                "resid_h": self.resid.get(h, 0.0), "resid_a": self.resid.get(a, 0.0),
                "susp_h": float(susp_count(self.susp, h, md)), "susp_a": float(susp_count(self.susp, a, md))}

    def predict(self, home, away, mdate, light=False):
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
        Fs = self.features(h, a, md, lam, p_poi, p_elo, fh, fa, rh, ra)
        F1 = {k: [v] for k, v in Fs.items()}
        known = ok_p or ok_e
        p = self.stacker.apply(pb[None, :], F1)[0] if known else pb
        out = {"hs": h, "as": a, "p": p, "p_base": pb, "known": ok_p and ok_e, "F": Fs}
        if ok_p:
            m = poi.matrix(h, a)[0]
            bh, ba = m.sum(1), m.sum(0)
            out["g_h"] = np.array([bh[0], bh[1], bh[2], bh[3:].sum()])
            out["g_a"] = np.array([ba[0], ba[1], ba[2], ba[3:].sum()])
        else:
            out["g_h"] = out["g_a"] = np.array([.25, .35, .24, .16])
        if light:
            return out
        expl = self.stacker.explain(pb[None, :], F1) if known else []
        out.update({"p_poi": p_poi, "p_elo": p_elo, "lam": lam, "note": note,
                    "scores": poi.top_scores(h, a) if ok_p else [],
                    "elo_h": elo.r.get(h), "elo_a": elo.r.get(a),
                    "form_h": fh, "form_a": fa, "rest_h": rh, "rest_a": ra,
                    "reasons": [(f, [float(x) for x in d]) for f, d in expl],
                    "stand_h": self.standing.get(h), "stand_a": self.standing.get(a),
                    "h2h": head_to_head(self.res, h, a, before=md)})
        return out

    def blend_vote(self, p, v):
        return blend_vote(p, v, self.vote_b, self.vote_c)


def market_lookup(market, hs, as_, d):
    """WINNER の記録から (ホーム, アウェイ, 日付±1日) で探す"""
    if not market:
        return None
    d = pd.Timestamp(d).date()
    for off in (0, -1, 1):
        x = market.get(f"{(d + timedelta(days=off)).isoformat()}|{hs}|{as_}")
        if x:
            return x
    return None


def analyze_card(card, pred: Predictor, votes, market=None):
    """market: {"YYYY-MM-DD|ホーム|アウェイ": {"p3","g_h","g_a"}}（WINNERの最新オッズ）"""
    out = {}
    for kind, g in card.groupby("kind"):
        rows = []
        for _, r in g.sort_values("no").iterrows():
            d = pred.predict(r["home"], r["away"], r["date"])
            d.update({"no": int(r["no"]), "date": r["date"], "time": r.get("time", ""),
                      "home": r["home"], "away": r["away"]})
            mk = market_lookup(market, d["hs"], d["as"], r["date"])
            if mk:
                d["m"] = np.asarray(mk["p3"]); d["m_g_h"] = np.asarray(mk["g_h"]); d["m_g_a"] = np.asarray(mk["g_a"])
            rows.append(d)
        v = votes.get(kind)
        if kind == "totoGOAL3":
            for i, row in enumerate(rows):
                if v:
                    row["v_h"], row["v_a"] = np.array(v[2 * i]), np.array(v[2 * i + 1])
                if row.get("m_g_h") is not None and pred.market_ag > 0:
                    row["g_h"] = log_pool(row["g_h"], row["m_g_h"], pred.market_ag)
                    row["g_a"] = log_pool(row["g_a"], row["m_g_a"], pred.market_ag)
                    row["market_used"] = True
        else:
            for i, row in enumerate(rows):
                row["p_model"] = row["p"]
                if v and i < len(v):
                    row["v"] = np.array(v[i])
                if row.get("m") is not None:
                    row["div"] = divergence(row["p"], row["m"])
                if row.get("m") is not None and pred.market_a > 0:
                    before = row["p"]
                    row["p"] = log_pool(row["p"], row["m"], pred.market_a)
                    row["market_used"] = True
                    dd = np.asarray(row["p"]) - np.asarray(before)
                    if np.abs(dd).max() >= 0.005:
                        row["reasons"] = row["reasons"] + [("market", [float(x) for x in dd])]
                elif row.get("v") is not None and (pred.vote_b or pred.vote_c):
                    before = row["p"]
                    row["p"] = pred.blend_vote(row["p"], row["v"])
                    dd = np.asarray(row["p"]) - np.asarray(before)
                    if np.abs(dd).max() >= 0.005:
                        row["reasons"] = row["reasons"] + [("vote", [float(x) for x in dd])]
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
