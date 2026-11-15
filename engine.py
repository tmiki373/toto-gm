# -*- coding: utf-8 -*-
"""
toto 予想エンジン
  - データ取得 : くじ対象試合・販売締切 / 投票率（スポーツくじ公式）、過去結果（Ｊリーグデータサイト）
  - モデル     : ① Dixon-Coles ポアソン  ② Elo + 順序ロジット  → バックテストで重み付けしたアンサンブル
  - 分析       : 直近成績・対戦成績・中何日・投票率との比較（妙味）
  - 買い目     : 予算内で「的中率」と「配当妙味」のバランスを取って自動配分
"""
import math
import re
import time
import unicodedata
from datetime import datetime, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

JST = ZoneInfo("Asia/Tokyo")
BASE = "https://store.toto-dream.com/dcs/subos/screen"
TOTO_INFO_URL = f"{BASE}/pi01/spin000/PGSPIN00001DisptotoLotInfo.form"
VOTE_URL = f"{BASE}/pi09/spin003/PGSPIN00301InitVoteRate.form"
JDATA_FORM_URL = "https://data.j-league.or.jp/SFMS01/"
JDATA_URL = "https://data.j-league.or.jp/SFMS01/search"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal toto analysis)"}
REQUEST_INTERVAL = 2.0
KINDS = ["toto", "mini toto-A組", "mini toto-B組", "totoGOAL3"]
COMMODITY = {"toto": "01", "totoGOAL3": "02", "mini toto-A組": "01", "mini toto-B組": "01"}
SEASONS = ["2025", "2026特別", "2026/27"]       # 画面の表示名（実際の検索値はフォームから自動取得）
CACHE_VER = "v2"
HALF_LIFE_DAYS = 180
MAX_GOALS = 10
OUT3 = ["1", "0", "2"]
OUT4 = ["0", "1", "2", "3"]
KEEP_COMP = "J1|J2|J3|百年構想|ルヴァン|昇格|入れ替え|参入"
DROP_COMP = "U-21|ユース|エリート|育成|サテライト|オールスター|ドリーム|ワールドチャレンジ|国際"

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
    cols = ["round", "kind", "no", "date", "time", "stadium", "home", "away", "deadline"]
    return pd.DataFrame(rows, columns=cols).drop_duplicates(["round", "kind", "no"])


def latest_rounds(card: pd.DataFrame, today=None) -> pd.DataFrame:
    """くじ種別ごとに『まだ試合が残っている最新の回』だけ残す"""
    today = today or now_jst().date()
    if card.empty:
        return card
    alive = card.groupby(["round", "kind"])["date"].transform("max") >= today
    c = card[alive] if alive.any() else card
    keep = c.groupby("kind")["round"].transform("max") == c["round"]
    return c[keep].reset_index(drop=True)


def fetch_card(round_no=None):
    html = http_get(TOTO_INFO_URL, {"holdCntId": round_no} if round_no else None)
    card = parse_toto_card(html)
    if card.empty:
        raise RuntimeError("くじ情報ページから試合を読み取れませんでした（ページ構成変更の可能性）")
    return card[card["round"] == round_no].reset_index(drop=True) if round_no else latest_rounds(card)


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


# ---------------------------------------------------------------- Ｊリーグデータサイト
def parse_form_options(html):
    """検索フォームの選択肢を {name: [(value, 表示名), ...]} で返す（select / checkbox / radio 両対応）"""
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
    """シーズン表示名 → 検索値、対象大会 → competition_frame_ids の値 を決める"""
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
    """
    シーズン × 大会ごとに検索（1回の検索は最大1,500件のため分割）。
    終了済みシーズンは保存して再利用、最新シーズンは毎回取り直す。
    """
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
        targets = frames or [(None, "全大会")]
        for fid, lab in targets:
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
            df = df.drop_duplicates(["date", "home", "away"])
            df.to_csv(cache, index=False, encoding="utf-8-sig")
            log(f"シーズン {s}: {len(df)} 試合（最新 {df['date'].max():%Y/%m/%d}）")
        else:
            log(f"シーズン {s}: 0 試合")
        out.append(df)
    out = [f for f in out if not f.empty]
    if not out:
        raise RuntimeError("過去の試合結果を取得できませんでした")
    res = pd.concat(out, ignore_index=True).drop_duplicates(["date", "home", "away"])
    return res.sort_values("date").reset_index(drop=True)


def team_leagues(res):
    """各チームの現在の所属カテゴリ（直近のリーグ戦から判定）"""
    lg = res[~res["comp"].str.contains("ルヴァン|昇格|入れ替え|参入")].copy()
    lg["lv"] = lg["comp"].str.extract(r"J([123])")[0]
    lg = lg.dropna(subset=["lv"])
    long = pd.concat([lg[["date", "home", "lv"]].rename(columns={"home": "team"}),
                      lg[["date", "away", "lv"]].rename(columns={"away": "team"})])
    return long.sort_values("date").groupby("team")["lv"].last().map(lambda x: f"J{x}").to_dict()


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
    def __init__(self, half_life=HALF_LIFE_DAYS, l2=0.05):
        self.half_life, self.l2 = half_life, l2

    def fit(self, df, ref_date):
        teams = sorted(set(df["home"]) | set(df["away"]))
        self.teams = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = df["home"].map(self.teams).to_numpy()
        a = df["away"].map(self.teams).to_numpy()
        hg, ag = df["hg"].to_numpy(float), df["ag"].to_numpy(float)
        w = 0.5 ** ((pd.Timestamp(ref_date) - df["date"]).dt.days.clip(lower=0).to_numpy() / self.half_life)
        l2 = self.l2

        def nll(p):
            att, de, home, mu = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
            lh = np.exp(mu + home + att[h] + de[a]); la = np.exp(mu + att[a] + de[h])
            f = np.sum(w * (lh - hg * np.log(lh) + la - ag * np.log(la))) + l2 * (att @ att + de @ de)
            rh, ra = w * (lh - hg), w * (la - ag)
            g = np.concatenate([np.bincount(h, rh, n) + np.bincount(a, ra, n) + 2 * l2 * att,
                                np.bincount(a, rh, n) + np.bincount(h, ra, n) + 2 * l2 * de,
                                [rh.sum()], [rh.sum() + ra.sum()]])
            return f, g

        x0 = np.zeros(2 * n + 2); x0[-2:] = 0.2
        p = minimize(nll, x0, jac=True, method="L-BFGS-B").x
        self.att, self.de, self.home_adv, self.mu = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
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
                             "攻撃力": np.exp(self.att), "守備力(失点倍率)": np.exp(self.de)})


# =============================================================================
# モデル② Elo + 順序ロジット
# =============================================================================
def _sig(x):
    return 1 / (1 + np.exp(-x))


class EloModel:
    def __init__(self, k=24, hfa=55):
        self.k, self.hfa = k, hfa

    def run(self, df):
        self.r = {}
        diffs = np.empty(len(df))
        for idx, (comp, h, a, hg, ag) in enumerate(df[["comp", "home", "away", "hg", "ag"]].itertuples(index=False)):
            init = 1600 if "J1" in comp else 1400 if "J3" in comp else 1500
            rh, ra = self.r.setdefault(h, init), self.r.setdefault(a, init)
            d = rh + self.hfa - ra
            diffs[idx] = d
            exp = 1 / (1 + 10 ** (-d / 400))
            res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            delta = self.k * (math.log(abs(hg - ag) + 1) + 1) * (res - exp)
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
# 学習 + バックテスト
# =============================================================================
def _metrics(P, y):
    P = np.clip(P, 1e-9, 1)
    onehot = np.eye(3)[y]
    return {"対数損失": float(-np.mean(np.log(P[np.arange(len(y)), y]))),
            "Brier": float(np.mean(np.sum((P - onehot) ** 2, axis=1))),
            "的中率(本命)": float(np.mean(P.argmax(1) == y))}


def train(res: pd.DataFrame, today=None, test_frac=0.2):
    today = pd.Timestamp(today or now_jst().date())
    res = res.sort_values("date").reset_index(drop=True)
    y = outcome_code(res)
    cut = int(len(res) * (1 - test_frac))
    tr, te = res.iloc[:cut], res.iloc[cut:]
    elo = EloModel()
    diffs = elo.run(res)
    wtr = 0.5 ** ((tr["date"].max() - tr["date"]).dt.days.to_numpy() / 365)
    elo.fit_logit(diffs[:cut], y[:cut], wtr)
    P_elo = elo.probs_from_diff(diffs[cut:])
    pm = PoissonModel().fit(tr, tr["date"].max())
    base = np.bincount(y[:cut], minlength=3) / cut
    P_poi = np.array([pm.probs(h, a)[0] if pm.known(h) and pm.known(a) else base
                      for h, a in zip(te["home"], te["away"])])
    yte = y[cut:]
    best_w, best_ll = 0.5, np.inf
    for wgt in np.linspace(0, 1, 11):
        ll = _metrics(wgt * P_poi + (1 - wgt) * P_elo, yte)["対数損失"]
        if ll < best_ll:
            best_w, best_ll = wgt, ll
    P_ens = best_w * P_poi + (1 - best_w) * P_elo
    bt = pd.DataFrame({
        "基準(平均)": _metrics(np.tile(base, (len(yte), 1)), yte),
        "ポアソン": _metrics(P_poi, yte), "Elo": _metrics(P_elo, yte),
        "アンサンブル": _metrics(P_ens, yte),
    }).T
    conf = P_ens.max(1)
    bins = pd.cut(conf, [0, .4, .5, .6, .7, 1.0])
    calib = pd.DataFrame({"予測確率帯": bins, "予測": conf, "実際": P_ens.argmax(1) == yte}) \
        .groupby("予測確率帯", observed=True).agg(試合数=("予測", "size"), 平均予測=("予測", "mean"),
                                                    実際の的中率=("実際", "mean")).reset_index()
    calib["予測確率帯"] = calib["予測確率帯"].map(lambda iv: f"{iv.left:.0%}〜{iv.right:.0%}")

    elo_all = EloModel(); d_all = elo_all.run(res)
    w_all = 0.5 ** ((res["date"].max() - res["date"]).dt.days.to_numpy() / 365)
    elo_all.fit_logit(d_all, y, w_all)
    poi_all = PoissonModel().fit(res, today)
    return {"poisson": poi_all, "elo": elo_all, "w": best_w, "backtest": bt, "calib": calib,
            "test_period": (te["date"].min(), te["date"].max(), len(te)),
            "base": np.bincount(y, minlength=3) / len(y)}


# =============================================================================
# 試合ごとの分析
# =============================================================================
def team_form(res, team, before, n=5):
    g = res[((res["home"] == team) | (res["away"] == team)) & (res["date"] < before)].tail(n)
    marks, gf, ga = [], 0, 0
    for _, r in g.iterrows():
        f, a = (r["hg"], r["ag"]) if r["home"] == team else (r["ag"], r["hg"])
        gf += f; ga += a
        marks.append("W" if f > a else "D" if f == a else "L")
    last = g["date"].max() if len(g) else None
    return {"marks": "".join(marks), "gf": int(gf), "ga": int(ga), "last": last}


def head_to_head(res, h, a, n=5):
    g = res[((res["home"] == h) & (res["away"] == a)) | ((res["home"] == a) & (res["away"] == h))].tail(n)
    w = d = l = 0
    for _, r in g.iterrows():
        f, x = (r["hg"], r["ag"]) if r["home"] == h else (r["ag"], r["hg"])
        w += f > x; d += f == x; l += f < x
    return f"{w}勝{d}分{l}敗" if len(g) else "対戦なし"


def analyze_card(card, res, M, votes):
    poi, elo, wgt, base = M["poisson"], M["elo"], M["w"], M["base"]
    out = {}
    for kind, g in card.groupby("kind"):
        g = g.sort_values("no")
        rows = []
        for _, r in g.iterrows():
            h, a = to_short(r["home"]), to_short(r["away"])
            md = pd.Timestamp(r["date"])
            ok_p, ok_e = poi.known(h) and poi.known(a), elo.known(h) and elo.known(a)
            p_poi = poi.probs(h, a)[0] if ok_p else None
            p_elo = elo.probs(h, a) if ok_e else None
            if ok_p and ok_e:
                p = wgt * p_poi + (1 - wgt) * p_elo; note = ""
            elif ok_p or ok_e:
                p = p_poi if ok_p else p_elo; note = "片方のモデルのみ"
            else:
                p = base.copy(); note = "データなし（平均値）"
            lam = poi.matrix(h, a)[1:] if ok_p else (None, None)
            fh, fa = team_form(res, h, md), team_form(res, a, md)
            rest = lambda f: (md - f["last"]).days if f["last"] is not None else None  # noqa
            row = {"no": int(r["no"]), "date": r["date"], "time": r.get("time", ""),
                   "home": r["home"], "away": r["away"], "hs": h, "as": a,
                   "p": p, "p_poi": p_poi, "p_elo": p_elo, "lam": lam, "note": note,
                   "scores": poi.top_scores(h, a) if ok_p else [],
                   "elo_h": elo.r.get(h), "elo_a": elo.r.get(a),
                   "form_h": fh, "form_a": fa, "rest_h": rest(fh), "rest_a": rest(fa),
                   "h2h": head_to_head(res, h, a)}
            if kind == "totoGOAL3":
                if ok_p:
                    m = poi.matrix(h, a)[0]
                    bh, ba = m.sum(1), m.sum(0)
                    row["g_h"] = np.array([bh[0], bh[1], bh[2], bh[3:].sum()])
                    row["g_a"] = np.array([ba[0], ba[1], ba[2], ba[3:].sum()])
                else:
                    row["g_h"] = row["g_a"] = np.array([.25, .35, .24, .16])
            rows.append(row)
        v = votes.get(kind)
        if v:
            if kind == "totoGOAL3":
                for i, row in enumerate(rows):
                    row["v_h"], row["v_a"] = np.array(v[2 * i]), np.array(v[2 * i + 1])
            else:
                for row, vv in zip(rows, v):
                    row["v"] = np.array(vv)
        out[kind] = rows
    return out
