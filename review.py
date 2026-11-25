# -*- coding: utf-8 -*-
"""
予想の保存（アーカイブ）と振り返り   v7
  - 毎回の予想を保存し、販売締切を過ぎたら確定（以後は上書きしない）
  - 振り返りは『公式くじ結果が1試合でも出ている一番新しい回』（途中経過 → 最終結果）
  - 外れた試合に『原因の候補タグ』を付ける。v7: 市場（WINNER）の1番人気が当てていた / 本命側に出場停止
  - 外れの記録（ledger）… 終わった回の全試合を、試合前の予想・特徴・結果ごと保存して学習に使う
  - v7: WINNER のオッズを毎回記録（market_hist）… 全Ｊリーグ試合の『予想・市場・結果』を貯めて市場の重みを学習
  - v7: 出場停止の一覧を記録（susp_hist）… 公式ページは現在分しか出ないため、蓄積して学習に使う
"""
import base64
import json
import os

import numpy as np
import pandas as pd
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import engine as E

ITER = 250_000
BASE3 = np.array([0.43, 0.27, 0.30])
KEEP_ROUNDS = 40
LOOKBACK = 6
VOTE_HIST_ROUNDS = 24
MARKET_KEEP = 4000
SUSP_KEEP = 4000

TAG_INFO = {
    "draw_miss": "引き分けの見逃し",
    "upset": "番狂わせ（結果の予想確率20%未満）",
    "close": "もともと混戦の予想",
    "goal_short": "本命側の得点が想定を大きく下回った",
    "fav_tired": "本命が連戦で休養日が短かった",
    "stake_gap": "懸かっているものの差（勝った側だけに懸かっていた）",
    "dead_rubber": "両チームとも消化試合",
    "form_under": "勝った側を予想が過小評価していた（最近の勝点が予想以上）",
    "vote_right": "toto投票の1番人気がこの結果だった",
    "market_right": "WINNERのオッズ（市場）の1番人気がこの結果だった",
    "fav_susp": "本命側に出場停止の選手がいた",
    "model_split": "2つのモデルの見立てが割れていた",
    "vote_drift": "締切前に投票がこの結果の方向へ動いていた",
}


# ---------------------------------------------------------------- 暗号化
def encrypt(obj, password):
    salt, iv = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, json.dumps(obj, ensure_ascii=False, default=_jd).encode(), None)
    b = lambda x: base64.b64encode(x).decode()  # noqa
    return {"salt": b(salt), "iv": b(iv), "ct": b(ct), "iter": ITER}


def decrypt(enc, password):
    d = lambda s: base64.b64decode(s)  # noqa
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=d(enc["salt"]),
                     iterations=enc["iter"]).derive(password.encode())
    return json.loads(AESGCM(key).decrypt(d(enc["iv"]), d(enc["ct"]), None))


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(type(o))


def pages_url():
    if os.environ.get("PAGES_URL"):
        return os.environ["PAGES_URL"].rstrip("/") + "/"
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in repo:
        return None
    owner, name = repo.split("/", 1)
    host = f"https://{owner.lower()}.github.io/"
    return host if name.lower() == f"{owner.lower()}.github.io" else f"{host}{name}/"


# ---------------------------------------------------------------- 状態の読み書き
def load_state(data_dir, password, log=print):
    p = data_dir / "state.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa
            log(f"保存済みの状態を読めませんでした: {e}")
    url = pages_url()
    if url and password:
        try:
            import requests
            r = requests.get(url + "state.json", timeout=30)
            if r.ok:
                st = decrypt(r.json(), password)
                log("公開サイトから過去の予想・学習記録を復元しました")
                return st
        except Exception as e:  # noqa
            log(f"公開サイトからの復元なし: {e}")
    return {}


def save_state(state, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    arch = state.get("archive", {})
    if len(arch) > KEEP_ROUNDS * 4:
        for k in sorted(arch, key=lambda k: int(k.split("|")[0]))[: len(arch) - KEEP_ROUNDS * 4]:
            arch.pop(k)
    vh = state.get("vote_hist", {})
    if len(vh) > 60:
        for k in sorted(vh, key=int)[: len(vh) - 60]:
            vh.pop(k)
    vt = state.get("vote_track", {})
    if len(vt) > 12:
        for k in sorted(vt, key=lambda k: int(k.split("|")[0]))[: len(vt) - 12]:
            vt.pop(k)
    mh = state.get("market_hist", {})
    if len(mh) > MARKET_KEEP:
        for k in sorted(mh)[: len(mh) - MARKET_KEEP]:
            mh.pop(k)
    state["susp_hist"] = state.get("susp_hist", [])[-SUSP_KEEP:]
    state["ledger"] = state.get("ledger", [])[-1500:]
    (data_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False, default=_jd), encoding="utf-8")


# ---------------------------------------------------------------- v7: 直前情報の記録
def record_susp(state, rows):
    """出場停止の一覧を蓄積（同じ選手・チーム・日付は1件）"""
    hist = state.setdefault("susp_hist", [])
    seen = {f"{x['player']}|{x['team']}|{x['date']}" for x in hist}
    n = 0
    for r in rows or []:
        k = f"{r['player']}|{r['team']}|{r['date']}"
        if k not in seen:
            hist.append({k2: r[k2] for k2 in ("player", "team", "comp", "kind", "date")})
            seen.add(k)
            n += 1
    return n


def record_market(state, wm, pred, now=None):
    """
    WINNER の最新オッズと、その時点の予想（市場を混ぜる前）を試合ごとに上書き保存。
    試合開始後は更新しない（締切直前の値が残る）。最初に見たオッズも残す（動きの分析用）。
    """
    now = now or E.now_jst()
    mh = state.setdefault("market_hist", {})
    n = 0
    for w in wm or []:
        if pd.Timestamp(w["ko"]) <= pd.Timestamp(now):
            continue
        if not (pred.poi.known(w["hs"]) and pred.poi.known(w["as"])):
            continue
        k = f"{w['date']}|{w['hs']}|{w['as']}"
        try:
            d = pred.predict(w["hs"], w["as"], w["date"], light=True)
        except Exception:  # noqa
            continue
        e = mh.get(k) or {"date": w["date"], "hs": w["hs"], "as": w["as"], "m_first": w["p3"],
                          "t_first": now.isoformat()}
        e.update({"ko": w["ko"], "m": w["p3"], "mg_h": w["g_h"], "mg_a": w["g_a"], "t": now.isoformat(),
                  "p": [round(float(x), 5) for x in d["p"]],
                  "g_h": [round(float(x), 5) for x in d["g_h"]], "g_a": [round(float(x), 5) for x in d["g_a"]]})
        mh[k] = e
        n += 1
    return n


def market_now(wm):
    """表示・予想用: 最新オッズを {date|hs|as: entry} に"""
    return {f"{w['date']}|{w['hs']}|{w['as']}": w for w in (wm or [])}


def market_saved(state):
    """記録済みの市場（試合開始前の最終値）も使えるように {date|hs|as: entry}"""
    return {k: {"p3": v["m"], "g_h": v["mg_h"], "g_a": v["mg_a"]} for k, v in state.get("market_hist", {}).items()}


# ---------------------------------------------------------------- アーカイブ
def _num(x):
    try:
        x = float(x)
        return None if np.isnan(x) else round(x, 4)
    except Exception:
        return None


def _ser_row(r):
    keys = ["no", "time", "home", "away", "hs", "as", "p", "p_model", "p_base", "p_poi", "p_elo", "v", "lam",
            "rest_h", "rest_a", "g_h", "g_a", "v_h", "v_a", "h2h", "note", "m", "m_g_h", "m_g_a", "div"]
    d = {k: r.get(k) for k in keys}
    d["date"] = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
    for k in ["p", "p_model", "p_base", "p_poi", "p_elo", "v", "g_h", "g_a", "v_h", "v_a", "m", "m_g_h", "m_g_a"]:
        if d[k] is not None:
            d[k] = [round(float(x), 5) for x in d[k]]
    d["div"] = _num(d["div"])
    d["lam"] = [None if x is None else round(float(x), 3) for x in (r.get("lam") or (None, None))]
    for side in ("h", "a"):
        f = r.get(f"form_{side}") or {}
        d[f"form_{side}"] = {"marks": f.get("marks", ""), "pts": int(f.get("pts", 7)),
                             "gf": int(f.get("gf", 0)), "ga": int(f.get("ga", 0)),
                             "drate": round(float(f.get("drate", 0.27)), 3)}
    d["F"] = {k: _num(v) for k, v in (r.get("F") or {}).items()}
    d["reasons"] = [[f, [round(float(x), 4) for x in dd]] for f, dd in (r.get("reasons") or [])]
    return d


def make_entry(kind, rnd, deadline, rows, now, reconstructed=False):
    tickets = E.build_ticket(E.slots_of(kind, rows), 10, 0.0)
    return {"round": int(rnd), "kind": kind, "deadline": deadline, "saved": now.isoformat(),
            "reconstructed": reconstructed, "frozen": reconstructed,
            "rows": [_ser_row(r) for r in rows], "ticket": tickets}


def archive_current(state, card, analysis, log=print):
    arch = state.setdefault("archive", {})
    now = E.now_jst()
    for kind, rows in analysis.items():
        g = card[card["kind"] == kind]
        rnd = int(g["round"].iloc[0])
        dl = g["deadline"].dropna().iloc[0] if g["deadline"].notna().any() else None
        key = f"{rnd}|{kind}"
        locked = (dl and now >= pd.Timestamp(dl)) or (now.date() >= min(g["date"]))
        ex = arch.get(key)
        if ex and ex.get("frozen"):
            continue
        if locked and ex:
            ex["frozen"] = True
            log(f"第{rnd}回 {kind}: 予想を確定")
            continue
        arch[key] = make_entry(kind, rnd, dl, rows, now)
        arch[key]["frozen"] = bool(locked)


def record_votes(state, card, votes):
    v = votes.get("toto")
    g = card[card["kind"] == "toto"]
    if not v or g.empty:
        return
    key = f"{int(g['round'].iloc[0])}|toto"
    tr = state.setdefault("vote_track", {}).setdefault(key, [])
    snap = [[round(float(x), 4) for x in vv] for vv in v]
    if not tr or tr[-1]["v"] != snap:
        tr.append({"t": E.now_jst().isoformat(), "v": snap})
    if len(tr) > 40:
        del tr[1:len(tr) - 39]


def vote_drift(state, rnd):
    tr = state.get("vote_track", {}).get(f"{rnd}|toto")
    if not tr or len(tr) < 2:
        return None
    a, b = np.array(tr[0]["v"]), np.array(tr[-1]["v"])
    if a.shape != b.shape:
        return None
    return (b - a).tolist()


# ---------------------------------------------------------------- 公式結果
class ResultCache:
    def __init__(self, log=print):
        self.d, self.log = {}, log

    def get(self, rnd):
        if rnd not in self.d:
            try:
                self.d[rnd] = E.fetch_round_result(rnd)
            except Exception as e:  # noqa
                self.log(f"第{rnd}回のくじ結果を取得できず: {e}")
                self.d[rnd] = None
        return self.d[rnd]


def _rows_of(c, kind, r):
    if c is None or c.empty:
        return pd.DataFrame()
    return c[(c["kind"] == kind) & (c["round"] == r)].sort_values("no")


def _official(g):
    out = {}
    for _, m in g.iterrows():
        ok = not (pd.isna(m["hg"]) or pd.isna(m["ag"]))
        out[str(int(m["no"]))] = [int(m["hg"]), int(m["ag"])] if ok else None
    return out


def _predictor_for(res, first, L, hp=None, w=None):
    hist = res[res["date"] < first]
    if len(hist) < 300:
        raise RuntimeError("その回より前の試合データが不足")
    return E.Predictor.fit(hist, first, w=L.get("w", 0.5) if w is None else w,
                           stacker=E.Stacker.from_dict(L.get("stacker")), vote_b=L.get("vote_b", 0.0),
                           hp=hp or L.get("hp"), vote_c=L.get("vote_c", 0.0))


def find_previous(state, cur, rc, res, log=print):
    arch = state.setdefault("archive", {})
    prev = {}
    for kind, rnd in cur.items():
        for r in range(rnd, rnd - 1 - LOOKBACK, -1):
            c = rc.get(r)
            g = _rows_of(c, kind, r)
            if g.empty or g["hg"].isna().all():
                continue
            key = f"{r}|{kind}"
            if key not in arch:
                try:
                    arch[key] = reconstruct(kind, r, c, g, res, state, log)
                except Exception as e:  # noqa
                    log(f"第{r}回 {kind} の予想の再現に失敗: {e}")
                    break
            arch[key]["official"] = _official(g)
            arch[key]["finished"] = bool(E.round_finished(g))
            prev[kind] = key
            break
        else:
            log(f"{kind}: 振り返れる回のくじ結果が見つかりません（第{rnd - LOOKBACK}〜{rnd}回）")
    return prev


def reconstruct(kind, r, c, g, res, state, log):
    first = pd.Timestamp(min(g["date"]))
    pred = _predictor_for(res, first, state.get("learning", {}))
    rows = []
    for _, m in g.iterrows():
        d = pred.predict(m["home"], m["away"], m["date"])
        d.update({"no": int(m["no"]), "date": m["date"], "time": "", "home": m["home"], "away": m["away"]})
        mk = E.market_lookup(market_saved(state), d["hs"], d["as"], m["date"])
        if mk:
            d["m"] = np.asarray(mk["p3"]); d["m_g_h"] = np.asarray(mk["g_h"]); d["m_g_a"] = np.asarray(mk["g_a"])
        rows.append(d)
    vk = "toto" if kind.startswith("mini") else kind
    try:
        src = _rows_of(c, vk, r)
        n = len(src) * (2 if vk == "totoGOAL3" else 1)
        v = E.fetch_vote_rates(r, vk, n) if n else None
        if v and kind == "totoGOAL3":
            for i, row in enumerate(rows):
                row["v_h"], row["v_a"] = np.array(v[2 * i]), np.array(v[2 * i + 1])
        elif v:
            key = {(E.to_short(h), E.to_short(a)): vv for (h, a), vv in zip(zip(src["home"], src["away"]), v)}
            for row in rows:
                vv = key.get((row["hs"], row["as"]))
                if vv:
                    row["v"] = np.array(vv)
                    row["p_model"] = row["p"]
                    row["p"] = pred.blend_vote(row["p"], row["v"])
    except Exception as e:  # noqa
        log(f"第{r}回 {kind} の投票率なし: {e}")
    log(f"第{r}回 {kind}: 試合前のデータだけで予想を再現して保存")
    return make_entry(kind, r, None, rows, E.now_jst(), reconstructed=True)


def vote_history(state, cur_round, rc, res, hp, w, log=print):
    vh = state.setdefault("vote_hist", {})
    L = state.get("learning", {})
    if state.get("vote_hist_ver") != 2:
        vh.clear()
        state["vote_hist_ver"] = 2
    added = 0
    for r in range(cur_round, cur_round - 1 - VOTE_HIST_ROUNDS, -1):
        if str(r) in vh:
            continue
        c = rc.get(r)
        if c is None:
            continue
        g = _rows_of(c, "toto", r)
        if g.empty:
            vh[str(r)] = []
            continue
        if not E.round_finished(g):
            continue
        g = g.dropna(subset=["hg", "ag"])
        try:
            v = E.fetch_vote_rates(r, "toto", len(_rows_of(c, "toto", r)))
        except Exception as e:  # noqa
            log(f"第{r}回の投票率を取得できず: {e}")
            continue
        if not v:
            vh[str(r)] = []
            continue
        try:
            pred = _predictor_for(res, pd.Timestamp(min(g["date"])), {**L, "vote_b": 0, "vote_c": 0}, hp=hp, w=w)
        except Exception as e:  # noqa
            log(f"第{r}回の予想再現をスキップ: {e}")
            vh[str(r)] = []
            continue
        vmap = dict(zip(_rows_of(c, "toto", r)["no"], v))
        rows = []
        for _, m in g.iterrows():
            d = pred.predict(m["home"], m["away"], m["date"], light=True)
            vv = vmap.get(m["no"])
            if not d["known"] or vv is None:
                continue
            hg, ag = int(m["hg"]), int(m["ag"])
            rows.append({"round": r, "home": m["home"], "away": m["away"], "date": str(m["date"]),
                         "v": [round(float(x), 4) for x in vv], "p": [round(float(x), 5) for x in d["p"]],
                         "a": 0 if hg > ag else 1 if hg == ag else 2})
        vh[str(r)] = rows
        added += 1
    n = sum(len(x) for x in vh.values())
    log(f"投票率の学習データ: {n}試合（今回追加 {added}回分）")
    return added


def _score(e, r, res):
    off = e.get("official")
    if off is not None and str(r["no"]) in off:
        return tuple(off[str(r["no"])]) if off[str(r["no"])] else None
    d = pd.Timestamp(r["date"])
    g = res[(res["home"] == r["hs"]) & (res["away"] == r["as"]) & ((res["date"] - d).abs() <= pd.Timedelta(days=3))]
    if g.empty:
        return None
    m = g.iloc[(g["date"] - d).abs().argsort().iloc[0]]
    return int(m["hg"]), int(m["ag"])


# ---------------------------------------------------------------- 外れの原因タグとコメント
def _pct(x):
    return f"{x * 100:.0f}%"


def miss_tags(r, pick, act, hg, ag, drift=None):
    p = np.asarray(r["p"])
    F = r.get("F") or {}
    t = []
    if act == 1:
        t.append("draw_miss")
    if p[act] < 0.2:
        t.append("upset")
    if max(p) < 0.45:
        t.append("close")
    lam = r.get("lam") or [None, None]
    if lam[0] is not None and pick != 1:
        fav_g, fav_l = (hg, lam[0]) if pick == 0 else (ag, lam[1])
        if fav_g - fav_l <= -1.0:
            t.append("goal_short")
    rh, ra = r.get("rest_h"), r.get("rest_a")
    if rh is not None and ra is not None and pick != 1:
        fr, orr = (rh, ra) if pick == 0 else (ra, rh)
        if fr <= 4 and orr - fr >= 3:
            t.append("fav_tired")
    prog = F.get("prog") or 0
    sh, sa = F.get("stake_h"), F.get("stake_a")
    if prog >= 0.5 and sh is not None and sa is not None:
        if sh == 0 and sa == 0:
            t.append("dead_rubber")
        elif act != 1 and ((act == 0 and sh > sa) or (act == 2 and sa > sh)):
            t.append("stake_gap")
    rsh, rsa = F.get("resid_h"), F.get("resid_a")
    if act != 1 and rsh is not None and rsa is not None:
        if (rsh - rsa if act == 0 else rsa - rsh) >= 0.3:
            t.append("form_under")
    if pick != 1:
        fs, os_ = (F.get("susp_h") or 0, F.get("susp_a") or 0) if pick == 0 else (F.get("susp_a") or 0, F.get("susp_h") or 0)
        if fs >= 1 and fs > os_:
            t.append("fav_susp")
    v = r.get("v")
    if v and int(np.argmax(v)) == act:
        t.append("vote_right")
    m = r.get("m")
    if m and int(np.argmax(m)) == act:
        t.append("market_right")
    pp, pe = r.get("p_poi"), r.get("p_elo")
    if pp and pe and int(np.argmax(pp)) != int(np.argmax(pe)):
        t.append("model_split")
    if drift is not None and drift[act] >= 0.04:
        t.append("vote_drift")
    return t


def comment_3way(r, pick, act, hg, ag, tags, drift=None):
    p = np.asarray(r["p"])
    pa = p[act]
    c = []
    kind = "大番狂わせ" if pa < 0.2 else "やや意外な結果" if pa < 0.3 else "十分ありえた結果"
    c.append(f"予想 {E.OUT3[pick]}（{_pct(p[pick])}）→ 結果 {E.OUT3[act]}。この結果の予想確率は{_pct(pa)}で「{kind}」。")
    lam = r.get("lam") or [None, None]
    if lam[0] is not None:
        for team, g, l in ((r["home"], hg, lam[0]), (r["away"], ag, lam[1])):
            if g - l <= -1.2:
                c.append(f"{team}の攻撃が不発（期待{l:.1f}点 → {g}点）。")
            elif g - l >= 1.5:
                c.append(f"{team}が想定以上に得点（期待{l:.1f}点 → {g}点）。")
    F = r.get("F") or {}
    if "dead_rubber" in tags:
        c.append("両チームとも順位的に懸かっているものが少ない終盤の試合でした。")
    if "stake_gap" in tags:
        c.append("勝った側にだけ優勝・昇格・残留などが懸かっていました。")
    if "form_under" in tags:
        rs = F.get("resid_h") if act == 0 else F.get("resid_a")
        c.append(f"勝った側は最近、予想より1試合あたり約{rs:+.1f}勝点多く取っており、予想が実力に追いついていなかった可能性。")
    if "fav_tired" in tags:
        c.append("本命側が連戦で休養日が短い状況でした。")
    if "fav_susp" in tags:
        n = int(F.get("susp_h") or 0) if pick == 0 else int(F.get("susp_a") or 0)
        c.append(f"本命側は出場停止の選手が{n}人いました。")
    if "draw_miss" in tags:
        fh, fa = r.get("form_h") or {}, r.get("form_a") or {}
        c.append(f"引き分け（{hg}-{ag}）。モデルの引き分け確率は{_pct(p[1])}でした。"
                 + (f"両チームとも直近は引き分けが多い傾向（{_pct(fh.get('drate', 0))}・{_pct(fa.get('drate', 0))}）。"
                    if fh.get("drate", 0) + fa.get("drate", 0) >= 0.8 else ""))
    m = r.get("m")
    if m:
        mm = int(np.argmax(m))
        if mm == act:
            c.append(f"WINNERのオッズ（試合直前の市場）は{_pct(m[act])}でこの結果を最有力と見ていました。"
                     "直前の情報（メンバー・コンディション）を市場が先に織り込んでいた可能性。")
        elif mm == pick:
            c.append(f"WINNERのオッズも{_pct(m[pick])}で同じ予想。市場も外した波乱でした。")
    v = r.get("v")
    if v:
        maj = int(np.argmax(v))
        if maj == act and not (m and int(np.argmax(m)) == act):
            c.append(f"toto投票は{_pct(v[act])}がこの結果を予想していました。")
        elif maj == pick and not m:
            c.append(f"toto投票の{_pct(v[pick])}も同じ予想で、多くの人が外した波乱でした。")
    if "vote_drift" in tags:
        c.append(f"販売期間中に、この結果への投票が{drift[act] * 100:+.0f}pt動いていました。")
    if "model_split" in tags:
        pp, pe = r["p_poi"], r["p_elo"]
        c.append(f"2つのモデルの見立てが割れていました（ポアソン:{E.OUT3[int(np.argmax(pp))]} / Elo:{E.OUT3[int(np.argmax(pe))]}）。")
    return c


def comment_goal(team, g_pred, picks, actual_g, lam):
    b = min(actual_g, 3)
    lab = lambda k: "3点以上" if k == 3 else f"{k}点"  # noqa
    c = [f"予想 {'/'.join(lab(k) for k in picks)} → 結果 {lab(b)}（この結果の予想確率{_pct(g_pred[b])}）。"]
    if lam is not None:
        if actual_g - lam >= 1.5:
            c.append(f"{team}が期待（{lam:.1f}点）を大きく上回る得点。")
        elif actual_g - lam <= -1.2:
            c.append(f"{team}は期待（{lam:.1f}点）を下回り得点が伸びず。")
    return c


# ---------------------------------------------------------------- 振り返り
def review_entry(e, res, drift=None):
    rows, known, hits, ll, ll_b, t_ok, t_miss = [], 0, 0, [], [], True, 0
    tag_count = {}
    draws = exp_draws = fav_hits = fav_n = mk_hits = mk_n = 0.0
    if e["kind"] == "totoGOAL3":
        for i, r in enumerate(e["rows"]):
            sc = _score(e, r, res)
            for j, (team, gk, li) in enumerate(((r["home"], "g_h", 0), (r["away"], "g_a", 1))):
                picks = e["ticket"][2 * i + j]
                item = {"no": f"{r['no']}{'H' if j == 0 else 'A'}", "team": team, "match": f"{r['home']} vs {r['away']}",
                        "date": r["date"], "p": r[gk], "picks": picks, "actual": None, "m": r.get("m_" + gk)}
                if sc:
                    g = sc[j]; b = min(g, 3); known += 1
                    item.update({"actual": b, "goals": g, "score": f"{sc[0]}-{sc[1]}", "hit": b in picks, "p_act": r[gk][b]})
                    hits += b in picks
                    ll.append(-np.log(max(r[gk][b], 1e-9)))
                    if b not in picks:
                        t_ok = False; t_miss += 1
                        item["comments"] = comment_goal(team, r[gk], picks, g, (r.get("lam") or [None, None])[li])
                rows.append(item)
    else:
        for i, r in enumerate(e["rows"]):
            p = np.asarray(r["p"]); pick = int(np.argmax(p)); picks = e["ticket"][i]
            sc = _score(e, r, res)
            dr = drift[i] if drift is not None and i < len(drift) else None
            item = {"no": r["no"], "home": r["home"], "away": r["away"], "date": r["date"], "p": r["p"], "v": r.get("v"),
                    "m": r.get("m"), "pick": pick, "picks": picks, "actual": None, "drift": dr}
            if sc:
                hg, ag = sc; act = 0 if hg > ag else 1 if hg == ag else 2
                known += 1
                hit = act == pick
                hits += hit
                draws += act == 1; exp_draws += p[1]
                if r.get("v"):
                    fav_n += 1; fav_hits += int(np.argmax(r["v"])) == act
                if r.get("m"):
                    mk_n += 1; mk_hits += int(np.argmax(r["m"])) == act
                ll.append(-np.log(max(p[act], 1e-9))); ll_b.append(-np.log(BASE3[act]))
                item.update({"actual": act, "score": f"{hg}-{ag}", "hit": hit, "in_ticket": act in picks,
                             "p_act": float(p[act])})
                if act not in picks:
                    t_ok = False; t_miss += 1
                if not hit:
                    tags = miss_tags(r, pick, act, hg, ag, dr)
                    item["tags"] = tags
                    item["comments"] = comment_3way(r, pick, act, hg, ag, tags, dr)
                    for tg in tags:
                        tag_count[tg] = tag_count.get(tg, 0) + 1
            rows.append(item)
    n = len(rows)
    if known == 0:
        ticket = "結果待ち"
    elif t_ok is False:
        ticket = f"外れ（{t_miss}か所）"
    elif known == n:
        ticket = "的中"
    else:
        ticket = "判定待ち"
    s = {"round": e["round"], "kind": e["kind"], "reconstructed": e.get("reconstructed", False),
         "n": n, "known": known, "hits": int(hits), "ticket": ticket,
         "ticket_units": int(np.prod([len(t) for t in e["ticket"]])),
         "ll": float(np.mean(ll)) if ll else None, "ll_base": float(np.mean(ll_b)) if ll_b else None,
         "tags": tag_count, "draws": int(draws), "exp_draws": round(float(exp_draws), 1),
         "fav_hits": int(fav_hits), "fav_n": int(fav_n), "mk_hits": int(mk_hits), "mk_n": int(mk_n),
         "complete": known == n or bool(e.get("finished"))}
    s["summary"] = summary_text(s)
    return {"summary": s, "rows": rows}


def summary_text(s):
    if not s["known"]:
        return "まだ結果が出ていません。"
    unit = "枠" if s["kind"] == "totoGOAL3" else "試合"
    t = ["【途中経過】" + f"{s['n']}{unit}中{s['known']}{unit}の結果が確定。" if s["known"] < s["n"] else "【最終結果】"]
    t.append(f"結果が出た{s['known']}{unit}のうち、{'買い目' if s['kind'] == 'totoGOAL3' else '本命'}は{s['hits']}{unit}的中"
             f"（{s['hits'] / s['known']:.0%}）。1,000円の買い目は「{s['ticket']}」。")
    if s["ll"] is not None and s["ll_base"] is not None:
        t.append(f"確率の当てはまり（対数損失）は {s['ll']:.3f}（平均的な予想なら {s['ll_base']:.3f}）で、"
                 f"{'平均より良い' if s['ll'] < s['ll_base'] else '平均より悪い'}結果。")
    if s["kind"] != "totoGOAL3":
        if s["draws"] - s["exp_draws"] >= 2:
            t.append(f"引き分けが{s['draws']}試合と、想定（約{s['exp_draws']:.1f}試合）を大きく上回る荒れた回。")
        elif s["exp_draws"] - s["draws"] >= 2:
            t.append(f"引き分けは{s['draws']}試合と、想定（約{s['exp_draws']:.1f}試合）より少ない回。")
        if s.get("mk_n"):
            t.append(f"WINNERのオッズの1番人気どおりは{s['mk_n']}試合中{s['mk_hits']}試合。")
        if s["fav_n"]:
            t.append(f"toto投票の1番人気どおりは{s['fav_n']}試合中{s['fav_hits']}試合。")
        if s["tags"]:
            top = sorted(s["tags"].items(), key=lambda x: -x[1])[:3]
            t.append("外れの主な原因候補: " + "、".join(f"{TAG_INFO[k]}（{v}）" for k, v in top) + "。")
    return "".join(t)


# ---------------------------------------------------------------- 外れの記録（学習用）
def update_ledger(state, keys, reviews):
    led = state.setdefault("ledger", [])
    seen = {(x["home"], x["away"], x["date"]) for x in led}
    arch = state.get("archive", {})
    added = 0
    for kind, key in keys.items():
        if kind == "totoGOAL3" or kind not in reviews:
            continue
        rv = reviews[kind]
        if not rv["summary"]["complete"]:
            continue
        e = arch[key]
        for r, it in zip(e["rows"], rv["rows"]):
            k = (r["hs"], r["as"], r["date"])
            if k in seen or it.get("actual") is None or not r.get("p_base") or not r.get("F"):
                continue
            seen.add(k)
            led.append({"round": e["round"], "kind": kind, "home": r["hs"], "away": r["as"], "date": r["date"],
                        "p_base": r["p_base"], "p": r["p"], "F": r["F"], "a": it["actual"],
                        "real": not e.get("reconstructed", False), "tags": it.get("tags", []),
                        "v": r.get("v"), "m": r.get("m")})
            added += 1
    return added


def triggers(reviews):
    m = {"draw_miss": ["closeness", "goal_env", "draw_form", "draw_j2", "draw_j3"],
         "dead_rubber": ["dead_rubber"], "stake_gap": ["stake_diff"], "form_under": ["resid_diff"],
         "fav_tired": ["rest_diff"], "model_split": ["model_gap"], "vote_right": ["vote"],
         "market_right": ["market"], "fav_susp": ["susp_diff"]}
    cnt = {}
    for rv in reviews:
        for k, v in rv["summary"].get("tags", {}).items():
            cnt[k] = cnt.get(k, 0) + v
    out = []
    for tg, n in sorted(cnt.items(), key=lambda x: -x[1]):
        for f in m.get(tg, []):
            out.append((f, f"{TAG_INFO[tg]}が{n}試合"))
    return out
