# -*- coding: utf-8 -*-
"""
予想の保存（アーカイブ）と振り返り   v4
  - 毎回の予想を保存し、販売締切を過ぎたら確定（以後は上書きしない）
  - 直前の回は、公式「くじ結果」ページから対象試合とスコアを取得して照合
    （予想が保存されていない回は、試合前のデータだけで予想を再現）
  - 外れた試合に分析コメントを付け、外れ方の傾向から次の検討要素を提案
  - 過去回の「投票率・試合前の予想・結果」を集め、投票率を混ぜる比率の学習に使う
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
LOOKBACK = 6           # 直前の回を探す範囲
VOTE_HIST_ROUNDS = 12  # 投票率の学習用に集める過去の回数


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
    if len(vh) > 40:
        for k in sorted(vh, key=int)[: len(vh) - 40]:
            vh.pop(k)
    (data_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False, default=_jd), encoding="utf-8")


# ---------------------------------------------------------------- アーカイブ
def _ser_row(r):
    keys = ["no", "time", "home", "away", "hs", "as", "p", "p_model", "p_base", "p_poi", "p_elo", "v", "lam",
            "rest_h", "rest_a", "g_h", "g_a", "v_h", "v_a", "h2h", "note"]
    d = {k: r.get(k) for k in keys}
    d["date"] = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
    for k in ["p", "p_model", "p_base", "p_poi", "p_elo", "v", "g_h", "g_a", "v_h", "v_a"]:
        if d[k] is not None:
            d[k] = [round(float(x), 5) for x in d[k]]
    d["lam"] = [None if x is None else round(float(x), 3) for x in (r.get("lam") or (None, None))]
    for side in ("h", "a"):
        f = r.get(f"form_{side}") or {}
        d[f"form_{side}"] = {"marks": f.get("marks", ""), "pts": int(f.get("pts", 7)),
                             "gf": int(f.get("gf", 0)), "ga": int(f.get("ga", 0))}
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


# ---------------------------------------------------------------- 公式結果
class ResultCache:
    """公式『くじ結果』ページを回ごとに1回だけ取得"""

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
                           hp=hp or L.get("hp"))


def find_previous(state, cur, rc, res, log=print):
    arch = state.setdefault("archive", {})
    prev = {}
    for kind, rnd in cur.items():
        for r in range(rnd - 1, rnd - 1 - LOOKBACK, -1):
            c = rc.get(r)
            g = _rows_of(c, kind, r)
            if g.empty:
                continue
            key = f"{r}|{kind}"
            if key not in arch:
                try:
                    arch[key] = reconstruct(kind, r, c, g, res, state, log)
                except Exception as e:  # noqa
                    log(f"第{r}回 {kind} の予想の再現に失敗: {e}")
                    break
            arch[key]["official"] = _official(g)
            prev[kind] = key
            break
        else:
            log(f"{kind}: 直前の回のくじ結果が見つかりません（第{rnd - LOOKBACK}〜{rnd - 1}回）")
    return prev


def reconstruct(kind, r, c, g, res, state, log):
    first = pd.Timestamp(min(g["date"]))
    pred = _predictor_for(res, first, state.get("learning", {}))
    rows = []
    for _, m in g.iterrows():
        d = pred.predict(m["home"], m["away"], m["date"])
        d.update({"no": int(m["no"]), "date": m["date"], "time": "", "home": m["home"], "away": m["away"]})
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
    """
    過去 VOTE_HIST_ROUNDS 回の toto について『試合前データだけの予想・最終投票率・結果』を集める。
    一度集めた回は保存して再利用（毎回の取得は新しく終わった回だけ）。
    """
    vh = state.setdefault("vote_hist", {})
    L = state.get("learning", {})
    added = 0
    for r in range(cur_round - 1, cur_round - 1 - VOTE_HIST_ROUNDS, -1):
        if str(r) in vh:
            continue
        c = rc.get(r)
        if c is None:
            continue
        g = _rows_of(c, "toto", r)
        if g.empty:
            vh[str(r)] = []                 # toto のない回（totoGOAL3 だけの回など）
            continue
        if g["hg"].isna().any() or g["ag"].isna().any():
            continue                        # 結果待ち → 次回また確認
        try:
            v = E.fetch_vote_rates(r, "toto", len(g))
        except Exception as e:  # noqa
            log(f"第{r}回の投票率を取得できず: {e}")
            continue
        if not v:
            vh[str(r)] = []
            continue
        try:
            pred = _predictor_for(res, pd.Timestamp(min(g["date"])), L, hp=hp, w=w)
        except Exception as e:  # noqa
            log(f"第{r}回の予想再現をスキップ: {e}")
            vh[str(r)] = []
            continue
        rows = []
        for (_, m), vv in zip(g.iterrows(), v):
            d = pred.predict(m["home"], m["away"], m["date"])
            if not d["known"]:
                continue                    # 海外の試合などモデルが知らない試合は除外
            hg, ag = int(m["hg"]), int(m["ag"])
            rows.append({"home": m["home"], "away": m["away"], "date": str(m["date"]),
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


# ---------------------------------------------------------------- コメント
def _pct(x):
    return f"{x * 100:.0f}%"


def comment_3way(r, pick, act, hg, ag):
    p = np.asarray(r["p"])
    pa = p[act]
    c = []
    kind = "大番狂わせ" if pa < 0.2 else "やや意外な結果" if pa < 0.3 else "十分ありえた結果"
    c.append(f"予想 {E.OUT3[pick]}（{_pct(p[pick])}）→ 結果 {E.OUT3[act]}。この結果の予想確率は{_pct(pa)}で「{kind}」。")
    if max(p) < 0.45:
        c.append("もともと本命が45%未満の混戦予想で、外れても不思議ではない試合でした。")
    lam = r.get("lam") or [None, None]
    if lam[0] is not None:
        for team, g, l in ((r["home"], hg, lam[0]), (r["away"], ag, lam[1])):
            if g - l <= -1.2:
                c.append(f"{team}の攻撃が不発（期待{l:.1f}点 → {g}点）。")
            elif g - l >= 1.5:
                c.append(f"{team}が想定以上に得点（期待{l:.1f}点 → {g}点）。")
    if act == 1:
        cl = 1 - abs(p[0] - p[2])
        c.append(f"引き分け（{hg}-{ag}）。モデルの引き分け確率は{_pct(p[1])}" +
                 ("で、実力差が小さく引き分けも十分ありえる試合でした。" if cl > 0.8 else "でした。"))
    v = r.get("v")
    if v:
        maj = int(np.argmax(v))
        if maj == act:
            c.append(f"世間の投票は{_pct(v[act])}がこの結果を予想していました。モデルが見ていない情報"
                     "（けが人・出場停止・モチベーションなど）が効いた可能性があります。")
        elif maj == pick:
            c.append(f"世間の{_pct(v[pick])}も同じ予想で、多くの人が外した波乱でした。")
    fh, fa = r.get("form_h", {}), r.get("form_a", {})
    if act != 1 and fh and fa:
        wt, wf, lt, lf = (r["home"], fh, r["away"], fa) if act == 0 else (r["away"], fa, r["home"], fh)
        if wf.get("pts", 7) - lf.get("pts", 7) >= 4:
            c.append(f"{wt}は直近5試合の勝点で上回っていました（{wf['pts']} 対 {lf['pts']}）。勢いの差が出た可能性。")
    rh, ra = r.get("rest_h"), r.get("rest_a")
    if rh is not None and ra is not None and pick != 1:
        fav_rest, other = (rh, ra) if pick == 0 else (ra, rh)
        fav = r["home"] if pick == 0 else r["away"]
        if fav_rest <= 4 and other - fav_rest >= 3:
            c.append(f"本命の{fav}は中{fav_rest}日（相手は中{other}日）で、連戦の疲労が影響した可能性。")
    pp, pe = r.get("p_poi"), r.get("p_elo")
    if pp and pe and int(np.argmax(pp)) != int(np.argmax(pe)):
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
def review_entry(e, res):
    rows, known, hits, ll, ll_b, t_ok, t_miss = [], 0, 0, [], [], True, 0
    miss_types = {"draw": 0, "upset": 0}
    if e["kind"] == "totoGOAL3":
        for i, r in enumerate(e["rows"]):
            sc = _score(e, r, res)
            for j, (team, gk, li) in enumerate(((r["home"], "g_h", 0), (r["away"], "g_a", 1))):
                picks = e["ticket"][2 * i + j]
                item = {"no": f"{r['no']}{'H' if j == 0 else 'A'}", "team": team, "match": f"{r['home']} vs {r['away']}",
                        "date": r["date"], "p": r[gk], "picks": picks, "actual": None}
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
            item = {"no": r["no"], "home": r["home"], "away": r["away"], "date": r["date"], "p": r["p"], "v": r.get("v"),
                    "pick": pick, "picks": picks, "actual": None}
            if sc:
                hg, ag = sc; act = 0 if hg > ag else 1 if hg == ag else 2
                known += 1
                hit = act == pick
                hits += hit
                ll.append(-np.log(max(p[act], 1e-9))); ll_b.append(-np.log(BASE3[act]))
                item.update({"actual": act, "score": f"{hg}-{ag}", "hit": hit, "in_ticket": act in picks,
                             "p_act": float(p[act])})
                if act not in picks:
                    t_ok = False; t_miss += 1
                if not hit:
                    item["comments"] = comment_3way(r, pick, act, hg, ag)
                    miss_types["draw"] += act == 1
                    miss_types["upset"] += p[act] < 0.2
                    item["feat"] = {"rest_h": r.get("rest_h"), "rest_a": r.get("rest_a"),
                                    "form_h": (r.get("form_h") or {}).get("pts"), "form_a": (r.get("form_a") or {}).get("pts"),
                                    "vote_right": bool(r.get("v") and int(np.argmax(r["v"])) == act),
                                    "model_split": bool(r.get("p_poi") and r.get("p_elo") and
                                                        int(np.argmax(r["p_poi"])) != int(np.argmax(r["p_elo"]))),
                                    "act": act, "pick": pick}
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
         "miss_types": {k: int(v) for k, v in miss_types.items()}, "complete": known == n}
    s["summary"] = summary_text(s)
    return {"summary": s, "rows": rows}


def summary_text(s):
    if not s["known"]:
        return "まだ結果が出ていません。"
    unit = "枠" if s["kind"] == "totoGOAL3" else "試合"
    t = []
    if not s["complete"]:
        t.append(f"{s['n']}{unit}中{s['known']}{unit}の結果が確定（残りは結果待ち）。")
    t.append(f"結果が出た{s['known']}{unit}のうち、{'買い目' if s['kind'] == 'totoGOAL3' else '本命'}は{s['hits']}{unit}的中"
             f"（{s['hits'] / s['known']:.0%}）。1,000円の買い目は「{s['ticket']}」。")
    if s["ll"] is not None and s["ll_base"] is not None:
        better = s["ll"] < s["ll_base"]
        t.append(f"確率の当てはまり（対数損失）は {s['ll']:.3f}、平均的な予想なら {s['ll_base']:.3f} で、"
                 f"{'平均より良い' if better else '平均より悪い'}結果でした。")
    misses = s["known"] - s["hits"]
    if misses and s["kind"] != "totoGOAL3":
        t.append(f"外れ{misses}試合のうち、引き分けの見逃し{s['miss_types']['draw']}、"
                 f"予想確率20%未満の番狂わせ{s['miss_types']['upset']}。")
    return "".join(t)


# ---------------------------------------------------------------- 外れ方の傾向 → 検討要素の提案
def triggers(reviews):
    seen, miss = set(), []
    for rv in reviews:
        if rv["summary"]["kind"] == "totoGOAL3":
            continue
        for r in rv["rows"]:
            key = (r.get("home"), r.get("away"), r.get("date"))
            if key in seen or r.get("actual") is None:
                continue
            seen.add(key)
            if r.get("feat"):
                miss.append(r["feat"])
    out = []
    if not miss:
        return out
    n = len(miss)
    draw = sum(m["act"] == 1 for m in miss)
    if draw / n >= 0.3:
        out.append(("closeness", f"外れ{n}試合中{draw}試合が引き分けの見逃し"))
        out.append(("goal_env", "引き分けの見逃しが多いため、得点が少なくなりそうな試合の扱いも検証"))
    tired = sum(1 for m in miss if m["rest_h"] is not None and m["rest_a"] is not None and m["pick"] != 1 and
                (m["rest_h"] if m["pick"] == 0 else m["rest_a"]) <= 4 and abs(m["rest_h"] - m["rest_a"]) >= 3)
    if tired:
        out.append(("rest_diff", f"本命が連戦で休養日が短かった外れが{tired}試合"))
    form = sum(1 for m in miss if m["act"] != 1 and m["form_h"] is not None and m["form_a"] is not None and
               ((m["form_h"] - m["form_a"]) if m["act"] == 0 else (m["form_a"] - m["form_h"])) >= 4)
    if form:
        out.append(("form_diff", f"直近の勢いで上回るチームが勝った外れが{form}試合"))
    vr = sum(m["vote_right"] for m in miss)
    if vr >= 2:
        out.append(("vote", f"世間の投票の多数派が当てていた外れが{vr}試合"))
    sp = sum(m["model_split"] for m in miss)
    if sp >= 2:
        out.append(("model_gap", f"2つのモデルの見立てが割れていた外れが{sp}試合"))
    return out
