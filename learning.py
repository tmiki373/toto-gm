# -*- coding: utf-8 -*-
"""
予想精度を上げる仕組み（回を重ねるごとに更新）
  1) 振り返り：終わった回の外れ方から「検討要素」の候補を追加（review.triggers）
     → 傾向が見つからない回も、未検証の候補を1つずつ順番に追加して検証を進める
  2) 検証   ：候補を加えた補正モデルを、過去データの時系列CV（過去で学習→未来で答え合わせ）で評価
  3) 採否   ：対数損失が改善したものだけ採用。効かなくなった要素は外す
  4) 投票率 ：振り返り済みの回が30試合分たまったら、世間の投票率を混ぜる比率を学習
  すべての判断は「学習の記録」として残り、サイトで確認できる
"""
import numpy as np

import engine as E
import review as R

ORDER = ["temperature", "draw_bias", "closeness", "goal_env", "rest_diff", "form_diff", "model_gap", "vote"]
ADOPT_GAIN = 0.001       # 採用に必要な改善幅（対数損失）
VOTE_MIN = 30            # 投票率ブレンドの学習に必要な試合数


def default_state():
    return {"active": [], "pool": ["temperature", "draw_bias"], "scores": {}, "adopted_at": {},
            "history": [], "track": [], "reviewed": [], "w": 0.5, "vote_b": 0.0, "vote_n": 0, "stacker": None}


def _log(L, event, detail, rnd=None):
    L["history"].append({"at": E.now_jst().strftime("%Y/%m/%d %H:%M"), "round": rnd, "event": event, "detail": detail})
    L["history"] = L["history"][-80:]


def select(L, P, F, y, rnd, log):
    """候補を1つずつ足して検証（前進選択）→ 不要になったものを外す（後退）"""
    active = [a for a in L["active"] if a in E.STACK_FEATS]
    base, _, _ = E.cv_score(active, P, F, y)
    log(f"検証: 現在の採用要素 {active or 'なし'} 対数損失 {base:.4f}")
    for _ in range(3):
        best, best_s = None, base
        for c in L["pool"]:
            if c in active or c not in E.STACK_FEATS:
                continue
            s, _, _ = E.cv_score(active + [c], P, F, y)
            L["scores"][c] = round(base - s, 5)
            log(f"  候補 {E.FEAT_INFO[c][0]}: {s:.4f}（{base - s:+.4f}）")
            if s < best_s - ADOPT_GAIN:
                best, best_s = c, s
        if best is None:
            break
        active.append(best)
        L["adopted_at"][best] = rnd
        _log(L, "採用", f"「{E.FEAT_INFO[best][0]}」を採用（対数損失 {base:.4f} → {best_s:.4f}）", rnd)
        base = best_s
    for a in list(active):
        rest = [x for x in active if x != a]
        s, _, _ = E.cv_score(rest, P, F, y)
        if s <= base - 0.0003:
            active = rest
            L["adopted_at"].pop(a, None)
            _log(L, "除外", f"「{E.FEAT_INFO[a][0]}」は効果がなくなったため除外（{base:.4f} → {s:.4f}）", rnd)
            base = s
    rejected = [c for c in L["pool"] if c not in active and c in E.STACK_FEATS and c in L["scores"]]
    for c in rejected:
        if not any(h["event"] == "見送り" and c in h["detail"] and h["round"] == rnd for h in L["history"]):
            _log(L, "見送り", f"「{E.FEAT_INFO[c][0]}」は改善せず見送り（{L['scores'][c]:+.4f}）", rnd)
    L["active"] = active + [a for a in L["active"] if a == "vote"]


def fit_vote(L, state, log, allow=True):
    """振り返り済みの回で『モデル × 投票率^b』の b を学習"""
    rows, seen = [], set()
    for e in state.get("archive", {}).values():
        if e["kind"] == "totoGOAL3":
            continue
        for r in e.get("_review_rows", []):
            k = (r["home"], r["away"], r["date"])
            if k in seen or r.get("actual") is None or not r.get("v"):
                continue
            seen.add(k)
            rows.append((np.asarray(r["p_model"]), np.asarray(r["v"]), r["actual"]))
    L["vote_n"] = len(rows)
    if len(rows) < VOTE_MIN or not allow:
        return
    def ll(b):  # noqa
        s = 0
        for p, v, a in rows:
            q = p * np.clip(v, 1e-3, 1) ** b
            s -= np.log(max(q[a] / q.sum(), 1e-9))
        return s / len(rows)
    grid = np.round(np.arange(0, 1.01, 0.1), 1)
    scores = {b: ll(b) for b in grid}
    b = min(scores, key=scores.get)
    old = L.get("vote_b", 0.0)
    if scores[b] < scores[0.0] - 0.002:
        if "vote" not in L["active"]:
            L["active"].append("vote")
            _log(L, "採用", f"「投票率」を比率{b}で採用（{len(rows)}試合で検証、対数損失 {scores[0.0]:.4f} → {scores[b]:.4f}）")
        elif b != old:
            _log(L, "調整", f"投票率の比率を {old} → {b} に調整")
        L["vote_b"] = float(b)
    else:
        if "vote" in L["active"]:
            L["active"].remove("vote")
            _log(L, "除外", "「投票率」は効果がなくなったため除外")
        L["vote_b"] = 0.0
    L["scores"]["vote"] = round(scores[0.0] - scores[b], 5)


def update(state, M, new_reviews, log=print):
    L = state.setdefault("learning", default_state())
    for k, v in default_state().items():
        L.setdefault(k, v)
    L["w"] = float(M["w"])
    oos = M["oos"]
    P, F, y = oos["P_base"], oos["F"], oos["y"]
    rnd = max((rv["summary"]["round"] for rv in new_reviews), default=None)

    need = not L["history"]
    if not L["history"]:
        _log(L, "開始", "学習を開始（初期の検討要素: 確率の強弱補正・引き分け補正）")
    if new_reviews:
        need = True
        for rv in new_reviews:
            s = rv["summary"]
            L["track"].append({k: s[k] for k in ["round", "kind", "n", "known", "hits", "ticket", "ll", "ll_base",
                                                 "reconstructed"]})
            L["reviewed"].append(f"{s['round']}|{s['kind']}")
        L["track"] = L["track"][-120:]
        L["reviewed"] = L["reviewed"][-400:]
        added = []
        for feat, why in R.triggers(new_reviews):
            if feat not in L["pool"]:
                L["pool"].append(feat); added.append(feat)
                _log(L, "候補追加", f"「{E.FEAT_INFO[feat][0]}」を検討要素に追加（理由: {why}）", rnd)
        if not added:
            nxt = next((f for f in ORDER if f not in L["pool"]), None)
            if nxt:
                L["pool"].append(nxt)
                _log(L, "候補追加", f"「{E.FEAT_INFO[nxt][0]}」を検討要素に追加（理由: 外れ方に目立つ傾向がなかったため、"
                                    "未検証の要素を順番に検証）", rnd)
    if need:
        select(L, P, F, y, rnd, log)
    fit_vote(L, state, log, allow="vote" in L["pool"])

    active = [a for a in L["active"] if a in E.STACK_FEATS]
    st = E.Stacker(active).fit(P, F, y)
    L["stacker"] = st.to_dict()

    # 精度表（同じ検証区間で比較）
    ll_cv, Q, te = E.cv_score(active, P, F, y)
    yt = y[te]
    rows = [("基準（平均）", np.tile(oos["base"], (len(te), 1))), ("ポアソン", oos["P_poi"][te]),
            ("Elo", oos["P_elo"][te]), ("アンサンブル", P[te]), ("学習補正後", Q)]
    M["backtest"] = [{"name": n, **E.metrics(p, yt)} for n, p in rows]
    conf = Q.max(1)
    bands = [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]
    M["calib"] = [{"band": f"{a:.0%}〜{min(b, 1):.0%}", "n": int(m.sum()), "pred": float(conf[m].mean()),
                   "act": float((Q.argmax(1)[m] == yt[m]).mean())}
                  for a, b in bands for m in [(conf >= a) & (conf < b)] if m.sum()]
    M["stacker"] = st
    M["vote_b"] = L["vote_b"]
    return L
