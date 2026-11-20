# -*- coding: utf-8 -*-
"""
予想精度を上げる仕組み（回を重ねるごとに更新）   v4
  1) モデル設定の自動調整：結果が30試合ふえるごとに、半減期・縮小の強さ・ルヴァン杯の重み・
     Elo の更新幅/ホーム有利/シーズン間の平均回帰 を候補から選び直す（engine.train）
  2) 振り返り：外れ方から「検討要素」の候補を追加 → 時系列CVで検証 → 改善したものだけ採用
  3) 投票率：過去回の『試合前の予想・投票率・結果』から、混ぜる比率を学習
"""
import numpy as np

import engine as E
import review as R

ORDER = ["temperature", "draw_bias", "closeness", "goal_env", "rest_diff", "form_diff", "model_gap"]
ADOPT_GAIN = 0.001
VOTE_MIN = 40
VOTE_GRID = np.round(np.arange(0, 2.01, 0.1), 1)


def default_state():
    return {"active": [], "pool": ["temperature", "draw_bias"], "scores": {}, "adopted_at": {},
            "history": [], "track": [], "reviewed": [], "w": 0.5, "vote_b": 0.0, "vote_n": 0,
            "stacker": None, "hp": None}


def _log(L, event, detail, rnd=None):
    L["history"].append({"at": E.now_jst().strftime("%Y/%m/%d %H:%M"), "round": rnd, "event": event, "detail": detail})
    L["history"] = L["history"][-80:]


def _hp_text(hp):
    p, e = hp["poi"], hp["elo"]
    return (f"直近重視の半減期{p['half_life']}日・個別差の縮小{p['l2']}・ルヴァン杯の重み{p['cup_w']}・"
            f"カテゴリ別の基準{'あり' if p.get('tier') else 'なし'} ／ "
            f"Elo更新幅{e['k']}・ホーム有利{e['hfa']}・シーズン間の平均回帰{e['regress']}・ルヴァン杯{e['cup_w']}")


def log_hp(L, M):
    if not M.get("retuned"):
        return
    old, new, rep = L.get("hp"), M["hp"], M.get("hp_report", {})
    gain = []
    for k, name in (("poi", "ポアソン"), ("elo", "Elo")):
        r = rep.get(k) or {}
        if r.get("default") is not None:
            gain.append(f"{name} {r['default']:.4f}→{r['best']:.4f}")
    msg = "モデル設定を再調整: " + _hp_text(new) + (f"（初期設定比の対数損失 {'、'.join(gain)}）" if gain else "")
    if not old or old.get("elo") != new["elo"] or old.get("poi") != new["poi"]:
        _log(L, "調整", msg)


def select(L, P, F, y, rnd, log):
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
    for c in [c for c in L["pool"] if c not in active and c in E.STACK_FEATS and c in L["scores"]]:
        if not any(h["event"] == "見送り" and c in h["detail"] and h["round"] == rnd for h in L["history"]):
            _log(L, "見送り", f"「{E.FEAT_INFO[c][0]}」は改善せず見送り（{L['scores'][c]:+.4f}）", rnd)
    L["active"] = active + [a for a in L["active"] if a == "vote"]


def _vote_rows(state):
    rows, seen = [], set()
    for rs in state.get("vote_hist", {}).values():
        for r in rs:
            k = (r["home"], r["away"], r["date"][:10])
            if k not in seen:
                seen.add(k)
                rows.append((np.asarray(r["p"]), np.asarray(r["v"]), int(r["a"])))
    for e in state.get("archive", {}).values():
        if e["kind"] == "totoGOAL3" or e.get("reconstructed"):
            continue
        for r in e.get("_review_rows", []):
            k = (r["home"], r["away"], str(r["date"])[:10])
            if k in seen or r.get("actual") is None or not r.get("v"):
                continue
            seen.add(k)
            rows.append((np.asarray(r["p_model"]), np.asarray(r["v"]), int(r["actual"])))
    return rows


def fit_vote(L, state, log):
    """予想 × 投票率^b の b を、過去回のデータで選ぶ（b=0 は投票率を使わない）"""
    rows = _vote_rows(state)
    L["vote_n"] = len(rows)
    if len(rows) < VOTE_MIN:
        log(f"投票率の学習: データ {len(rows)}試合（{VOTE_MIN}試合以上で開始）")
        return
    if "vote" not in L["pool"]:
        L["pool"].append("vote")
    P = np.array([p for p, _, _ in rows]); V = np.clip(np.array([v for _, v, _ in rows]), 1e-3, 1)
    y = np.array([a for _, _, a in rows])

    def ll(b, idx=slice(None)):
        Q = P[idx] * V[idx] ** b
        Q = Q / Q.sum(1, keepdims=True)
        yy = y[idx]
        return -np.mean(np.log(np.clip(Q[np.arange(len(yy)), yy], 1e-9, 1)))

    # 過学習を避けるため、前半のデータで選んだ b が後半でも効くかを確認してから採用
    half = len(rows) // 2
    first, second = slice(0, half), slice(half, None)
    sc = {b: ll(b) for b in VOTE_GRID}
    b = min(sc, key=sc.get)
    b_first = min(VOTE_GRID, key=lambda x: ll(x, first))
    hold0, hold_b = ll(0.0, second), ll(b_first, second)
    vote_only = -np.mean(np.log(np.clip(V[np.arange(len(y)), y] / V.sum(1), 1e-9, 1)))
    log(f"投票率の学習: {len(rows)}試合 / 予想のみ {sc[0.0]:.4f}・投票率のみ {vote_only:.4f}・最適 b={b} {sc[b]:.4f}"
        f"（後半での確認 {hold0:.4f}→{hold_b:.4f}）")
    old = L.get("vote_b", 0.0)
    L["scores"]["vote"] = round(sc[0.0] - sc[b], 5)
    if b > 0 and sc[b] < sc[0.0] - 0.002 and hold_b < hold0:
        if "vote" not in L["active"]:
            L["active"].append("vote")
            _log(L, "採用", f"「投票率」を比率{b}で採用（{len(rows)}試合で検証、対数損失 {sc[0.0]:.4f} → {sc[b]:.4f}）")
        elif abs(b - old) >= 0.05:
            _log(L, "調整", f"投票率の比率を {old} → {b} に調整（{len(rows)}試合）")
        L["vote_b"] = float(b)
    else:
        if "vote" in L["active"]:
            L["active"].remove("vote")
            _log(L, "除外", "「投票率」は効果が確認できなくなったため除外")
        L["vote_b"] = 0.0


def update(state, M, new_reviews, log=print):
    L = state.setdefault("learning", default_state())
    for k, v in default_state().items():
        L.setdefault(k, v)
    if not L["history"]:
        _log(L, "開始", "学習を開始（初期の検討要素: 確率の強弱補正・引き分け補正）")
    log_hp(L, M)
    L["hp"] = M["hp"]
    L["w"] = float(M["w"])
    oos = M["oos"]
    P, F, y = oos["P_base"], oos["F"], oos["y"]
    rnd = max((rv["summary"]["round"] for rv in new_reviews), default=None)

    if new_reviews:
        for rv in new_reviews:
            s = rv["summary"]
            L["track"].append({k: s[k] for k in ["round", "kind", "n", "known", "hits", "ticket", "ll", "ll_base",
                                                 "reconstructed"]})
            L["reviewed"].append(f"{s['round']}|{s['kind']}")
        L["track"] = L["track"][-120:]
        L["reviewed"] = L["reviewed"][-400:]
        added = []
        for feat, why in R.triggers(new_reviews):
            if feat not in L["pool"] and feat != "vote":
                L["pool"].append(feat); added.append(feat)
                _log(L, "候補追加", f"「{E.FEAT_INFO[feat][0]}」を検討要素に追加（理由: {why}）", rnd)
        if not added:
            nxt = next((f for f in ORDER if f not in L["pool"]), None)
            if nxt:
                L["pool"].append(nxt)
                _log(L, "候補追加", f"「{E.FEAT_INFO[nxt][0]}」を検討要素に追加（理由: 外れ方に目立つ傾向がなかったため、"
                                    "未検証の要素を順番に検証）", rnd)
    # 設定の再調整・新しい振り返りがあったときは検討要素を検証し直す
    if new_reviews or M.get("retuned") or not L["scores"]:
        select(L, P, F, y, rnd, log)
    fit_vote(L, state, log)

    active = [a for a in L["active"] if a in E.STACK_FEATS]
    st = E.Stacker(active).fit(P, F, y)
    L["stacker"] = st.to_dict()

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
