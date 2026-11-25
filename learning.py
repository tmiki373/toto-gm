# -*- coding: utf-8 -*-
"""
外れから学ぶ仕組み   v7
  1) 外れの記録 : 終わった回の全試合（試合前の予想・特徴・結果・原因タグ）を貯める
  2) 診断       : 過去全試合の答え合わせ ＋ 外れの記録 で『どの種類の試合で、どちら向きに外しているか』を統計的に調べる
  3) 補正の検証 : 時間順の答え合わせで、対数損失とRPSが両方改善したものだけ採用（効かなくなったら外す）
  4) 反映       : 採用した補正を次回の予想に使い、試合ごとに『何を何pt動かしたか』を記録
  5) 市場（v7） : WINNERのオッズを全Ｊリーグ試合で記録し、『予想・市場・結果』がたまったら
                  予想と市場の重み付き平均の重みを学習。古い試合で決めた重みが新しい試合でも効く場合だけ採用
  6) 出場停止（v7）: 公式一覧を蓄積し、補正候補『出場停止』として同じ検証にかける
  7) 投票率     : 市場を使えない試合のために、予想と toto 投票率の重み付き平均も学習
"""
import numpy as np

import engine as E
import review as R

ADOPT_LL = 0.001
LEDGER_W_REAL = 2.0
LEDGER_W_RECON = 1.0
VOTE_MIN = 40
VOTE_B = np.round(np.arange(0, 1.01, 0.1), 1)
VOTE_C = np.round(np.arange(-0.2, 0.61, 0.1), 1)
MARKET_MIN = 80           # 市場の重みを学習し始める試合数（WINNERは週に最大50試合ほど）
MARKET_A = np.round(np.arange(0, 1.01, 0.1), 1)
LEARN_VER = 7
Z_SIG = 2.0
MIN_SEG = 40


def default_state():
    return {"active": [], "pool": list(E.STACK_FEATS), "scores": {}, "adopted_at": {},
            "history": [], "track": [], "reviewed": [], "w": 0.5, "vote_b": 0.0, "vote_c": 0.0, "vote_n": 0,
            "market_a": 0.0, "market_ag": 0.0, "market_n": 0, "market_eval": None,
            "stacker": None, "hp": None, "ver": LEARN_VER, "findings": [], "lessons": []}


def _log(L, event, detail, rnd=None):
    L["history"].append({"at": E.now_jst().strftime("%Y/%m/%d %H:%M"), "round": rnd, "event": event, "detail": detail})
    L["history"] = L["history"][-100:]
    return detail


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
    if not old or old.get("elo") != new["elo"] or old.get("poi") != new["poi"]:
        _log(L, "調整", "モデル設定を再調整: " + _hp_text(new) + (f"（初期設定比の対数損失 {'、'.join(gain)}）" if gain else ""))


# ---------------------------------------------------------------- 学習データ
def training_set(M, state):
    oos = M["oos"]
    P, F, y = oos["P_base"], dict(oos["F"]), oos["y"]
    n0 = len(y)
    # v7: 蓄積した出場停止を過去全試合の特徴にも反映（記録を始めた日以降の試合が対象）
    smap = E.susp_map(state.get("susp_hist", []))
    if smap:
        F["susp_h"] = np.array([E.susp_count(smap, h, d) for h, d in zip(oos["home"], oos["dates"])], float)
        F["susp_a"] = np.array([E.susp_count(smap, a, d) for a, d in zip(oos["away"], oos["dates"])], float)
    w = np.ones(n0)
    src = np.zeros(n0, int)
    led = sorted(state.get("ledger", []), key=lambda x: x["date"])
    led = [x for x in led if x.get("p_base") and x.get("F")]
    F = {k: np.asarray(F.get(k, np.full(n0, E.F_DEFAULT[k])), float) for k in E.F_KEYS}
    if led:
        P2 = np.array([x["p_base"] for x in led], float)
        F2 = {k: np.array([(x["F"].get(k) if x["F"].get(k) is not None else E.F_DEFAULT[k]) for x in led], float)
              for k in E.F_KEYS}
        y2 = np.array([x["a"] for x in led], int)
        w2 = np.array([LEDGER_W_REAL if x.get("real") else LEDGER_W_RECON for x in led])
        s2 = np.array([1 if x.get("real") else 2 for x in led])
        P = np.vstack([P, P2])
        F = {k: np.concatenate([F[k], F2[k]]) for k in E.F_KEYS}
        y = np.concatenate([y, y2]); w = np.concatenate([w, w2]); src = np.concatenate([src, s2])
    return P, F, y, w, src


# ---------------------------------------------------------------- 診断
def _seg(name, mask, prob, hit, w, feats, desc_fn, min_n=MIN_SEG):
    n = float(w[mask].sum())
    if mask.sum() < min_n:
        return None
    pr = float(np.sum(w[mask] * prob[mask]) / n)
    ac = float(np.sum(w[mask] * hit[mask]) / n)
    se = np.sqrt(max(pr * (1 - pr), 1e-6) / mask.sum())
    z = (ac - pr) / se
    level = "確定" if abs(z) >= Z_SIG else "監視中" if abs(z) >= 1.0 else "問題なし"
    return {"key": name, "n": int(mask.sum()), "pred": round(pr, 3), "act": round(ac, 3), "z": round(float(z), 1),
            "level": level, "feats": feats, "text": desc_fn(pr, ac, int(mask.sum()))}


def diagnose(Q, F, y, w, src):
    X = E.feat_values(Q, F)
    tier = np.asarray(F["tier"])
    out = []
    D = (y == 1).astype(float)
    H = (y == 0).astype(float)
    fav = Q.argmax(1)
    favhit = (fav == y).astype(float)
    pfav = Q.max(1)

    def dtxt(label):
        return lambda p, a, n: f"{label}の引き分けを{'過小' if a > p else '過大'}評価（予想{p:.0%}・実際{a:.0%}、{n}試合）"

    def htxt(label):
        return lambda p, a, n: f"{label}のホーム勝ちを{'過小' if a > p else '過大'}評価（予想{p:.0%}・実際{a:.0%}、{n}試合）"

    def side(name, mask, fwd, feats, label):
        sp = np.where(fwd, Q[:, 0], Q[:, 2])
        sh = np.where(fwd, y == 0, y == 2).astype(float)
        return _seg(name, mask, sp, sh, w, feats,
                    lambda p, a, n: f"{label}の勝ちを{'過小' if a > p else '過大'}評価（予想{p:.0%}・実際{a:.0%}、{n}試合）")

    for k, lab, df, hf in ((0, "J1", "draw_bias", "temperature"), (1, "J2", "draw_j2", "home_j2"),
                           (2, "J3", "draw_j3", "home_j3")):
        m = tier == k
        out.append(_seg(f"draw_{lab}", m, Q[:, 1], D, w, [df, "closeness", "goal_env"], dtxt(lab)))
        out.append(_seg(f"home_{lab}", m, Q[:, 0], H, w, [hf], htxt(lab)))
    out.append(_seg("draw_close", X["closeness"] > 0.85, Q[:, 1], D, w, ["closeness", "draw_form"], dtxt("実力が拮抗した試合")))
    out.append(_seg("draw_low", X["goal_env"] > 0.4, Q[:, 1], D, w, ["goal_env"], dtxt("得点が少なそうな試合")))
    out.append(_seg("draw_dead", X["dead_rubber"] > 0, Q[:, 1], D, w, ["dead_rubber"], dtxt("消化試合")))
    for th, lab in ((0.6, "本命が60%以上の試合"), (0.0, "全試合")):
        m = pfav >= th if th else np.ones(len(y), bool)
        out.append(_seg(f"fav_{th}", m, pfav, favhit, w, ["temperature"],
                        lambda p, a, n, lab=lab: f"{lab}で本命が{'思ったより当たる' if a > p else '思ったより当たらない'}"
                                                 f"（予想{p:.0%}・実際{a:.0%}、{n}試合）"))
    sd = X["stake_diff"]
    out.append(side("stake", sd != 0, sd > 0, ["stake_diff"], "終盤に懸かっているものがある側"))
    rd = X["resid_diff"]
    out.append(side("resid", np.abs(rd) >= 0.3, rd > 0, ["resid_diff"], "最近『予想以上に勝点を取っている』側"))
    rs = X["rest_diff"] * 7
    out.append(side("rest", np.abs(rs) >= 3, rs < 0, ["rest_diff"], "休養日が3日以上短い側"))
    su = X["susp_diff"]
    out.append(_seg("susp", su != 0, np.where(su > 0, Q[:, 0], Q[:, 2]), np.where(su > 0, y == 0, y == 2).astype(float),
                    w, ["susp_diff"],
                    lambda p, a, n: f"出場停止が少ない側の勝ちを{'過小' if a > p else '過大'}評価（予想{p:.0%}・実際{a:.0%}、{n}試合）",
                    min_n=25))
    real = src == 1
    out.append(_seg("toto_draw", real, Q[:, 1], D, w, ["draw_bias", "closeness"], dtxt("toto対象試合"), min_n=25))
    out.append(_seg("toto_fav", real, pfav, favhit, w, ["temperature"],
                    lambda p, a, n: f"toto対象試合で本命が{'思ったより当たる' if a > p else '思ったより当たらない'}"
                                    f"（予想{p:.0%}・実際{a:.0%}、{n}試合）", min_n=25))
    return sorted([o for o in out if o], key=lambda o: -abs(o["z"]))


TAG_SEGS = {"draw_miss": ["draw_J1", "draw_J2", "draw_J3", "draw_close", "draw_low", "toto_draw"],
            "upset": ["fav_0.6", "toto_fav"], "close": ["fav_0.0", "toto_fav"],
            "stake_gap": ["stake"], "dead_rubber": ["draw_dead"], "form_under": ["resid"],
            "fav_tired": ["rest"], "fav_susp": ["susp"], "model_split": [], "goal_short": [],
            "vote_right": [], "market_right": [], "vote_drift": []}


def lesson_checks(tags, segs, L):
    by = {s["key"]: s for s in segs}
    out = []
    for tg, n in tags:
        keys = [k for k in TAG_SEGS.get(tg, []) if k in by]
        if tg == "market_right":
            ev = L.get("market_eval") or {}
            a = L.get("market_a", 0.0)
            chk = (f"記録した{ev.get('n', 0)}試合で、市場の対数損失 {ev['ll_market']:.3f} / 予想 {ev['ll_model']:.3f}"
                   if ev.get("n") else f"WINNERのオッズを記録中（{L.get('market_n', 0)}/{MARKET_MIN}試合）")
            out.append({"tag": R.TAG_INFO[tg], "n": n, "check": chk,
                        "verdict": f"市場を重み{a}で予想に反映中" if a else
                        ("記録がたまり次第、市場を混ぜる重みを検証して反映" if L.get("market_n", 0) < MARKET_MIN
                         else "現時点では市場を混ぜても改善が確認できず不採用（毎回再検証）")})
            continue
        if tg == "vote_right":
            vb = L.get("vote_b", 0.0)
            out.append({"tag": R.TAG_INFO[tg], "n": n, "check": f"過去{L.get('vote_n', 0)}試合で投票率を混ぜる効果を検証",
                        "verdict": f"投票率を重み{vb}で反映中（市場を使えない試合）" if vb else "現時点では投票率を混ぜても改善しないため不採用"})
            continue
        if tg == "vote_drift":
            out.append({"tag": R.TAG_INFO[tg], "n": n, "check": "販売期間中の投票の動きを記録中",
                        "verdict": "WINNERのオッズの動き（最初と直前の差）とあわせて記録継続"})
            continue
        if not keys:
            msg = {"goal_short": "得点の想定は毎回の再学習で更新（チーム別の攻撃力に反映）",
                   "model_split": "2つのモデルの混ぜ方と『モデル間の不一致』補正で対応（毎回検証）",
                   "fav_susp": f"出場停止の記録を蓄積中（{L.get('susp_n', 0)}件）。試合数がたまり次第、補正『出場停止』を検証"
                   }.get(tg, "累積データで検証中")
            out.append({"tag": R.TAG_INFO[tg], "n": n, "check": msg, "verdict": "継続監視"})
            continue
        worst = max((by[k] for k in keys), key=lambda x: abs(x["z"]))
        v = {"確定": "累積でも偏りを確認 → 対応する補正を優先して検証",
             "監視中": "累積で弱い偏り → 補正候補として検証・監視",
             "問題なし": "累積ではずれが小さい → 今回分は確率どおりの外れと判断（補正は変えず再学習のみ）"}[worst["level"]]
        out.append({"tag": R.TAG_INFO[tg], "n": n, "check": worst["text"], "verdict": v})
    return out


# ---------------------------------------------------------------- 補正の検証と採否
def select(L, P, F, y, w, rnd, log, focus=()):
    active = [a for a in L["active"] if a in E.STACK_FEATS]
    base_m, _, _ = E.cv_score(active, P, F, y, w)
    log(f"検証: 現在の採用 {active or 'なし'} 対数損失 {base_m['ll']:.4f} / RPS {base_m['rps']:.4f}")
    changes = []
    order = list(dict.fromkeys(list(focus) + L["pool"]))
    while True:
        best, best_m = None, base_m
        for c in order:
            if c in active or c not in E.STACK_FEATS:
                continue
            m, _, _ = E.cv_score(active + [c], P, F, y, w)
            L["scores"][c] = round(base_m["ll"] - m["ll"], 5)
            log(f"  候補 {E.FEAT_INFO[c][0]}: 対数損失 {m['ll']:.4f}（{base_m['ll'] - m['ll']:+.4f}）・"
                f"RPS {m['rps']:.4f}（{base_m['rps'] - m['rps']:+.4f}）")
            if m["ll"] < best_m["ll"] - ADOPT_LL and m["rps"] <= base_m["rps"]:
                best, best_m = c, m
        if best is None:
            break
        active.append(best)
        L["adopted_at"][best] = rnd
        changes.append(_log(L, "採用", f"「{E.FEAT_INFO[best][0]}」を採用（対数損失 {base_m['ll']:.4f}→{best_m['ll']:.4f}、"
                                      f"RPS {base_m['rps']:.4f}→{best_m['rps']:.4f}）", rnd))
        base_m = best_m
    for a in list(active):
        rest = [x for x in active if x != a]
        m, _, _ = E.cv_score(rest, P, F, y, w)
        if m["ll"] <= base_m["ll"] - 0.0003:
            active = rest
            L["adopted_at"].pop(a, None)
            changes.append(_log(L, "除外", f"「{E.FEAT_INFO[a][0]}」は効果がなくなったため除外"
                                          f"（{base_m['ll']:.4f}→{m['ll']:.4f}）", rnd))
            base_m = m
    L["active"] = active + [a for a in L["active"] if a in ("vote", "market")]
    return changes


def _vote_rows(state):
    rows, seen = [], set()
    for rs in state.get("vote_hist", {}).values():
        for r in rs:
            k = (r["home"], r["away"], r["date"][:10])
            if k not in seen:
                seen.add(k)
                rows.append((np.asarray(r["p"]), np.asarray(r["v"]), int(r["a"]), int(r.get("round", 0))))
    for x in state.get("ledger", []):
        k = (x["home"], x["away"], str(x["date"])[:10])
        if k in seen or not x.get("v") or not x.get("real"):
            continue
        seen.add(k)
        rows.append((np.asarray(x["p_base"]), np.asarray(x["v"]), int(x["a"]), int(x["round"])))
    rows.sort(key=lambda x: x[3])
    return rows


def fit_vote(L, state, log):
    rows = _vote_rows(state)
    L["vote_n"] = len(rows)
    if len(rows) < VOTE_MIN:
        log(f"投票率の学習: データ {len(rows)}試合（{VOTE_MIN}試合以上で開始）")
        return []
    P = np.array([r[0] for r in rows]); V = np.clip(np.array([r[1] for r in rows]), 1e-3, 1)
    y = np.array([r[2] for r in rows])
    lv = np.log(V)

    def ll(b, c, idx=slice(None)):
        Lg = (1 - b) * np.log(np.clip(P[idx], 1e-9, 1)) + b * lv[idx]
        Lg[:, 1] += c
        Lg -= Lg.max(1, keepdims=True)
        Q = np.exp(Lg); Q /= Q.sum(1, keepdims=True)
        yy = y[idx]
        return -np.mean(np.log(np.clip(Q[np.arange(len(yy)), yy], 1e-9, 1)))

    grid = [(b, c) for b in VOTE_B for c in VOTE_C]
    sc = {bc: ll(*bc) for bc in grid}
    b, c = min(grid, key=lambda bc: sc[bc] + 0.02 * bc[1] ** 2)
    chunks = np.array_split(np.arange(len(rows)), 4)
    hold0 = hold = 0.0
    nte = 0
    for j in range(1, 4):
        tr, te = np.concatenate(chunks[:j]), chunks[j]
        if len(tr) < 30:
            continue
        bj, cj = min(grid, key=lambda bc: ll(*bc, tr) + 0.02 * bc[1] ** 2)
        hold0 += ll(0.0, 0.0, te) * len(te)
        hold += ll(bj, cj, te) * len(te)
        nte += len(te)
    if not nte:
        return []
    hold0, hold = hold0 / nte, hold / nte
    base0 = sc[(0.0, 0.0)]
    log(f"投票率の学習: {len(rows)}試合 / 予想のみ {base0:.4f}・最適 b={b} c={c} {sc[(b, c)]:.4f}"
        f"（時系列での確認 {hold0:.4f}→{hold:.4f}）")
    L["scores"]["vote"] = round(base0 - sc[(b, c)], 5)
    ob, oc = L.get("vote_b", 0.0), L.get("vote_c", 0.0)
    active_now = "vote" in L["active"]
    ok_new = (b or c) and sc[(b, c)] < base0 - 0.002 and hold < hold0
    ok_keep = active_now and (b or c) and sc[(b, c)] < base0 and hold <= hold0 + 0.008
    ch = []
    if ok_new or ok_keep:
        if not active_now:
            L["active"].append("vote")
            ch.append(_log(L, "採用", f"「toto投票率」を重み{b}・引き分けの人気補正{c:+.1f}で採用"
                                     f"（{len(rows)}試合、対数損失 {base0:.4f}→{sc[(b, c)]:.4f}）"))
        elif abs(b - ob) >= 0.05 or abs(c - oc) >= 0.05:
            ch.append(_log(L, "調整", f"投票率の反映を b={ob}・c={oc} → b={b}・c={c} に調整（{len(rows)}試合）"))
        L["vote_b"], L["vote_c"] = float(b), float(c)
    else:
        if active_now:
            L["active"].remove("vote")
            ch.append(_log(L, "除外", "「toto投票率」は効果が確認できなくなったため除外"))
        L["vote_b"], L["vote_c"] = 0.0, 0.0
    return ch


# ---------------------------------------------------------------- v7: 市場（WINNERのオッズ）
def _market_rows(state, res):
    """記録した WINNER の試合のうち、結果が出たもの（試合日の古い順）"""
    if res is None or not len(res):
        return []
    rmap = {}
    for h, a, d, hg, ag in res[["home", "away", "date", "hg", "ag"]].itertuples(index=False):
        rmap[f"{d.date().isoformat()}|{h}|{a}"] = (int(hg), int(ag))
    rows = []
    for k, e in state.get("market_hist", {}).items():
        sc = None
        d0 = E.pd.Timestamp(e["date"]).date()
        for off in (0, -1, 1):
            sc = rmap.get(f"{(d0 + E.timedelta(days=off)).isoformat()}|{e['hs']}|{e['as']}")
            if sc:
                break
        if not sc or not e.get("p"):
            continue
        hg, ag = sc
        rows.append({"date": e["date"], "p": e["p"], "m": e["m"], "m_first": e.get("m_first") or e["m"],
                     "a": 0 if hg > ag else 1 if hg == ag else 2,
                     "g_h": e["g_h"], "g_a": e["g_a"], "mg_h": e["mg_h"], "mg_a": e["mg_a"],
                     "bh": min(hg, 3), "ba": min(ag, 3)})
    rows.sort(key=lambda r: r["date"])
    return rows


def _pool_ll(P, M, y, a):
    Q = E.log_pool(P, M, a)
    return -np.mean(np.log(np.clip(Q[np.arange(len(y)), y], 1e-9, 1))), Q


def _fit_weight(P, M, y, name, L, key, log, min_n):
    """重み a を全体で選び、時系列の確認（古い区間で決めた a を新しい区間で評価）で採否を決める"""
    sc = {a: _pool_ll(P, M, y, a)[0] for a in MARKET_A}
    a = min(sc, key=sc.get)
    chunks = np.array_split(np.arange(len(y)), 4)
    h0 = h1 = 0.0
    nte = 0
    for j in range(1, 4):
        tr, te = np.concatenate(chunks[:j]), chunks[j]
        if len(tr) < min_n // 2:
            continue
        aj = min(MARKET_A, key=lambda x: _pool_ll(P[tr], M[tr], y[tr], x)[0])
        h0 += _pool_ll(P[te], M[te], y[te], 0.0)[0] * len(te)
        h1 += _pool_ll(P[te], M[te], y[te], aj)[0] * len(te)
        nte += len(te)
    if not nte:
        return 0.0, sc, None
    h0, h1 = h0 / nte, h1 / nte
    log(f"市場の学習（{name}）: {len(y)}件 / 予想のみ {sc[0.0]:.4f}・市場のみ {sc[1.0]:.4f}・最適 a={a} {sc[a]:.4f}"
        f"（時系列での確認 {h0:.4f}→{h1:.4f}）")
    old = L.get(key, 0.0)
    ok_new = a > 0 and sc[a] < sc[0.0] - 0.002 and h1 < h0
    ok_keep = old > 0 and a > 0 and sc[a] < sc[0.0] and h1 <= h0 + 0.008
    return (float(a) if (ok_new or ok_keep) else 0.0), sc, (h0, h1)


def fit_market(L, state, res, log):
    rows = _market_rows(state, res)
    L["market_n"] = len(rows)
    L["market_rec"] = len(state.get("market_hist", {}))
    ch = []
    if len(rows) >= 20:
        P = np.array([r["p"] for r in rows]); M = np.array([r["m"] for r in rows])
        M0 = np.array([r["m_first"] for r in rows]); y = np.array([r["a"] for r in rows])
        mp, mm, m0 = E.metrics(P, y), E.metrics(M, y), E.metrics(M0, y)
        L["market_eval"] = {"n": len(rows), "ll_model": mp["ll"], "ll_market": mm["ll"], "ll_first": m0["ll"],
                            "rps_model": mp["rps"], "rps_market": mm["rps"],
                            "acc_model": mp["acc"], "acc_market": mm["acc"]}
        log(f"市場の比較: {len(rows)}試合 / 予想 対数損失 {mp['ll']:.4f}・的中{mp['acc']:.1%} ／ "
            f"市場(直前) {mm['ll']:.4f}・{mm['acc']:.1%} ／ 市場(販売開始直後) {m0['ll']:.4f}")
    if len(rows) < MARKET_MIN:
        log(f"市場の学習: 記録 {L['market_rec']}試合・結果あり {len(rows)}試合（{MARKET_MIN}試合で重みの学習を開始）")
        if L.get("market_a") or L.get("market_ag"):
            L["market_a"] = L["market_ag"] = 0.0
        return ch
    if "market" not in L["pool"]:
        L["pool"].append("market")
    old_a, old_ag = L.get("market_a", 0.0), L.get("market_ag", 0.0)
    a, sc, _ = _fit_weight(P, M, y, "1/0/2", L, "market_a", log, MARKET_MIN)
    # RPS も悪化しないことを確認
    if a > 0 and E.metrics(E.log_pool(P, M, a), y)["rps"] > E.metrics(P, y)["rps"]:
        a = 0.0
    G = np.array([r["g_h"] for r in rows] + [r["g_a"] for r in rows])
    MG = np.array([r["mg_h"] for r in rows] + [r["mg_a"] for r in rows])
    yg = np.array([r["bh"] for r in rows] + [r["ba"] for r in rows])
    order = np.argsort(np.array([r["date"] for r in rows] * 2), kind="stable")
    ag, _, _ = _fit_weight(G[order], MG[order], yg[order], "得点", L, "market_ag", log, MARKET_MIN * 2)
    L["scores"]["market"] = round(sc[0.0] - sc[a], 5) if a else round(sc[0.0] - min(sc.values()), 5)
    if a and not old_a:
        ch.append(_log(L, "採用", f"「WINNERのオッズ」を重み{a}で採用（{len(rows)}試合、対数損失 {sc[0.0]:.4f}→{sc[a]:.4f}）"))
    elif a and abs(a - old_a) >= 0.05:
        ch.append(_log(L, "調整", f"WINNERのオッズの重みを {old_a} → {a} に調整（{len(rows)}試合）"))
    elif not a and old_a:
        ch.append(_log(L, "除外", "「WINNERのオッズ」は効果が確認できなくなったため除外"))
    if ag != old_ag:
        ch.append(_log(L, "調整", f"totoGOAL3（得点予想）への市場の重みを {old_ag} → {ag} に{'設定' if ag else '解除'}"))
    L["market_a"], L["market_ag"] = a, ag
    if a and "market" not in L["active"]:
        L["active"].append("market")
    if not a and "market" in L["active"]:
        L["active"].remove("market")
    return ch


# ---------------------------------------------------------------- まとめ
def update(state, M, new_reviews, log=print, res=None):
    fresh = "learning" not in state
    L = state.setdefault("learning", default_state())
    upgraded = not fresh and L.get("ver", 0) < LEARN_VER
    for k, v in default_state().items():
        L.setdefault(k, v)
    if not L["history"]:
        _log(L, "開始", "学習を開始")
    if upgraded:
        L["pool"] = list(dict.fromkeys(L["pool"] + E.STACK_FEATS))
        L["active"] = [a for a in L["active"] if a in E.STACK_FEATS or a in ("vote", "market")]
        L["ver"] = LEARN_VER
        _log(L, "調整", "直前情報（v7）を導入: WINNERのオッズ（試合開始10分前まで動く市場の予想）と"
                        "公式の出場停止情報の記録を開始。たまり次第、効果を検証して反映")
    L["susp_n"] = len(state.get("susp_hist", []))
    log_hp(L, M)
    L["hp"] = M["hp"]
    L["w"] = float(M["w"])
    old_params = E.Stacker.from_dict(L.get("stacker")).params()

    P, F, y, w, src = training_set(M, state)
    n_real = int((src == 1).sum())
    n_susp = int(((F["susp_h"] + F["susp_a"]) > 0).sum())
    log(f"学習データ: 過去全試合 {int((src == 0).sum())} ＋ toto対象試合 {n_real}（重み{LEDGER_W_REAL}）"
        f" ＋ 再現 {int((src == 2).sum())} ／ 出場停止ありの試合 {n_susp}")
    rnd = max((rv["summary"]["round"] for rv in new_reviews), default=None)

    active = [a for a in L["active"] if a in E.STACK_FEATS]
    _, Q0, te0 = E.cv_score(active, P, F, y, w)
    find0 = diagnose(Q0, E.subF(F, te0), y[te0], w[te0], src[te0])

    focus, changes = [], []
    for f in find0:
        if f["level"] != "問題なし":
            focus += f["feats"]
    if new_reviews:
        for rv in new_reviews:
            s = rv["summary"]
            L["track"].append({k: s.get(k) for k in ["round", "kind", "n", "known", "hits", "ticket", "ll", "ll_base",
                                                     "reconstructed", "draws", "exp_draws", "fav_hits", "fav_n",
                                                     "mk_hits", "mk_n"]})
            L["reviewed"].append(f"{s['round']}|{s['kind']}")
        L["track"] = L["track"][-120:]
        L["reviewed"] = L["reviewed"][-400:]
        for feat, why in R.triggers(new_reviews):
            focus.insert(0, feat)
            _log(L, "重点検証", f"「{E.FEAT_INFO[feat][0]}」（理由: {why}）", rnd)

    if new_reviews or M.get("retuned") or upgraded or not L["scores"]:
        changes += select(L, P, F, y, w, rnd, log, focus=[f for f in focus if f in E.STACK_FEATS])
    changes += fit_vote(L, state, log)
    changes += fit_market(L, state, res, log)

    active = [a for a in L["active"] if a in E.STACK_FEATS]
    st = E.Stacker(active).fit(P, F, y, w)
    L["stacker"] = st.to_dict()
    if new_reviews:
        for k, v in st.params().items():
            o = old_params.get(k)
            if o is not None and abs(v - o) >= 0.1:
                changes.append(_log(L, "調整", f"「{E.FEAT_INFO[k][0]}」の強さを {o:+.2f} → {v:+.2f} に更新", rnd))

    m_cv, Q, te = E.cv_score(active, P, F, y, w)
    yt, wt = y[te], w[te]
    Fte = E.subF(F, te)
    findings = diagnose(Q, Fte, yt, wt, src[te])
    keep = ("key", "n", "pred", "act", "z", "level", "text")
    L["findings"] = [{k: f[k] for k in keep} for f in find0 if f["level"] != "問題なし"][:8]
    L["findings_after"] = [{k: f[k] for k in keep} for f in findings if f["level"] != "問題なし"][:8]
    oos_n = len(M["oos"]["y"])
    rows = [("基準（平均）", np.tile(M["oos"]["base"], (len(te), 1))),
            ("ポアソン", np.vstack([M["oos"]["P_poi"], P[oos_n:]])[te]),
            ("Elo", np.vstack([M["oos"]["P_elo"], P[oos_n:]])[te]),
            ("アンサンブル", P[te]), ("補正後", Q)]
    M["backtest"] = [{"name": n, **E.metrics(p, yt, wt)} for n, p in rows]
    tier = np.asarray(Fte["tier"]).astype(int)
    M["by_tier"] = {f"J{k + 1}": E.metrics(Q[tier == k], yt[tier == k], wt[tier == k])
                    for k in range(3) if (tier == k).sum() >= 30}
    real = src[te] == 1
    if real.sum() >= 13:
        M["by_tier"]["toto対象"] = E.metrics(Q[real], yt[real], wt[real])
    for k, m in M["by_tier"].items():
        log(f"  {k}: 引き分け 予想{m['draw_pred']:.1%}/実際{m['draw_act']:.1%}・本命的中{m['acc']:.1%}・RPS {m['rps']:.4f}")
    conf = Q.max(1)
    bands = [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]
    M["calib"] = [{"band": f"{a:.0%}〜{min(b, 1):.0%}", "n": int(m.sum()), "pred": float(conf[m].mean()),
                   "act": float((Q.argmax(1)[m] == yt[m]).mean())}
                  for a, b in bands for m in [(conf >= a) & (conf < b)] if m.sum()]
    M["stacker"] = st
    M["vote_b"], M["vote_c"] = L["vote_b"], L["vote_c"]
    M["market_a"], M["market_ag"] = L["market_a"], L["market_ag"]

    main = [rv for rv in new_reviews if rv["summary"]["kind"] == "toto"] or \
           [rv for rv in new_reviews if rv["summary"]["kind"] != "totoGOAL3"][:1]
    for rv in main:
        s = rv["summary"]
        tags = sorted(s.get("tags", {}).items(), key=lambda x: -x[1])
        L["lessons"] = [x for x in L["lessons"] if not (x["round"] == s["round"] and x["kind"] == s["kind"])]
        L["lessons"].append({
            "round": s["round"], "kind": s["kind"], "at": E.now_jst().strftime("%Y/%m/%d %H:%M"),
            "hits": s["hits"], "known": s["known"], "draws": s.get("draws"), "exp_draws": s.get("exp_draws"),
            "mk_hits": s.get("mk_hits"), "mk_n": s.get("mk_n"),
            "tags": [{"tag": k, "name": R.TAG_INFO[k], "n": v} for k, v in tags],
            "checks": lesson_checks(tags, find0, L),
            "changes": changes or ["補正の採否に変更なし（今回の結果を加えて補正の強さを再学習済み）"],
            "effects": []})
    L["lessons"] = L["lessons"][-20:]
    return L


def attach_effects(L, analysis):
    if not L.get("lessons"):
        return
    eff = []
    for kind in ("toto", "mini toto-A組", "mini toto-B組"):
        for r in analysis.get(kind, []):
            for f, d in r.get("reasons", []):
                if f in ("temperature",):
                    continue
                k = int(np.argmax(np.abs(d)))
                if abs(d[k]) >= 0.02:
                    eff.append({"kind": kind, "no": r["no"], "match": f"{r['home']}-{r['away']}",
                                "feat": E.FEAT_INFO[f][0], "out": E.OUT3[k], "pt": round(d[k] * 100, 1)})
        if eff:
            break
    eff.sort(key=lambda x: -abs(x["pt"]))
    L["lessons"][-1]["effects"] = eff[:8]
