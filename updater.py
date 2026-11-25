# -*- coding: utf-8 -*-
"""
取得 → 直前情報の記録（WINNERのオッズ・出場停止）→ 振り返り → 学習（診断・補正の検証・市場の重み）
→ 予想（補正と市場を反映）→ 保存   v7
"""
import pickle
import traceback
from pathlib import Path

import engine as E
import learning as LRN
import review as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SNAP = DATA / "snapshot.pkl"
SNAP_VER = 5


def load_snapshot():
    if SNAP.exists():
        try:
            with open(SNAP, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def refresh(force_results=False, password=None, log=print):
    prev = load_snapshot() or {}
    if prev.get("ver") != SNAP_VER:
        prev = {}
    logs, errors = [], []

    def lg(m):
        logs.append(f"{E.now_jst():%H:%M:%S} {m}")
        log(m)

    # 1) 対象試合
    try:
        card = E.fetch_card(log=lg)
        lg("対象試合: " + ", ".join(f"{k} 第{g['round'].iloc[0]}回 {len(g)}試合" for k, g in card.groupby("kind")))
    except Exception as e:
        errors.append(f"対象試合の取得に失敗（前回のデータを表示）: {e}")
        card = prev.get("card")
        if card is None:
            raise
    # 2) 過去結果
    try:
        res = E.fetch_results(DATA / "cache", refresh=force_results, log=lg)
    except Exception as e:
        errors.append(f"試合結果の取得に失敗（前回のデータを使用）: {e}")
        res = prev.get("res")
        if res is None:
            raise
    res = E.clean_res(res, log=lg)
    lg(f"結果データ合計: {len(res)}試合（{res['date'].min():%Y/%m/%d}〜{res['date'].max():%Y/%m/%d}）")

    # 3) toto 投票率
    votes = {}
    for kind, g in card.groupby("kind"):
        if kind.startswith("mini"):
            continue
        n = len(g) * (2 if kind == "totoGOAL3" else 1)
        try:
            votes[kind] = E.fetch_vote_rates(int(g["round"].iloc[0]), kind, n)
            lg(f"{kind} 投票率: {'取得' if votes[kind] else '未公開'}")
        except Exception as e:
            lg(f"{kind} 投票率の取得に失敗: {e}")
    if votes.get("toto"):
        t = card[card["kind"] == "toto"].sort_values("no")
        key = {(E.to_short(h), E.to_short(a)): v for (h, a), v in zip(zip(t["home"], t["away"]), votes["toto"])}
        for kind in ["mini toto-A組", "mini toto-B組"]:
            g = card[card["kind"] == kind].sort_values("no")
            vv = [key.get((E.to_short(h), E.to_short(a))) for h, a in zip(g["home"], g["away"])]
            if len(g) and all(x is not None for x in vv):
                votes[kind] = vv

    last = res["date"].max()
    lag = (E.now_jst().date() - last.date()).days
    if lag > 21:
        errors.append(f"試合結果が {last:%Y/%m/%d} で止まっています（{lag}日前）。")

    state = R.load_state(DATA, password, lg)

    # 4) v7: 直前情報（失敗しても予想は続ける）
    wm = []
    try:
        wm = E.fetch_winner_odds(lg)
    except Exception as e:
        lg(f"WINNERオッズの取得に失敗（市場なしで続行）: {e}")
    try:
        n = R.record_susp(state, E.fetch_suspensions(lg))
        lg(f"出場停止の記録: {len(state.get('susp_hist', []))}件（今回追加 {n}）")
    except Exception as e:
        lg(f"出場停止の取得に失敗（続行）: {e}")
    R.record_votes(state, card, votes)

    # 5) 振り返り
    L0 = state.get("learning", {})
    cur = {k: int(g["round"].iloc[0]) for k, g in card.groupby("kind")}
    rc = R.ResultCache(lg)
    try:
        prev_keys = R.find_previous(state, cur, rc, res, lg)
    except Exception:
        lg("振り返る回の特定で失敗:\n" + traceback.format_exc())
        prev_keys = {}
    reviewed = set(L0.get("reviewed", []))
    arch = state.get("archive", {})
    reviews, new_reviews = {}, []
    for kind, key in prev_keys.items():
        e = arch[key]
        drift = R.vote_drift(state, e["round"]) if kind == "toto" else None
        try:
            rv = R.review_entry(e, res, drift)
        except Exception:
            lg(f"第{e['round']}回 {kind} の振り返りで失敗:\n" + traceback.format_exc())
            continue
        reviews[kind] = rv
        s = rv["summary"]
        lg(f"振り返り {kind} 第{s['round']}回: 結果 {s['known']}/{s['n']} 確定・的中 {s['hits']}"
           f"{'（最終）' if s['complete'] else '（途中経過）'}")
        if s["complete"] and key not in reviewed:
            new_reviews.append(rv)
    n_led = R.update_ledger(state, prev_keys, reviews)
    lg(f"外れの記録: {len(state.get('ledger', []))}試合（今回追加 {n_led}）")

    # 6) 学習
    lg("学習開始")
    M = E.train(res, hp=L0.get("hp"), log=lg)
    lg(f"モデル設定: {LRN._hp_text(M['hp'])}{'（今回再調整）' if M['retuned'] else ''}")
    if "toto" in cur:
        try:
            R.vote_history(state, cur["toto"], rc, res, M["hp"], M["w"], lg)
        except Exception:
            lg("投票率の学習データ収集で失敗（続行）:\n" + traceback.format_exc())
    L = LRN.update(state, M, new_reviews, log=lg, res=res)
    for f in L.get("findings", [])[:5]:
        lg(f"  診断: {f['text']}（z={f['z']}）")
    lg(f"学習完了 / 採用中の補正: {[E.FEAT_INFO[a][0] for a in L['active']] or 'なし'}"
       f" / 市場の重み {L['market_a']}（得点 {L['market_ag']}） / 投票率 b={L['vote_b']} c={L['vote_c']}")

    # 7) 予想（市場を混ぜる前の予想を WINNER の記録にも残す）
    smap = E.susp_map(state.get("susp_hist", []))
    pred = E.Predictor(res, M["poisson"], M["elo"], M["w"], M["base"], M["stacker"], M["vote_b"], M["vote_c"],
                       resid=M["resid"], standing=M["standing"], susp=smap,
                       market_a=M["market_a"], market_ag=M["market_ag"])
    try:
        n = R.record_market(state, wm, pred)
        lg(f"WINNERオッズの記録: {len(state.get('market_hist', {}))}試合（今回更新 {n}）")
    except Exception:
        lg("WINNERオッズの記録で失敗（続行）:\n" + traceback.format_exc())
    market = {**R.market_saved(state), **R.market_now(wm)}
    analysis = E.analyze_card(card, pred, votes, market)
    nm = sum(1 for k in analysis for r in analysis[k] if r.get("m") is not None or r.get("m_g_h") is not None)
    lg(f"対象試合のうち WINNERのオッズあり: {nm}試合")
    LRN.attach_effects(L, analysis)
    try:
        leagues = E.team_leagues(res)
    except Exception as e:
        lg(f"所属リーグの判定に失敗: {e}")
        leagues = {}
    R.archive_current(state, card, analysis, lg)
    R.save_state(state, DATA)
    lg("分析完了")

    snap = {"ver": SNAP_VER, "updated": E.now_jst(), "card": card, "res": res, "votes": votes, "model": M,
            "analysis": analysis, "leagues": leagues, "logs": logs, "errors": errors,
            "reviews": reviews, "learning": L, "state": state}
    DATA.mkdir(exist_ok=True)
    tmp = SNAP.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(snap, f)
    tmp.replace(SNAP)
    return snap


if __name__ == "__main__":
    import os
    import sys
    s = refresh(force_results="--refresh" in sys.argv, password=os.environ.get("SITE_PASSWORD"))
    print("更新完了:", s["updated"], "エラー:", s["errors"] or "なし")
