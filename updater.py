# -*- coding: utf-8 -*-
"""取得 → 振り返り → 学習（設定の自動調整・検討要素・投票率）→ 予想 → 保存 をまとめて実行   v6（オッズ市場モデル）"""
import pickle
import traceback
from pathlib import Path

import engine as E
import learning as LRN
import review as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SNAP = DATA / "snapshot.pkl"
SNAP_VER = 3


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

    # 1) 対象試合（次回）
    try:
        card = E.fetch_card()
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

    # 3) 投票率（次回）
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

    # v6: 任意の data/odds.csv。未取得・不正でも空辞書で安全に継続。
    import os
    odds_path = Path(os.environ.get("ODDS_CSV_PATH", DATA / "odds.csv"))
    odds = E.load_odds(card, odds_path)
    lg(f"オッズ: {len(odds)}試合取得" if odds else "オッズ: 未取得（従来モデルのみで継続）")

    last = res["date"].max()
    lag = (E.now_jst().date() - last.date()).days
    if lag > 21:
        errors.append(f"試合結果が {last:%Y/%m/%d} で止まっています（{lag}日前）。")

    # 4) 直前の回の振り返り
    state = R.load_state(DATA, password, lg)
    L0 = state.get("learning", {})
    cur = {k: int(g["round"].iloc[0]) for k, g in card.groupby("kind")}
    rc = R.ResultCache(lg)
    try:
        prev_keys = R.find_previous(state, cur, rc, res, lg)
    except Exception:
        lg("直前の回の特定で失敗:\n" + traceback.format_exc())
        prev_keys = {}
    reviewed = set(L0.get("reviewed", []))
    arch = state.get("archive", {})
    reviews, new_reviews = {}, []
    for kind, key in prev_keys.items():
        e = arch[key]
        try:
            rv = R.review_entry(e, res)
        except Exception:
            lg(f"第{e['round']}回 {kind} の振り返りで失敗:\n" + traceback.format_exc())
            continue
        if kind != "totoGOAL3":
            e["_review_rows"] = [{"home": r["home"], "away": r["away"], "date": r["date"], "actual": it["actual"],
                                  "v": r.get("v"), "p_model": r.get("p_model") or r["p"]}
                                 for r, it in zip(e["rows"], rv["rows"])]
        reviews[kind] = rv
        s = rv["summary"]
        lg(f"振り返り {kind} 第{s['round']}回: 結果 {s['known']}/{s['n']} 確定・的中 {s['hits']}")
        if s["complete"] and key not in reviewed:
            new_reviews.append(rv)

    # v6: 完了した回だけをオッズ学習履歴へ追加（途中結果・未来情報は不使用）。
    oh = state.setdefault("odds_hist", [])
    seen_odds = {(x.get("round"), x.get("kind"), x.get("no")) for x in oh}
    for kind, key in prev_keys.items():
        if kind == "totoGOAL3" or kind not in reviews:
            continue
        rv, e = reviews[kind], arch[key]
        if not rv["summary"].get("complete"):
            continue
        for saved, judged in zip(e.get("rows", []), rv.get("rows", [])):
            k = (int(e["round"]), kind, int(saved["no"]))
            pm, actual = saved.get("p_market"), judged.get("actual")
            p0 = saved.get("p_pre_odds") or saved.get("p_model") or saved.get("p")
            if k in seen_odds or pm is None or p0 is None or actual is None:
                continue
            oh.append({"round": k[0], "kind": kind, "no": k[2], "date": saved.get("date", ""),
                       "p_model": p0, "p_market": pm, "a": int(actual)})
            seen_odds.add(k)
    state["odds_hist"] = oh[-1200:]

    # 5) 学習
    lg("学習開始")
    M = E.train(res, hp=L0.get("hp"), log=lg)
    lg(f"モデル設定: {LRN._hp_text(M['hp'])}{'（今回再調整）' if M['retuned'] else ''}")
    if "toto" in cur:
        try:
            R.vote_history(state, cur["toto"], rc, res, M["hp"], M["w"], lg)
        except Exception:
            lg("投票率の学習データ収集で失敗（続行）:\n" + traceback.format_exc())
    L = LRN.update(state, M, new_reviews, log=lg)
    lg(f"学習完了 / 採用中の検討要素: {[E.FEAT_INFO[a][0] for a in L['active']] or 'なし'}"
       f" / 投票率の比率 {L['vote_b']}")

    # 6) 次回の予想
    pred = E.Predictor(res, M["poisson"], M["elo"], M["w"], M["base"], M["stacker"], M["vote_b"], L.get("odds_w", 0.0))
    analysis = E.analyze_card(card, pred, votes, odds)
    try:
        leagues = E.team_leagues(res)
    except Exception as e:
        lg(f"所属リーグの判定に失敗: {e}")
        leagues = {}
    R.archive_current(state, card, analysis, lg)
    R.save_state(state, DATA)
    lg("分析完了")

    snap = {"ver": SNAP_VER, "updated": E.now_jst(), "card": card, "res": res, "votes": votes, "odds": odds, "model": M,
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
