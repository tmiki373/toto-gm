# -*- coding: utf-8 -*-
"""最新開催回・投票率・試合結果を取り直し、モデルを再学習して data/snapshot.pkl に保存する"""
import pickle
from pathlib import Path

import engine as E

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SNAP = DATA / "snapshot.pkl"


def load_snapshot():
    if SNAP.exists():
        with open(SNAP, "rb") as f:
            return pickle.load(f)
    return None


def refresh(force_results=False, log=print):
    prev = load_snapshot() or {}
    logs, errors = [], []

    def lg(m):
        logs.append(f"{E.now_jst():%H:%M:%S} {m}")
        log(m)

    # 1) 対象試合
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
    # 3) 投票率（mini は toto の同じ試合を流用）
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
    # 4) 鮮度チェック
    last = res["date"].max()
    lag = (E.now_jst().date() - last.date()).days
    if lag > 21:
        errors.append(f"試合結果が {last:%Y/%m/%d} で止まっています（{lag}日前）。予想は古いデータに基づいています。")
    # 5) 学習・分析
    M = E.train(res)
    analysis = E.analyze_card(card, res, M, votes)
    lg(f"学習 {len(res)}試合 / アンサンブル重み ポアソン{M['w']:.0%}")
    snap = {"updated": E.now_jst(), "card": card, "res": res, "votes": votes, "model": M,
            "analysis": analysis, "leagues": E.team_leagues(res), "logs": logs, "errors": errors}
    DATA.mkdir(exist_ok=True)
    tmp = SNAP.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(snap, f)
    tmp.replace(SNAP)
    return snap


if __name__ == "__main__":
    import sys
    s = refresh(force_results="--refresh" in sys.argv)
    print("更新完了:", s["updated"], "エラー:", s["errors"] or "なし")
