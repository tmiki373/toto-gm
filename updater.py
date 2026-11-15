# -*- coding: utf-8 -*-
"""最新開催回・投票率・試合結果を取り直し、モデルを再学習して data/snapshot.pkl に保存する"""
import pickle
import traceback
from pathlib import Path

import pandas as pd

import engine as E

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SNAP = DATA / "snapshot.pkl"
SNAP_VER = 2          # 保存形式のバージョン（古い形式は読み込まない）


def load_snapshot():
    if SNAP.exists():
        try:
            with open(SNAP, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def refresh(force_results=False, log=print):
    prev = load_snapshot() or {}
    if prev.get("ver") != SNAP_VER:
        prev = {}
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
    # 保存済みCSV（文字列の日付）と新規取得分（日付型）が混ざると日付列が文字列扱いになるため統一
    res = res.copy()
    res["date"] = pd.to_datetime(res["date"].astype(str).str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    res["hg"] = pd.to_numeric(res["hg"], errors="coerce")
    res["ag"] = pd.to_numeric(res["ag"], errors="coerce")
    bad = res[["date", "home", "away", "hg", "ag"]].isna().any(axis=1).sum()
    if bad:
        lg(f"不正な行を除外: {bad}件")
    res = res.dropna(subset=["date", "home", "away", "hg", "ag"]).drop_duplicates(["date", "home", "away"])
    res["comp"] = res["comp"].fillna("").astype(str)
    res["home"], res["away"] = res["home"].astype(str), res["away"].astype(str)
    res[["hg", "ag"]] = res[["hg", "ag"]].astype(int)
    res = res.sort_values("date").reset_index(drop=True)
    lg(f"結果データ合計: {len(res)}試合（{res['date'].min():%Y/%m/%d}〜{res['date'].max():%Y/%m/%d}）")

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

    # 5) 学習・分析（どこで失敗したか分かるよう段階ごとにログ）
    lg("学習開始")
    try:
        M = E.train(res)
    except Exception:
        lg("学習で失敗:\n" + traceback.format_exc())
        raise
    lg(f"学習完了 / アンサンブル重み ポアソン{M['w']:.0%}")
    try:
        analysis = E.analyze_card(card, res, M, votes)
    except Exception:
        lg("試合分析で失敗:\n" + traceback.format_exc())
        raise
    try:
        leagues = E.team_leagues(res)
    except Exception as e:
        lg(f"所属リーグの判定に失敗（順位表示なしで続行）: {e}")
        leagues = {}
    lg("分析完了")

    snap = {"ver": SNAP_VER, "updated": E.now_jst(), "card": card, "res": res, "votes": votes, "model": M,
            "analysis": analysis, "leagues": leagues, "logs": logs, "errors": errors}
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
