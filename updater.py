# -*- coding: utf-8 -*-
"""
自動更新：一定間隔で 最新開催回・投票率・試合結果 を取り直し、モデルを再学習して保存する。
  単体で1回だけ更新:  python updater.py
"""
import pickle
import threading
import time
import traceback
from pathlib import Path

import engine as E

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SNAP = DATA / "snapshot.pkl"
INTERVAL_MIN = 60          # 通常の更新間隔（分）
VOTE_INTERVAL_MIN = 20     # 販売締切が近い日（当日・前日）の更新間隔（分）
_lock = threading.Lock()


def load_snapshot():
    if SNAP.exists():
        with open(SNAP, "rb") as f:
            return pickle.load(f)
    return None


def refresh(force_results=False, log=print):
    """全データを取り直して snapshot を作る。失敗した部分は前回の値を使う。"""
    with _lock:
        prev = load_snapshot() or {}
        logs, errors = [], []
        lg = lambda m: (logs.append(f"{E.now_jst():%H:%M:%S} {m}"), log(m))  # noqa

        # 1) 対象試合
        try:
            card = E.fetch_card()
            lg("対象試合: " + ", ".join(f"{k} 第{g['round'].iloc[0]}回 {len(g)}試合"
                                     for k, g in card.groupby("kind")))
        except Exception as e:
            errors.append(f"対象試合の取得失敗: {e}")
            card = prev.get("card")
            if card is None:
                raise
        # 2) 過去結果
        try:
            res = E.fetch_results(DATA / "cache", refresh=force_results, log=lg)
        except Exception as e:
            errors.append(f"試合結果の取得失敗: {e}")
            res = prev.get("res")
            if res is None:
                raise
        # 3) 投票率
        votes = {}
        for kind, g in card.groupby("kind"):
            if kind.startswith("mini"):              # mini は toto の投票率を流用（下で割り当て）
                continue
            n = len(g) * (2 if kind == "totoGOAL3" else 1)
            rnd = int(g["round"].iloc[0])
            try:
                votes[kind] = E.fetch_vote_rates(rnd, kind, n)
            except Exception as e:
                errors.append(f"{kind} 投票率の取得失敗: {e}")
        # mini toto は toto の同一カードの投票率を割り当て
        if votes.get("toto"):
            t = card[card["kind"] == "toto"].sort_values("no")
            key = {(E.to_short(h), E.to_short(a)): v for (h, a), v in zip(zip(t["home"], t["away"]), votes["toto"])}
            for kind in ["mini toto-A組", "mini toto-B組"]:
                g = card[card["kind"] == kind].sort_values("no")
                vv = [key.get((E.to_short(h), E.to_short(a))) for h, a in zip(g["home"], g["away"])]
                if len(g) and all(x is not None for x in vv):
                    votes[kind] = vv
        # 4) 学習・分析
        M = E.train(res)
        analysis = E.analyze_card(card, res, M, votes)
        lg(f"学習 {len(res)}試合 / アンサンブル重み ポアソン{M['w']:.0%}")
        snap = {"updated": E.now_jst(), "card": card, "res": res, "votes": votes,
                "model": M, "analysis": analysis, "logs": logs, "errors": errors}
        DATA.mkdir(exist_ok=True)
        tmp = SNAP.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(snap, f)
        tmp.replace(SNAP)
        return snap


def _next_interval(snap):
    try:
        today = E.now_jst().date()
        days = (snap["card"]["date"].min() - today).days
        return VOTE_INTERVAL_MIN if days <= 1 else INTERVAL_MIN
    except Exception:
        return INTERVAL_MIN


def loop():
    while True:
        snap = None
        try:
            snap = refresh()
        except Exception:
            traceback.print_exc()
        time.sleep(_next_interval(snap or load_snapshot() or {}) * 60)


def start_background():
    t = threading.Thread(target=loop, daemon=True, name="toto-updater")
    t.start()
    return t


if __name__ == "__main__":
    s = refresh(force_results="--refresh" in __import__("sys").argv)
    print("更新完了:", s["updated"], "エラー:", s["errors"] or "なし")
