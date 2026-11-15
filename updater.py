# -*- coding: utf-8 -*-
"""取得 → 振り返り → 学習（検討要素の追加・検証）→ 予想 → 保存 をまとめて実行"""
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
    res = E.clean_res(res, log=lg)
    lg(f"結果データ合計: {len(res)}試合（{res['date'].min():%Y/%m/%d}〜{res['date'].max():%Y/%m/%d}）")

    # 3) 投票率
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

    # 4) 過去の予想を読み込み → 振り返り
    state = R.load_state(DATA, password, lg)
    try:
        R.backfill_previous(state, card, res, lg)
    except Exception:
        lg("前回分の再現で失敗（振り返りなしで続行）:\n" + traceback.format_exc())
    cur = {k: int(g["round"].iloc[0]) for k, g in card.groupby("kind")}
    reviewed = set(state.get("learning", {}).get("reviewed", []))
    reviews, new_reviews = {}, []
    arch = state.get("archive", {})
    for key in sorted(arch, key=lambda k: int(k.split("|")[0])):
        e = arch[key]
        if e["kind"] in cur and e["round"] >= cur[e["kind"]]:
            continue
        try:
            rv = R.review_entry(e, res)
        except Exception:
            lg(f"第{e['round']}回 {e['kind']} の振り返りで失敗:\n" + traceback.format_exc())
            continue
        if e["kind"] != "totoGOAL3":
            e["_review_rows"] = [{"home": r["home"], "away": r["away"], "date": r["date"], "actual": it["actual"],
                                  "v": r.get("v"), "p_model": r.get("p_model") or r["p"]}
                                 for r, it in zip(e["rows"], rv["rows"])]
        reviews[e["kind"]] = rv            # 種別ごとに最新の回が残る＝直前1回
        if rv["summary"]["complete"] and rv["summary"]["known"] and key not in reviewed:
            new_reviews.append(rv)
    lg(f"振り返り: {', '.join(f'{k} 第{v['summary']['round']}回' for k, v in reviews.items()) or 'なし'}"
       f" / 新たに完了 {len(new_reviews)}件")

    # 5) 学習（検討要素の追加・検証・採否）
    lg("学習開始")
    M = E.train(res)
    L = LRN.update(state, M, new_reviews, log=lg)
    lg(f"学習完了 / 採用中の検討要素: {[E.FEAT_INFO[a][0] for a in L['active']] or 'なし'}")

    # 6) 予想
    pred = E.Predictor(res, M["poisson"], M["elo"], M["w"], M["base"], M["stacker"], M["vote_b"])
    analysis = E.analyze_card(card, pred, votes)
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
