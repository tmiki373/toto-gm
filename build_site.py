# -*- coding: utf-8 -*-
"""
GitHub Actions 上で実行：取得 → 振り返り → 学習 → 予想 → 暗号化した静的サイトを public/ に出力
  必要な環境変数: SITE_PASSWORD
"""
import json
import math
import os
import shutil
import traceback
from pathlib import Path

import numpy as np

import engine as E
import review as R
import updater as U

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "public"


def f(x, nd=4):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [f(v, nd) for v in x]
    x = float(x)
    return None if math.isnan(x) else round(x, nd)


def form(fm):
    fm = fm or {}
    return {"marks": fm.get("marks", ""), "gf": int(fm.get("gf", 0)), "ga": int(fm.get("ga", 0)), "pts": int(fm.get("pts", 0))}


def to_payload(snap):
    M, A, card, res = snap["model"], snap["analysis"], snap["card"], snap["res"]
    leagues = snap.get("leagues") or {}
    L = snap["learning"]

    recent = res[res["date"] >= res["date"].max() - np.timedelta64(200, "D")]
    active = set(recent["home"]) | set(recent["away"])
    rt = M["poisson"].ratings()
    rt["elo"] = rt["チーム"].map(M["elo"].r)
    rt["lg"] = rt["チーム"].map(leagues).fillna("-")
    rt = rt[rt["チーム"].isin(active)].sort_values("elo", ascending=False)
    rt["rank"] = rt.groupby("lg").cumcount() + 1
    rank = dict(zip(rt["チーム"], zip(rt["lg"], rt["rank"])))
    ratings = [{"team": r["チーム"], "lg": r["lg"], "rank": int(r["rank"]), "att": f(r["攻撃力"], 2),
                "def": f(r["守備力(失点倍率)"], 2), "elo": f(r["elo"], 0)} for _, r in rt.iterrows()]

    def tinfo(short):
        lg, rk = rank.get(short, ("", None))
        return {"lg": lg, "rank": int(rk) if rk is not None else None}

    kinds = []
    for kind in E.KINDS:
        if kind not in A:
            continue
        g = card[card["kind"] == kind]
        rows = []
        for r in A[kind]:
            lam = r.get("lam") or (None, None)
            d = {"no": r["no"], "date": r["date"].strftime("%m/%d"), "dow": "月火水木金土日"[r["date"].weekday()],
                 "time": r.get("time", ""), "home": r["home"], "away": r["away"],
                 "ti_h": tinfo(r["hs"]), "ti_a": tinfo(r["as"]),
                 "p": f(r["p"]), "p_base": f(r.get("p_base")), "v": f(r.get("v")),
                 "lam": f(lam, 2) if lam[0] is not None else None,
                 "scores": [[i, j, f(p, 3)] for i, j, p in r.get("scores", [])],
                 "p_poi": f(r.get("p_poi")), "p_elo": f(r.get("p_elo")),
                 "elo_h": f(r.get("elo_h"), 0), "elo_a": f(r.get("elo_a"), 0),
                 "form_h": form(r.get("form_h")), "form_a": form(r.get("form_a")),
                 "rest_h": r.get("rest_h"), "rest_a": r.get("rest_a"), "h2h": r.get("h2h", ""), "note": r.get("note", "")}
            if kind == "totoGOAL3":
                d.update({"g_h": f(r["g_h"]), "g_a": f(r["g_a"]), "v_h": f(r.get("v_h")), "v_a": f(r.get("v_a"))})
            rows.append(d)
        dl = g["deadline"].dropna().iloc[0] if g["deadline"].notna().any() else None
        kinds.append({"kind": kind, "round": int(g["round"].iloc[0]), "deadline": dl, "rows": rows})

    reviews = []
    for kind in E.KINDS:
        rv = snap.get("reviews", {}).get(kind)
        if rv:
            reviews.append({"kind": kind, "summary": rv["summary"],
                            "rows": [{k: (f(v) if isinstance(v, (list, np.ndarray)) and k in ("p", "v") else v)
                                      for k, v in r.items() if k != "feat"} for r in rv["rows"]]})

    feats = []
    for k, (name, desc) in E.FEAT_INFO.items():
        st = "採用中" if k in L["active"] else "検証済み・見送り" if k in L["scores"] else "検証待ち" if k in L["pool"] else "未着手"
        feats.append({"key": k, "name": name, "desc": desc, "status": st, "gain": L["scores"].get(k),
                      "since": L["adopted_at"].get(k)})
    learning = {"features": feats, "history": L["history"][-40:][::-1], "track": L["track"][-30:],
                "vote": {"b": L["vote_b"], "n": L["vote_n"], "need": 30},
                "params": {k: round(v, 3) for k, v in M["stacker"].params().items()}}

    s, e, n = M["test_period"]
    last = res["date"].max()
    by_month = res.assign(y=res["date"].dt.to_period("M").astype(str)).groupby("y").size()
    return {"updated": snap["updated"].isoformat(), "errors": snap["errors"], "logs": snap["logs"],
            "kinds": kinds, "reviews": reviews, "learning": learning,
            "backtest": [{k: (f(v) if k != "name" else v) for k, v in b.items()} for b in M["backtest"]],
            "calib": [{**c, "pred": f(c["pred"]), "act": f(c["act"])} for c in M["calib"]], "ratings": ratings,
            "test": f"{s:%Y/%m/%d}〜{e:%Y/%m/%d}の{n}試合",
            "n_res": len(res), "res_from": f"{res['date'].min():%Y/%m/%d}", "res_to": f"{last:%Y/%m/%d}",
            "lag": (E.now_jst().date() - last.date()).days,
            "monthly": [[k, int(v)] for k, v in by_month.tail(18).items()],
            "home_adv": f(math.exp(M["poisson"].home_adv), 2), "w_poi": f(M["w"], 2)}


def main():
    pw = os.environ.get("SITE_PASSWORD")
    if not pw:
        raise SystemExit("環境変数 SITE_PASSWORD が設定されていません（GitHub の Secrets に登録してください）")
    try:
        snap = U.refresh(force_results=os.environ.get("FORCE_REFRESH") == "1", password=pw)
    except Exception as e:
        print("=" * 60, "\n[更新処理でエラー] 以下が本当の原因です\n", "=" * 60, flush=True)
        traceback.print_exc()
        snap = U.load_snapshot()
        if snap is None or snap.get("ver") != U.SNAP_VER:
            raise SystemExit("前回の互換データがないため停止します（上のエラーを確認してください）")
        snap["errors"] = snap.get("errors", []) + [f"今回の更新に失敗したため前回のデータを表示: {type(e).__name__}: {e}"]
    payload = to_payload(snap)
    OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(R.encrypt(payload, pw)), encoding="utf-8")
    (OUT / "state.json").write_text(json.dumps(R.encrypt(snap["state"], pw)), encoding="utf-8")
    for fn in ["index.html", "manifest.webmanifest", "icon.svg"]:
        shutil.copy(ROOT / "web" / fn, OUT / fn)
    (OUT / ".nojekyll").write_text("")
    (OUT / "config.json").write_text(json.dumps({"repo": os.environ.get("GITHUB_REPOSITORY", "")}), encoding="utf-8")
    print("サイト生成完了:", payload["updated"], "最新結果:", payload["res_to"], "エラー:", payload["errors"] or "なし")


if __name__ == "__main__":
    main()
