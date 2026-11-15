# -*- coding: utf-8 -*-
"""
GitHub Actions 上で実行：最新データ取得 → 学習・分析 → 暗号化データ付きの静的サイトを public/ に出力
  必要な環境変数: SITE_PASSWORD
"""
import base64
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import engine as E
import updater as U

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "public"
ITER = 250_000


def f(x, nd=4):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [f(v, nd) for v in x]
    x = float(x)
    return None if math.isnan(x) else round(x, nd)


def form(fm):
    return {"marks": fm["marks"], "gf": int(fm["gf"]), "ga": int(fm["ga"])}


def to_payload(snap):
    M, A, card = snap["model"], snap["analysis"], snap["card"]
    leagues = snap.get("leagues") or E.team_leagues(snap["res"])

    # チーム力ランキング（現所属カテゴリ内の Elo 順位も付与）
    res = snap["res"]
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
        return {"lg": lg, "rank": rk}

    kinds = []
    for kind in E.KINDS:
        if kind not in A:
            continue
        g = card[card["kind"] == kind]
        rows = []
        for r in A[kind]:
            d = {"no": r["no"], "date": r["date"].strftime("%m/%d"),
                 "dow": "月火水木金土日"[r["date"].weekday()], "time": r["time"],
                 "home": r["home"], "away": r["away"], "ti_h": tinfo(r["hs"]), "ti_a": tinfo(r["as"]),
                 "p": f(r["p"]), "v": f(r.get("v")),
                 "lam": f(r["lam"], 2) if r["lam"][0] is not None else None,
                 "scores": [[i, j, f(p, 3)] for i, j, p in r["scores"]],
                 "p_poi": f(r["p_poi"]), "p_elo": f(r["p_elo"]),
                 "elo_h": f(r["elo_h"], 0), "elo_a": f(r["elo_a"], 0),
                 "form_h": form(r["form_h"]), "form_a": form(r["form_a"]),
                 "rest_h": r["rest_h"], "rest_a": r["rest_a"], "h2h": r["h2h"], "note": r["note"]}
            if kind == "totoGOAL3":
                d.update({"g_h": f(r["g_h"]), "g_a": f(r["g_a"]), "v_h": f(r.get("v_h")), "v_a": f(r.get("v_a"))})
            rows.append(d)
        dl = g["deadline"].dropna().iloc[0] if "deadline" in g and g["deadline"].notna().any() else None
        kinds.append({"kind": kind, "round": int(g["round"].iloc[0]), "deadline": dl, "rows": rows})

    bt = [{"name": n, "ll": f(r["対数損失"]), "brier": f(r["Brier"]), "acc": f(r["的中率(本命)"])}
          for n, r in M["backtest"].iterrows()]
    cal = [{"band": r["予測確率帯"], "n": int(r["試合数"]), "pred": f(r["平均予測"]), "act": f(r["実際の的中率"])}
           for _, r in M["calib"].iterrows()]
    s, e, n = M["test_period"]
    last = res["date"].max()
    by_season = res.assign(y=res["date"].dt.to_period("M").astype(str)).groupby("y").size()
    return {"updated": snap["updated"].isoformat(), "errors": snap["errors"], "logs": snap["logs"],
            "kinds": kinds, "backtest": bt, "calib": cal, "ratings": ratings,
            "test": f"{s:%Y/%m/%d}〜{e:%Y/%m/%d}の{n}試合",
            "n_res": len(res), "res_from": f"{res['date'].min():%Y/%m/%d}", "res_to": f"{last:%Y/%m/%d}",
            "lag": (E.now_jst().date() - last.date()).days,
            "monthly": [[k, int(v)] for k, v in by_season.tail(18).items()],
            "home_adv": f(math.exp(M["poisson"].home_adv), 2), "w_poi": f(M["w"], 2)}


def encrypt(obj, password):
    salt, iv = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITER).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, json.dumps(obj, ensure_ascii=False).encode(), None)
    b = lambda x: base64.b64encode(x).decode()  # noqa
    return {"salt": b(salt), "iv": b(iv), "ct": b(ct), "iter": ITER}


def main():
    pw = os.environ.get("SITE_PASSWORD")
    if not pw:
        raise SystemExit("環境変数 SITE_PASSWORD が設定されていません（GitHub の Secrets に登録してください）")
    try:
        snap = U.refresh(force_results=os.environ.get("FORCE_REFRESH") == "1")
    except Exception as e:
        snap = U.load_snapshot()
        if snap is None:
            raise
        snap["errors"] = snap.get("errors", []) + [f"今回の更新に失敗したため前回のデータを表示: {e}"]
    payload = to_payload(snap)
    OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(encrypt(payload, pw)), encoding="utf-8")
    for fn in ["index.html", "manifest.webmanifest", "icon.svg"]:
        shutil.copy(ROOT / "web" / fn, OUT / fn)
    (OUT / ".nojekyll").write_text("")
    (OUT / "config.json").write_text(json.dumps({"repo": os.environ.get("GITHUB_REPOSITORY", "")}), encoding="utf-8")
    print("サイト生成完了:", payload["updated"], "最新結果:", payload["res_to"], "エラー:", payload["errors"] or "なし")


if __name__ == "__main__":
    main()
