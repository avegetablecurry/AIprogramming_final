"""2027년 전형별 결과 및 같은 절차의 과거 연도 검증."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

import models as m

ROOT = Path(__file__).resolve().parent
TRACKS = {
    "일반전형": "일반전형", "지역인재교과우수자전형": "지역인재교과우수자전형",
    "한남인재전형": "한남인재Ⅰ(서류)전형", "지역인재전형": "한남인재Ⅱ(서류+면접)전형",
}
FIELDS = {"Slots": "quota", "ApplyRate": "competition", "Waitlist": "waitlist",
          "FirsMax": "first_best", "FirsMin": "first_worst", "FinalMax": "final_best",
          "FinalAvg": "final_avg"}
LABEL = {"quota": "모집인원", "competition": "경쟁률", "waitlist": "최종 충원 예비순위",
         "first_best": "최초 최고등급", "first_worst": "최초 최저등급", "final_best": "최종 최고등급",
         "final_avg": "최종 평균등급", "final_cut70": "최종 70%컷", "final_cut80": "최종 80%컷",
         "applicants": "지원인원"}
YEAR = 2027


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load(folder):
    susi = []
    for source, track in TRACKS.items():
        for row in read(folder / f"수시_{source}_데이터.csv"):
            r = {key: m.num(row[col]) for col, key in FIELDS.items()}
            if not m.present(r["quota"]):
                continue
            r.update(year=int(row["Year"]), mode="수시", track=track)
            for pct in (70, 80):
                r[f"final_cut{pct}"] = m.num(row["FinalCut"]) if row["CutCriteria"] == f"{pct}%" else np.nan
            susi.append(r)
    conditions = read(folder / "conditions.csv")
    for c in conditions:
        for r in susi:
            if (r["year"], r["track"]) == (int(c["Year"]), c["Track"]):
                r["competition"] = round(int(c["Applicants"]) / int(c["Slots"]), 2)
    susi.sort(key=lambda r: (r["year"], r["track"]))
    regular = defaultdict(list)
    for file, scope in [("ai_학과", "AI융합학과"), ("한남대_전체", "한남대 전체 파일")]:
        for row in read(folder / f"정시_{file}_데이터.csv"):
            if row["Slots"] and row["Applicants"]:
                regular[scope, row["Type"]].append(dict(year=int(row["Year"]), group=row["Group"],
                    quota=int(row["Slots"]), applicants=int(row["Applicants"]), competition=float(row["Rate"])))
    return susi, regular, conditions


def estimate(target, value, method, n=0, rf=None, ridge=None, baseline=None):
    return dict(target=target, value=value, method=method, n=n, RF=rf, Ridge=ridge, prior=baseline)


def susi_forecast(history, queries, explain=None):
    if not queries:
        return []
    year = queries[0]["year"]
    past = [r for r in history if r["year"] < year]
    hist = m.make_lags(past, past)
    future = m.make_lags(queries, past)
    outputs = [[] for _ in future]
    for target in ("waitlist", "first_best", "first_worst", "final_best", "final_avg"):
        train = [r for r in hist if m.present(r[target])]
        if not m.adequate(train):
            continue
        x, _, _ = m.feature_data(future, "수시", target)
        models = [m.fit_model(train, "수시", target, alg) for alg in m.MODELS]
        rf, ridge = [model.predict(x) for model in models]
        if explain and target == "final_avg":
            explain.mkdir(exist_ok=True, parents=True)
            print(m.build_shap("수시", target, {("수시", target, "Random Forest"): models[0]}, future, explain))
        for i, q in enumerate(future):
            prev = next((r[target] for r in past if r["track"] == q["track"] and r["year"] == year-1), np.nan)
            value = (rf[i] + ridge[i]) / 2
            if target == "waitlist":
                value = round(np.clip(value, 0, max(0, round(q["quota"] * q["competition"]) - q["quota"])))
            else:
                value = np.clip(value, 1, 9)
            outputs[i].append(estimate(target, value, "RF·Ridge 평균", len(train), rf[i], ridge[i], prev))
    for group, q in zip(outputs, future):
        avg = next((e["value"] for e in group if e["target"] == "final_avg"), None)
        for pct in (70, 80):
            key = f"final_cut{pct}"
            matched = [r for r in past if r["track"] == q["track"] and m.present(r[key]) and m.present(r["final_avg"])]
            gaps = [r[key] - r["final_avg"] for r in matched]
            value = np.clip(avg + mean(gaps), 1, 9) if gaps and avg is not None else None
            group.append(estimate(key, value, "평균 예측 + 동일 전형·동일 컷의 과거 평균 차이" if value is not None else "동일 기준 자료 부족", len(gaps)))
    return outputs


def jeongsi_forecast(past, year, known_quota=None):
    recent = sorted((r for r in past if r["year"] < year), key=lambda r: r["year"])[-3:]
    if not recent:
        return []
    quota = known_quota if known_quota is not None else max(1, round(median(r["quota"] for r in recent)))
    rate = median(r["competition"] for r in recent)
    applicants = round(quota * rate)
    return [estimate("quota", quota, "요강 인원(이월 전)" if known_quota else "최근 관측 모집인원 중앙값", len(recent)),
            estimate("applicants", applicants, "모집인원 × 최근 관측 경쟁률 중앙값, 정수 반올림", len(recent)),
            estimate("competition", applicants / quota, "예측 지원인원 ÷ 모집인원", len(recent))]


def run(folder, out, shap):
    susi, regular, conditions = load(folder)
    scores, records = [], []
    for suffix in (".png", "_values.csv"):
        (out / "plots" / f"SHAP_susi_final_avg{suffix}").unlink(missing_ok=True)
    if shap:
        m.set_korean_font()
    years = sorted({r["year"] for r in susi if r["year"] < YEAR})
    for year in years + [YEAR]:
        queries = [r for r in susi if r["year"] == year] if year != YEAR else [
            dict(mode="수시", year=year, track=c["Track"], quota=int(c["Slots"]),
                 competition=round(int(c["Applicants"])/int(c["Slots"]), 2))
            for c in conditions if int(c["Year"]) == year]
        groups = susi_forecast(susi, queries, out / "plots" if shap and year == YEAR else None)
        for q, group in zip(queries, groups):
            if year == YEAR:
                for k in ("quota", "competition"):
                    group.append(estimate(k, q[k], "수시 마감 공시값"))
                group.append(estimate("applicants", next(int(c["Applicants"]) for c in conditions if c["Track"]==q["track"] and int(c["Year"])==year), "수시 마감 공시값"))
            for e in group:
                row = dict(year=year, mode="수시", scope="AI융합학과", track=q["track"], group="", **e)
                if year == YEAR:
                    records.append(row)
                elif e["value"] is not None and m.present(q[e["target"]]):
                    scores.append(dict(row, actual=q[e["target"]]))
    for (scope, track), series in regular.items():
        for year in sorted({r["year"] for r in series if r["year"] < YEAR} | {YEAR}):
            known = scope == "AI융합학과" and track == "Normal"
            actual = next((r for r in series if r["year"] == year), {})
            quota = (3 if year == YEAR else actual.get("quota")) if known else None
            prior = [r for r in series if r["year"] < year]
            for e in jeongsi_forecast(prior, year, quota):
                group = actual.get("group") or ("C" if known else max(prior, key=lambda r:r["year"])["group"])
                row = dict(year=year, mode="정시", scope=scope, track=track, group=group, **e)
                if year == YEAR:
                    records.append(row)
                elif not (known and e["target"] == "quota"):
                    scores.append(dict(row, actual=actual[e["target"]]))
    for r in records:
        matched = [s for s in scores if all(s[k] == r[k] for k in ("mode","scope","track","target"))]
        r["검증건수"] = len(matched)
        r["MAE"] = mean(abs(s["actual"]-s["value"]) for s in matched) if matched else None
        r["항목"] = LABEL[r["target"]]
    m.write_csv(out/"2027_전항목.csv", records, list(records[0]))
    m.write_csv(out/"연도별_검증.csv", scores, list(scores[0]))
    return records, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT/"data")
    parser.add_argument("--output", type=Path, default=ROOT/"results")
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()
    records, scores = run(args.data, args.output, not args.no_shap)
    print(f"2027 결과 {len(records)}행, 검증 {len(scores)}행 저장: {args.output}")
