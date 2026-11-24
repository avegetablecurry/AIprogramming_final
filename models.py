from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_KO = {
    "competition": "경쟁률", "waitlist": "최종 충원 예비순위",
    "final_avg": "최종합격자 평균등급", "final_cut80": "최종합격자 80%컷",
    "final_cut70": "최종합격자 70%컷",
}
MIN_TRAIN_ROWS = 5
MIN_TRAIN_YEARS = 3
MODELS = ("Random Forest", "Ridge")


def set_korean_font():
    candidates = ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK JP"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for family in candidates:
        if family in installed:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False


def num(s):
    if s is None:
        return np.nan
    raw = str(s).strip().replace(",", "")
    if not raw or raw in {"-", "미공개", "NaN", "nan"}:
        return np.nan
    try:
        return float(raw)
    except ValueError:
        return np.nan


def present(x):
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def make_lags(rows, history):
    """이전 '학년도' 행만 사용. 다른 전형이나 이후 연도 결과를 가져오지 않음."""
    lookup = {(r["mode"], r["track"], r["year"]): r for r in history}
    result = []
    for original in rows:
        r = dict(original)
        previous = lookup.get((r["mode"], r["track"], r["year"] - 1))
        for c in ("competition", "final_avg", "waitlist", "final_cut70", "final_cut80"):
            r["prev_" + c] = previous.get(c, np.nan) if previous else np.nan
        result.append(r)
    return result


def columns_for(mode, target):
    nums = ["year", "quota", "prev_competition", "prev_final_avg", "prev_waitlist"]
    if target != "competition":
        nums.append("competition")  # 합격선 발표 전, 이미 확정된 경쟁률만 조건부 입력
    cats = ["track"] if mode == "수시" else ["recruitment_group"]
    return nums, cats


def feature_data(rows, mode, target):
    """동일한 입력 필드 정의를 학습, 검증, 2027 예측 모두에 적용."""
    nums, cats = columns_for(mode, target)
    cols = nums + cats
    matrix = np.empty((len(rows), len(cols)), dtype=object)
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            if col in cats:
                matrix[i, j] = row.get(col) or "미확인"
            else:
                matrix[i, j] = row.get(col, np.nan)
    return matrix, nums, cats


def make_model(algorithm, nums, cats):
    """RF는 결정트리, Ridge는 StandardScaler와 L2 정규화 사용."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if algorithm == "Ridge":
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    prep = ColumnTransformer(
        [("numeric", numeric_pipe, list(range(len(nums)))),
         ("categorical", cat_pipe, list(range(len(nums), len(nums) + len(cats))))],
        sparse_threshold=0,
    )
    if algorithm == "Random Forest":
        reg = RandomForestRegressor(n_estimators=150, max_depth=3,
                                    min_samples_leaf=2, random_state=42, n_jobs=-1)
    else:
        reg = Ridge(alpha=10.0)
    return Pipeline([("prep", prep), ("model", reg)])


def fit_model(rows, mode, target, algorithm):
    X, nums, cats = feature_data(rows, mode, target)
    y = np.array([r[target] for r in rows], dtype=float)
    model = make_model(algorithm, nums, cats)
    model.fit(X, y)
    return model


def adequate(train):
    return len(train) >= MIN_TRAIN_ROWS and len({r["year"] for r in train}) >= MIN_TRAIN_YEARS


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None or (isinstance(row.get(k), (float, np.floating)) and not np.isfinite(row[k])) else row.get(k, "")) for k in fieldnames})


def build_shap(mode, target, fitted, future, outdir):
    """SHAP은 정확도를 높이는 과정이 아닌 Random Forest 개별 예측 설명 단계."""
    key = (mode, target, "Random Forest")
    if key not in fitted:
        return "해당 모델이 학습되지 않아 SHAP 분석을 생략"
    target_future = next((r for r in future if r["mode"] == mode and present(r.get("quota")) and present(r.get("competition"))), None)
    if target_future is None:
        return "2027 입력 데이터가 없어 SHAP 분석을 생략"
    try:
        import shap
        pipe = fitted[key]
        X, nums, cats = feature_data([target_future], mode, target)
        transform = pipe.named_steps["prep"]
        X2 = np.asarray(transform.transform(X), dtype=float)
        # 배열 기반 입력에서는 sklearn 출력명이 x0, x1이 될 수 있으므로
        # 우리가 정의한 원래 열 이름으로 SHAP 설명 라벨을 복원한다.
        cat_names = list(transform.named_transformers_["categorical"].get_feature_names_out(cats))
        names = list(nums) + cat_names
        model = pipe.named_steps["model"]
        explainer = shap.TreeExplainer(model)
        explanations = explainer(X2)
        values = np.asarray(explanations.values)[0]
        base = float(np.ravel(explanations.base_values)[0])
        np.testing.assert_allclose(base + values.sum(), pipe.predict(X)[0], atol=1e-7)
        data = []
        for nm, val, raw in zip(names, values, X2[0]):
            data.append({"mode": mode, "track": target_future["track"],
                         "feature": nm, "feature_value_after_preprocessing": float(raw), "shap_contribution": float(val)})
        write_csv(outdir / f"SHAP_{'susi' if mode=='수시' else 'jeongsi'}_{target}_values.csv", data,
                  ["mode", "track", "feature", "feature_value_after_preprocessing", "shap_contribution"])
        # 작은 프로젝트에서는 기본 waterfall의 'Feature 0' 표시를 피하려고,
        # SHAP 값을 직접 그린 막대그래프를 제공한다(계산 자체는 TreeExplainer).
        names_ko = {
            "year": "학년도", "quota": "모집인원", "competition": "경쟁률",
            "prev_competition": "전년도 경쟁률", "prev_final_avg": "전년도 평균등급",
            "prev_waitlist": "전년도 예비순위",
        }
        pretty_names = [names_ko.get(n, n.replace("track_", "전형: ").replace("recruitment_group_", "모집군: ")) for n in names]
        ranked = sorted(zip(pretty_names, values), key=lambda x: abs(x[1]), reverse=True)[:10]
        fig, ax = plt.subplots(figsize=(9.4, max(4.4, len(ranked)*0.47+1.7)), constrained_layout=True)
        for index, (label, val) in enumerate(reversed(ranked)):
            col = "#2F8C94" if val < 0 else "#D8904B"
            ax.barh(index, val, color=col, height=.68)
            ax.text(val + (0.004 if val >= 0 else -0.004), index, f"{val:+.3f}",
                    va="center", ha="left" if val >= 0 else "right", fontsize=9)
        ax.set_yticks(range(len(ranked)), [nm for nm, _ in reversed(ranked)])
        ax.axvline(0, color="#6B7280", lw=1)
        lim=max([abs(x[1]) for x in ranked] + [.01])
        ax.set_xlim(-lim*1.43, lim*1.43)
        ax.set_xlabel("예측 평균등급에 대한 SHAP 기여도 (음수 = 등급 숫자 감소)")
        ax.set_title(f"{mode} {target_future['track']} | {TARGET_KO[target]} SHAP 분석\n"
                     f"기준 {base:.3f}등급 + 기여도 {sum(values):+.3f} = 예측 {base + sum(values):.3f}등급")
        fig.savefig(outdir / f"SHAP_{'susi' if mode=='수시' else 'jeongsi'}_{target}.png", dpi=160)
        plt.close(fig)
        return f"SHAP 완료: 기준값 {base:.3f}, 기여도 합계 {sum(values):+.3f}"
    except Exception as e:
        plt.close('all')
        return f"SHAP만 생략됨 (다른 실험은 정상): {e.__class__.__name__}: {str(e)[:140]}"


