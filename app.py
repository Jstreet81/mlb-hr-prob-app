# -*- coding: utf-8 -*-
"""
MLB HR確率予測 Webアプリ（Gradio, 可視化＋PDF出力, NotoSansJP, 1D曲線/比較曲線込）

依存関係（例）:
    pip install -U gradio>=4.44.1 plotly==5.22.0 scikit-learn pandas numpy matplotlib reportlab pillow

CSV（同一ディレクトリ配置想定）:
    mlb2024_top50_hr_bbe.csv
    mlb2024_top50_nonhrhit_bbe.csv
"""

import os
import io
import time
import tempfile
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# Matplotlib / Plotly
import matplotlib
matplotlib.use("Agg")  # headless for Gradio/Spaces
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.graph_objects as go

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

# Gradio
import gradio as gr

# PDF (ReportLab)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ============================
# 設定
# ============================
HR_CSV = os.getenv("HR_CSV", "mlb2024_top50_hr_bbe.csv")
NONHR_CSV = os.getenv("NONHR_CSV", "mlb2024_top50_nonhrhit_bbe.csv")
RANDOM_STATE = 42

NOTO_REG_PATH_CAND = [
    "./NotoSansJP-Regular.ttf",
    "/mnt/data/NotoSansJP-Regular.ttf",
]
NOTO_BOLD_PATH_CAND = [
    "./NotoSansJP-Bold.ttf",
    "/mnt/data/NotoSansJP-Bold.ttf",
]


@dataclass
class ModelBundle:
    name: str
    estimator: object
    best_threshold: float
    feature_indices: Tuple[int, ...]  # (0,1) for EV+LA, (0,) for EV-only


# ============================
# フォント設定
# ============================
def _find_font(cands: List[str]) -> str:
    for p in cands:
        if os.path.exists(p):
            return p
    return ""

def setup_fonts() -> Tuple[str, str]:
    """NotoSansJP を Matplotlib/ReportLab に登録（存在すれば）。"""
    reg = _find_font(NOTO_REG_PATH_CAND)
    bold = _find_font(NOTO_BOLD_PATH_CAND)

    # Matplotlib
    try:
        if reg:
            import matplotlib.font_manager as fm
            fm.fontManager.addfont(reg)
            plt.rcParams["font.family"] = "Noto Sans JP"
            plt.rcParams["font.sans-serif"] = ["Noto Sans JP"]
        plt.rcParams["font.size"] = 11
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        print("Matplotlib font setup failed:", e)

    # ReportLab
    try:
        if reg:
            pdfmetrics.registerFont(TTFont("NotoSansJP-Regular", reg))
        if bold:
            pdfmetrics.registerFont(TTFont("NotoSansJP-Bold", bold))
    except Exception as e:
        print("ReportLab font setup failed:", e)

    return reg, bold


# ============================
# データ & モデル
# ============================
def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    j_best = int(np.argmax(j_scores))
    return float(thresholds[j_best])

def load_dataset(hr_path: str, nonhr_path: str) -> Tuple[np.ndarray, np.ndarray]:
    df_hr = pd.read_csv(hr_path).assign(y=1)
    df_nonhr = pd.read_csv(nonhr_path).assign(y=0)
    df_all = pd.concat([
        df_hr[["launch_speed","launch_angle","y"]],
        df_nonhr[["launch_speed","launch_angle","y"]]
    ], ignore_index=True).dropna()
    X = df_all[["launch_speed","launch_angle"]].values
    y = df_all["y"].values.astype(int)
    return X, y

def build_models(X: np.ndarray, y: np.ndarray) -> Dict[str, ModelBundle]:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # 2D (EV, LA) モデル
    models_2d = {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()),
                                        ("clf", LogisticRegression(max_iter=200, random_state=RANDOM_STATE))]),
        "SVM_RBF": Pipeline([("scaler", StandardScaler()),
                             ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }

    bundles: Dict[str, ModelBundle] = {}

    # 2D モデル学習 & Youden しきい値
    for name, est in models_2d.items():
        est.fit(X_tr, y_tr)
        y_prob_va = est.predict_proba(X_va)[:, 1]
        thr = youden_threshold(y_va, y_prob_va)
        est.fit(X, y)  # フルで再学習
        bundles[name] = ModelBundle(name=name, estimator=est, best_threshold=thr, feature_indices=(0,1))

    # 1D (EVのみ) ロジスティック回帰
    model_1d = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=200, random_state=RANDOM_STATE))])
    model_1d.fit(X_tr[:, [0]], y_tr)
    y_prob_va_1d = model_1d.predict_proba(X_va[:, [0]])[:, 1]
    thr_1d = youden_threshold(y_va, y_prob_va_1d)
    model_1d.fit(X[:, [0]], y)
    bundles["LogReg_EV_only"] = ModelBundle(
        name="LogReg_EV_only", estimator=model_1d, best_threshold=thr_1d, feature_indices=(0,)
    )
    return bundles


# ============================
# 可視化ユーティリティ
# ============================
def make_prob_map_png(model, ev_grid: np.ndarray, la_grid: np.ndarray,
                      ev_point: float, la_point: float, out_path: str) -> str:
    xx, yy = np.meshgrid(ev_grid, la_grid)
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict_proba(grid)[:,1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6,5))
    cs = ax.contourf(xx, yy, zz, levels=20)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("HR確率")
    ax.contour(xx, yy, zz, levels=[0.5])
    ax.scatter([ev_point], [la_point], s=60)
    ax.set_xlabel("打球速度 (mph)")
    ax.set_ylabel("打球角度 (度)")
    ax.set_title("確率マップ＆決定境界")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def make_prob_surface_png(model, ev_grid: np.ndarray, la_grid: np.ndarray, ev_point: float, la_point: float, out_path: str) -> str:
    xx, yy = np.meshgrid(ev_grid, la_grid)
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    fig = plt.figure(figsize=(7, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, alpha=0.5, edgecolor="none")
    ax.view_init(elev=30, azim=-120)
    if ev_point is not None and la_point is not None:
        prob_pt = float(model.predict_proba([[ev_point, la_point]])[0, 1])
        ax.scatter([ev_point], [la_point], [prob_pt], s=80, marker="X")
        try:
            ax.text(ev_point, la_point, prob_pt, f" p={prob_pt:.3f}")
        except Exception:
            pass

    ax.set_xlabel("打球速度 (mph)")
    ax.set_ylabel("打球角度 (度)")
    ax.set_zlabel("HR確率")
    ax.set_title("HR確率 3D曲面")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def _eye_from_elev_azim(elev_deg=30, azim_deg=-120, r=2.2):
    th = np.deg2rad(elev_deg)
    ph = np.deg2rad(azim_deg)
    return dict(
        x = float(r*np.cos(th)*np.cos(ph)),
        y = float(r*np.cos(th)*np.sin(ph)),
        z = float(r*np.sin(th))
    )

def make_prob_surface_plotly(model, ev_grid: np.ndarray, la_grid: np.ndarray,
                             ev_point: float = None, la_point: float = None):
    xx, yy = np.meshgrid(ev_grid, la_grid)
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    surf = go.Surface(x=xx, y=yy, z=zz, opacity=0.6, showscale=True)
    fig = go.Figure(data=[surf])

    if ev_point is not None and la_point is not None:
        prob_pt = float(model.predict_proba([[ev_point, la_point]])[0, 1])
        fig.add_trace(go.Scatter3d(
            x=[ev_point], y=[la_point], z=[prob_pt],
            mode="markers+text",
            marker=dict(size=8, symbol="x"),
            text=[f"p={prob_pt:.3f}"],
            textposition="top center"
        ))
    fig.update_layout(scene_camera=dict(eye=_eye_from_elev_azim(30, -120, r=2.2)))
    fig.update_scenes(
        xaxis_title="打球速度 (mph)",
        yaxis_title="打球角度 (度)",
        zaxis_title="HR確率",
        aspectmode="cube"
    )
    fig.update_layout(title="HR確率 3D曲面（インタラクティブ）", margin=dict(l=0, r=0, t=30, b=0))
    return fig

def make_prob_curve_ev_png(
    estimator, ev_grid: np.ndarray, ev_point: float, threshold: float, out_path: str
) -> str:
    x = ev_grid.reshape(-1, 1)
    probs = estimator.predict_proba(x)[:, 1]
    p_at_ev = float(estimator.predict_proba(np.array([[ev_point]]))[:,1])

    fig, ax = plt.subplots(figsize=(6,4.2))
    ax.plot(ev_grid, probs)
    ax.axvline(ev_point, linestyle='--')
    ax.axhline(threshold, linestyle=':')
    ax.scatter([ev_point], [p_at_ev], s=60)
    ax.set_xlabel("打球速度 (mph)")
    ax.set_ylabel("HR確率")
    ax.set_title("LogReg（EVのみ）: 確率曲線")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path

def make_prob_curve_compare_png(
    estimator_ev_only, estimator_2d, la_fixed: float,
    ev_min: float, ev_max: float, ev_point: float, threshold: float, out_path: str,
    n_steps: int = 400, title: str = None
) -> str:
    ev_axis = np.linspace(ev_min, ev_max, n_steps)
    X_ev = ev_axis.reshape(-1, 1)
    X_2d = np.column_stack([ev_axis, np.full_like(ev_axis, la_fixed)])

    probs_ev = estimator_ev_only.predict_proba(X_ev)[:, 1]
    probs_2d = estimator_2d.predict_proba(X_2d)[:, 1]
    prob_ev_point = float(estimator_ev_only.predict_proba([[ev_point]])[0, 1])
    prob_2d_point = float(estimator_2d.predict_proba([[ev_point, la_fixed]])[0, 1])

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.plot(ev_axis, probs_ev, linewidth=2, label="EVのみ LogReg")
    ax.plot(ev_axis, probs_2d, linewidth=2, linestyle="--", label="2D LogReg (LA固定)")
    ax.axvline(ev_point, linestyle="--", linewidth=1.5)
    ax.axhline(threshold, linestyle=":", linewidth=1.5)
    ax.scatter([ev_point], [prob_ev_point], s=40)
    ax.scatter([ev_point], [prob_2d_point], s=40, marker="x")
    ax.set_xlabel("EV (mph)")
    ax.set_ylabel("HR 確率")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"モデル比較：EVのみ vs 2D（LA={la_fixed:.1f}°固定）")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ============================
# PDF生成
# ============================
def build_pdf(pdf_path: str,
              table_rows: List[List[str]],
              map_paths: Dict[str, str],
              surf_paths: Dict[str, str],
              curve_ev_only_path: str,
              compare_curve_path: str,
              ev: float, la: float, thr_mode: str,
              have_noto: Tuple[str, str]) -> str:
    reg_path, bold_path = have_noto

    styles = getSampleStyleSheet()
    if reg_path and bold_path:
        styles.add(ParagraphStyle(name="HeadingJP", fontName="NotoSansJP-Bold", fontSize=14, leading=18, spaceAfter=10))
        styles.add(ParagraphStyle(name="NormalJP", fontName="NotoSansJP-Regular", fontSize=11, leading=14))
        font_hdr = "NotoSansJP-Bold"
        font_norm = "NotoSansJP-Regular"
    else:
        styles.add(ParagraphStyle(name="HeadingJP", fontSize=14, leading=18, spaceAfter=10))
        styles.add(ParagraphStyle(name="NormalJP", fontSize=11, leading=14))
        font_hdr = "Helvetica-Bold"
        font_norm = "Helvetica"

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    story = []

    story.append(Paragraph("MLB 2024 HR確率予測（可視化＋PDF出力）", styles["HeadingJP"]))
    story.append(Paragraph("※ 本モデルは EV と LA のみを使用。球場・環境要因は未考慮。", styles["NormalJP"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"入力条件：EV = {ev:.1f} mph, LA = {la:.1f}°, しきい値 = {thr_mode}", styles["NormalJP"]))
    story.append(Spacer(1, 10))

    tbl = Table([["モデル","HR確率","使用しきい値","判定"]] + table_rows, colWidths=[140, 100, 100, 120])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.grey),
        ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("GRID",(0,0),(-1,-1),0.5,colors.black),
        ("FONTNAME",(0,0),(-1,0), font_hdr),
        ("FONTNAME",(0,1),(-1,-1), font_norm),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 12))

    story.append(Paragraph("確率マップ＆決定境界（各モデル）", styles["HeadingJP"]))
    for k in ["LogisticRegression","SVM_RBF","RandomForest"]:
        story.append(RLImage(map_paths[k], width=460, height=460*0.75))
        story.append(Paragraph(k, styles["NormalJP"]))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    story.append(Paragraph("HR確率 3D曲面（各モデル）", styles["HeadingJP"]))
    for k in ["LogisticRegression","SVM_RBF","RandomForest"]:
        story.append(RLImage(surf_paths[k], width=460, height=460*0.75))
        story.append(Paragraph(k, styles["NormalJP"]))
        story.append(Spacer(1, 6))

    story.append(PageBreak())

    if os.path.exists(curve_ev_only_path):
        story.append(Paragraph("EVのみロジスティック回帰：HR確率曲線（横軸=EV）", styles["HeadingJP"]))
        story.append(RLImage(curve_ev_only_path, width=460, height=460*0.65))
        story.append(Spacer(1, 10))

    if os.path.exists(compare_curve_path):
        story.append(Paragraph("モデル比較：EVのみ vs 2D（LA固定）", styles["HeadingJP"]))
        story.append(RLImage(compare_curve_path, width=460, height=460*0.65))
        story.append(Spacer(1, 10))

    doc.build(story)
    return pdf_path


# ============================
# アプリ本体
# ============================
def main():
    have_noto = setup_fonts()

    # データ＆モデル
    try:
        X, y = load_dataset(HR_CSV, NONHR_CSV)
    except Exception as e:
        with gr.Blocks() as demo:
            gr.Markdown("### データ読み込みエラー")
            gr.Markdown(f"`{e}`")
        demo.launch()
        return

    bundles = build_models(X, y)

    # グリッド範囲（少し余白）
    ev_min, ev_max = float(X[:,0].min()) - 2.0, float(X[:,0].max()) + 2.0
    la_min, la_max = float(X[:,1].min()) - 5.0, float(X[:,1].max()) + 5.0
    ev_grid = np.linspace(ev_min, ev_max, 160)
    la_grid = np.linspace(max(-10, la_min), min(80, la_max), 160)

    def predict_and_plot(ev: float, la: float, thr_mode: str):
        # 結果表
        x_full = np.array([[ev, la]])
        rows = []
        for name, bundle in bundles.items():
            x_sub = x_full[:, list(bundle.feature_indices)]
            prob = float(bundle.estimator.predict_proba(x_sub)[0,1])
            thr = bundle.best_threshold if thr_mode == "最適（Youden）" else 0.5
            label = "HR" if prob >= thr else "その他ヒット"
            disp_name = {
                "LogisticRegression": "LogisticRegression",
                "SVM_RBF": "SVM_RBF",
                "RandomForest": "RandomForest",
                "LogReg_EV_only": "LogReg（EVのみ）"
            }.get(name, name)
            rows.append([disp_name, f"{prob:.3f}", f"{thr:.3f}", label])
        df = pd.DataFrame(rows, columns=["モデル","HR確率","使用しきい値","判定"])

        # 図ファイルを一時保存
        tmpdir = tempfile.mkdtemp(prefix="hr_maps_")
        map_paths, surf_paths = {}, {}

        # 2D/3D 静止図（2D特徴モデルのみ）
        for key_for_plot in ["LogisticRegression","SVM_RBF","RandomForest"]:
            bundle = bundles[key_for_plot]
            p_map = os.path.join(tmpdir, f"map_{bundle.name}.png")
            p_surf = os.path.join(tmpdir, f"surf_{bundle.name}.png")
            make_prob_map_png(bundle.estimator, ev_grid, la_grid, ev, la, p_map)
            make_prob_surface_png(bundle.estimator, ev_grid, la_grid, ev, la, p_surf)
            map_paths[key_for_plot] = p_map
            surf_paths[key_for_plot] = p_surf

        # インタラクティブ3D（Plotly Figureをそのまま返す）
        fig_log = make_prob_surface_plotly(bundles['LogisticRegression'].estimator, ev_grid, la_grid, ev, la)
        fig_svm = make_prob_surface_plotly(bundles['SVM_RBF'].estimator,       ev_grid, la_grid, ev, la)
        fig_rf  = make_prob_surface_plotly(bundles['RandomForest'].estimator,   ev_grid, la_grid, ev, la)

        # EVのみ 1D 曲線
        thr_ev = bundles["LogReg_EV_only"].best_threshold if thr_mode == "最適（Youden）" else 0.5
        curve_path = os.path.join(tmpdir, "curve_logreg_ev_only.png")
        make_prob_curve_ev_png(bundles["LogReg_EV_only"].estimator, ev_grid, ev, thr_ev, curve_path)

        # モデル比較曲線
        thr_2d = bundles["LogisticRegression"].best_threshold if thr_mode == "最適（Youden）" else 0.5
        compare_curve_path = os.path.join(tmpdir, "curve_compare_ev_vs_2d.png")
        make_prob_curve_compare_png(
            estimator_ev_only=bundles["LogReg_EV_only"].estimator,
            estimator_2d=bundles["LogisticRegression"].estimator,
            la_fixed=la,
            ev_min=ev_min, ev_max=ev_max,
            ev_point=ev, threshold=thr_2d,
            out_path=compare_curve_path,
            title=f"モデル比較：EVのみ vs 2D（LA={la:.1f}°固定）"
        )

        # 返却（outputs と完全一致：17個）
        return (
            df,                                        # 1  out_table
            map_paths["LogisticRegression"],           # 2  img_logreg
            map_paths["SVM_RBF"],                      # 3  img_svm
            map_paths["RandomForest"],                 # 4  img_rf
            surf_paths["LogisticRegression"],          # 5  surf_logreg
            surf_paths["SVM_RBF"],                     # 6  surf_svm
            surf_paths["RandomForest"],                # 7  surf_rf
            rows,                                      # 8  state_table_rows
            map_paths,                                 # 9  state_map_paths
            surf_paths,                                # 10 state_surf_paths
            fig_log,                                   # 11 plot_log  (gr.Plot)
            fig_svm,                                   # 12 plot_svm  (gr.Plot)
            fig_rf,                                    # 13 plot_rf   (gr.Plot)
            curve_path,                                # 14 curve_ev_only (Image)
            compare_curve_path,                        # 15 curve_compare (Image)
            curve_path,                                # 16 state_curve_path
            compare_curve_path                         # 17 state_compare_curve_path
        )

    def batch_predict(file_obj, thr_mode: str):
        # 期待する列: launch_speed, launch_angle
        try:
            df_in = pd.read_csv(file_obj.name)
        except Exception as e:
            return None, f"CSV読み込みエラー: {e}"

        if not {"launch_speed", "launch_angle"}.issubset(df_in.columns):
            return None, "CSVに 'launch_speed','launch_angle' 列が必要です。"

        # 使うモデル（ここでは 2D ロジスティック回帰を代表に）
        mdl = bundles["LogisticRegression"].estimator
        thr = bundles["LogisticRegression"].best_threshold if thr_mode == "最適（Youden）" else 0.5

        Xb = df_in[["launch_speed", "launch_angle"]].to_numpy()
        probs = mdl.predict_proba(Xb)[:, 1]
        labels = np.where(probs >= thr, "HR", "その他ヒット")

        df_out = df_in.copy()
        df_out["prob_LogReg2D"] = probs
        df_out["label_at_thr"] = labels
        df_out["threshold_used"] = thr

        # 保存（Spaceの一時領域）
        out_path = os.path.join(tempfile.gettempdir(), f"batch_result_{int(time.time())}.csv")
        df_out.to_csv(out_path, index=False, encoding="utf-8")

        return out_path, f"OK: {len(df_out)} 行を処理しました。"

    
    def make_pdf(ev: float, la: float, thr_mode: str, table_rows, map_paths, surf_paths, curve_path, compare_curve_path):
        out_pdf = os.path.join(tempfile.gettempdir(), f"HR_Predict_Report_{int(time.time())}.pdf")
        pdf_path = build_pdf(out_pdf, table_rows, map_paths, surf_paths, curve_path, compare_curve_path, ev, la, thr_mode, have_noto)
        return pdf_path

    # ===== Gradio UI =====
    with gr.Blocks(title="MLB 2024 HR確率予測（可視化＋PDF出力）") as demo:
        gr.Markdown("## MLB 2024 HR確率予測（可視化＋PDF出力）")
        gr.Markdown("※ 本モデルは EV と LA のみを使用。球場・環境要因は未考慮。")

        with gr.Row():
            ev = gr.Slider(60, 120, value=95, step=0.1, label="打球速度 (mph)")
            la = gr.Slider(-10, 60, value=15, step=0.5, label="打球角度 (度)")
        thr_mode = gr.Radio(choices=["最適（Youden）","0.5固定"], value="最適（Youden）", label="しきい値の選択")

        btn_predict = gr.Button("予測する", variant="primary")
        out_table = gr.Dataframe(headers=["モデル","HR確率","使用しきい値","判定"],
                                 datatype=["str","str","str","str"], label="結果", wrap=True)

        gr.Markdown("### 確率マップ／決定境界（各モデル）")
        with gr.Row():
            img_logreg = gr.Image(type="filepath", label="Logistic Regression")
            img_svm = gr.Image(type="filepath", label="SVM (RBF)")
            img_rf = gr.Image(type="filepath", label="Random Forest")

        gr.Markdown("### HR確率 3D曲面（各モデル）")
        with gr.Row():
            surf_logreg = gr.Image(type="filepath", label="Logistic Regression")
            surf_svm = gr.Image(type="filepath", label="SVM (RBF)")
            surf_rf  = gr.Image(type="filepath", label="Random Forest")

        gr.Markdown("### EVのみロジスティック回帰：HR確率曲線（横軸=EV）")
        curve_ev_only = gr.Image(type="filepath", label="LogReg（EVのみ）確率曲線")

        gr.Markdown("### モデル比較：LogReg(EVのみ) vs LogReg(2D)（LA固定）")
        curve_compare = gr.Image(type="filepath", label="比較曲線（EV vs 2D）")

        gr.Markdown("### インタラクティブ 3D（マウスで回転・拡大）")
        with gr.Row():
            plot_log = gr.Plot(label="Logistic Regression (3D Interactive)")
            plot_svm = gr.Plot(label="SVM (RBF) (3D Interactive)")
            plot_rf  = gr.Plot(label="Random Forest (3D Interactive)")

        # PDF生成用の状態
        state_table_rows = gr.State()
        state_map_paths = gr.State()
        state_surf_paths = gr.State()
        state_curve_path = gr.State()
        state_compare_curve_path = gr.State()

        btn_predict.click(
            predict_and_plot,
            inputs=[ev, la, thr_mode],
            outputs=[
                out_table,                 # 1  結果表
                img_logreg, img_svm, img_rf,            # 2-4  2Dマップ
                surf_logreg, surf_svm, surf_rf,         # 5-7  3D静止画
                state_table_rows, state_map_paths, state_surf_paths,  # 8-10 States
                plot_log, plot_svm, plot_rf,            # 11-13 インタラクティブ3D（Plotly Figure）
                curve_ev_only,                          # 14  1D曲線 画像
                curve_compare,                          # 15  比較曲線 画像
                state_curve_path, state_compare_curve_path  # 16-17 States
            ]
        )

        gr.Markdown("---")
        btn_pdf = gr.Button("PDFを生成")
        out_pdf = gr.File(label="生成されたPDF")

        btn_pdf.click(
            make_pdf,
            inputs=[ev, la, thr_mode, state_table_rows, state_map_paths, state_surf_paths, state_curve_path, state_compare_curve_path],
            outputs=[out_pdf]
        )
        gr.Markdown("### バッチ予測（CSV一括）")
        with gr.Row():
            csv_in = gr.File(label="入力CSV（列: launch_speed, launch_angle）", file_types=[".csv"])
        with gr.Row():
            btn_batch = gr.Button("CSVを処理して結果をダウンロード")
        with gr.Row():
            csv_out = gr.File(label="出力CSV（確率と判定を追記）")
            batch_msg = gr.Markdown()
    
        btn_batch.click(
            batch_predict,
            inputs=[csv_in, thr_mode],
            outputs=[csv_out, batch_msg]
        )
    demo.launch()  # HF Spacesでは share=False でOK


if __name__ == "__main__":
    main()


