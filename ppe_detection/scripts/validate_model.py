"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         VALIDATION COMPLÈTE DES MODÈLES PPE — STATISTIQUES & MÉTRIQUES     ║
║  Génère : mAP, Précision, Rappel, F1, Matrice de confusion, rapport HTML    ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:
  python validate_model.py --model best.pt --data /chemin/dataset --split val
  python validate_model.py --model best.pt --data /chemin/dataset --split test
  python validate_model.py --all   # valide best.pt ET best_gloves.pt ensemble

STRUCTURE DATASET ATTENDUE (format YOLO):
  dataset/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── labels/
      ├── train/
      ├── val/
      └── test/

"""

import argparse, json, time, sys
from pathlib import Path
import numpy as np

# ─── Dépendances optionnelles (installées si nécessaires) ─────────────────────
def check_deps():
    missing = []
    try: import ultralytics
    except ImportError: missing.append("ultralytics")
    try: import yaml
    except ImportError: missing.append("pyyaml")
    try: import matplotlib
    except ImportError: missing.append("matplotlib")
    try: import seaborn
    except ImportError: missing.append("seaborn")
    if missing:
        print(f"Installation des dépendances manquantes: {missing}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing)

check_deps()

from ultralytics import YOLO
import yaml
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("validation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Couleurs pour les graphiques
COLORS = {
    "primary":   "#3B82F6",
    "success":   "#22C55E",
    "warning":   "#F59E0B",
    "danger":    "#EF4444",
    "neutral":   "#6B7280",
    "purple":    "#8B5CF6",
    "teal":      "#14B8A6",
}

# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════

def validate_model(model_path: str, data_yaml: str, split: str = "val",
                   conf: float = 0.25, iou_thresh: float = 0.50,
                   imgsz: int = 640) -> dict:
    """
    Lance la validation YOLO et retourne les métriques complètes.
    """
    print(f"\n{'═'*60}")
    print(f"  Validation : {model_path}")
    print(f"  Dataset    : {data_yaml}")
    print(f"  Split      : {split} | conf={conf} | IoU={iou_thresh}")
    print(f"{'═'*60}\n")

    model = YOLO(model_path)
    start = time.time()

    results = model.val(
        data       = data_yaml,
        split      = split,
        conf       = conf,
        iou        = iou_thresh,
        imgsz      = imgsz,
        save_json  = True,
        plots      = True,
        verbose    = False,
    )

    elapsed = time.time() - start

    # ── Extraction des métriques ──────────────────────────────────────────────
    metrics = results.box  # Ultralytics MetricsBox

    class_names = model.names
    n_classes   = len(class_names)

    # mAP global
    map50    = float(metrics.map50)      if hasattr(metrics, "map50")   else 0.0
    map5095  = float(metrics.map)        if hasattr(metrics, "map")      else 0.0

    # Précision / Rappel / F1 par classe
    precision_per_class = metrics.p   if hasattr(metrics, "p")  else [0]*n_classes
    recall_per_class    = metrics.r   if hasattr(metrics, "r")  else [0]*n_classes
    f1_per_class        = metrics.f1  if hasattr(metrics, "f1") else [0]*n_classes
    ap50_per_class      = metrics.ap50 if hasattr(metrics, "ap50") else [0]*n_classes

    precision_per_class = list(precision_per_class)
    recall_per_class    = list(recall_per_class)
    f1_per_class        = list(f1_per_class)
    ap50_per_class      = list(ap50_per_class)

    # Moyennes globales
    mean_precision = float(np.mean(precision_per_class)) if precision_per_class else 0.0
    mean_recall    = float(np.mean(recall_per_class))    if recall_per_class    else 0.0
    mean_f1        = float(np.mean(f1_per_class))        if f1_per_class        else 0.0

    # Vitesse (ms/image)
    speed          = results.speed if hasattr(results, "speed") else {}
    inference_ms   = speed.get("inference", 0.0)

    stats = {
        "model":              model_path,
        "dataset":            data_yaml,
        "split":              split,
        "timestamp":          TIMESTAMP,
        "elapsed_sec":        elapsed,
        "inference_ms":       inference_ms,
        "n_classes":          n_classes,
        "class_names":        [class_names[i] for i in range(n_classes)],
        "map50":              map50,
        "map50_95":           map5095,
        "mean_precision":     mean_precision,
        "mean_recall":        mean_recall,
        "mean_f1":            mean_f1,
        "precision_per_class": precision_per_class,
        "recall_per_class":    recall_per_class,
        "f1_per_class":        f1_per_class,
        "ap50_per_class":      ap50_per_class,
        "confusion_matrix":   None,   # rempli séparément si disponible
    }

    # ── Matrice de confusion ──────────────────────────────────────────────────
    if hasattr(results, "confusion_matrix") and results.confusion_matrix is not None:
        cm = results.confusion_matrix
        if hasattr(cm, "matrix"):
            stats["confusion_matrix"] = cm.matrix.tolist()

    print(f"\n  mAP@50       : {map50:.4f}  ({map50*100:.1f}%)")
    print(f"  mAP@50:95    : {map5095:.4f}  ({map5095*100:.1f}%)")
    print(f"  Précision    : {mean_precision:.4f}  ({mean_precision*100:.1f}%)")
    print(f"  Rappel       : {mean_recall:.4f}  ({mean_recall*100:.1f}%)")
    print(f"  F1-Score     : {mean_f1:.4f}  ({mean_f1*100:.1f}%)")
    print(f"  Vitesse      : {inference_ms:.1f} ms/image")
    print(f"  Durée totale : {elapsed:.1f}s\n")

    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  GÉNÉRATION DES GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════

def plot_per_class_metrics(stats: dict, save_dir: Path) -> Path:
    """Graphique barres groupées par classe (Précision / Rappel / F1 / AP50)."""
    names     = stats["class_names"]
    prec      = stats["precision_per_class"]
    rec       = stats["recall_per_class"]
    f1        = stats["f1_per_class"]
    ap50      = stats["ap50_per_class"]

    n     = len(names)
    x     = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(12, n * 1.2), 7))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    bars_prec = ax.bar(x - 1.5*width, prec, width, label="Précision", color=COLORS["primary"],  alpha=0.9)
    bars_rec  = ax.bar(x - 0.5*width, rec,  width, label="Rappel",    color=COLORS["success"],  alpha=0.9)
    bars_f1   = ax.bar(x + 0.5*width, f1,   width, label="F1-Score",  color=COLORS["purple"],   alpha=0.9)
    bars_ap   = ax.bar(x + 1.5*width, ap50, width, label="AP@50",     color=COLORS["warning"],  alpha=0.9)

    ax.set_xlabel("Classes", color="white", fontsize=12)
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.set_title(f"Métriques par classe — {Path(stats['model']).stem}",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", color="white", fontsize=9)
    ax.set_ylim(0, 1.10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#334155")
    ax.yaxis.grid(True, color="#334155", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    leg = ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="white")
    for bar_group in [bars_prec, bars_rec, bars_f1, bars_ap]:
        for bar in bar_group:
            h = bar.get_height()
            if h > 0.05:
                ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color="white")

    plt.tight_layout()
    path = save_dir / f"per_class_metrics_{TIMESTAMP}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Graphique] Par classe : {path}")
    return path


def plot_confusion_matrix(stats: dict, save_dir: Path) -> Path | None:
    """Heatmap de la matrice de confusion."""
    cm_data = stats.get("confusion_matrix")
    if cm_data is None:
        print("  [Info] Matrice de confusion non disponible.")
        return None

    cm    = np.array(cm_data)
    names = stats["class_names"] + ["Background"]
    if cm.shape[0] == len(names) - 1:
        names = stats["class_names"]

    # Normalisation par ligne (vrai positif ratio)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_norm = cm / row_sum

    fig_size = max(10, len(names) * 0.7)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=names, yticklabels=names,
                ax=ax, linewidths=0.5, linecolor="#334155",
                cbar_kws={"shrink": 0.8})

    ax.set_title(f"Matrice de confusion (normalisée) — {Path(stats['model']).stem}",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Prédit", color="white", fontsize=11)
    ax.set_ylabel("Réel",   color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    path = save_dir / f"confusion_matrix_{TIMESTAMP}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Graphique] Matrice de confusion : {path}")
    return path


def plot_pr_curve_summary(stats: dict, save_dir: Path) -> Path:
    """Résumé P/R/F1 sous forme de jauge circulaire."""
    vals   = [stats["map50"], stats["mean_precision"],
              stats["mean_recall"], stats["mean_f1"]]
    labels = ["mAP@50", "Précision", "Rappel", "F1-Score"]
    colors = [COLORS["warning"], COLORS["primary"],
              COLORS["success"], COLORS["purple"]]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5),
                             subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor("#0F172A")

    for ax, val, label, color in zip(axes, vals, labels, colors):
        ax.set_facecolor("#1E293B")
        theta = val * 2 * np.pi
        bars  = ax.bar(theta/2, val, width=theta, bottom=0.0,
                       color=color, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 2 * np.pi)
        ax.axis("off")
        ax.text(0, 0, f"{val*100:.1f}%", ha="center", va="center",
                fontsize=18, fontweight="bold", color="white",
                transform=ax.transData)
        ax.text(0, -1.4, label, ha="center", va="center",
                fontsize=11, color=color, transform=ax.transData,
                fontweight="bold")

    fig.suptitle(f"Résumé global — {Path(stats['model']).stem}",
                 color="white", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = save_dir / f"summary_gauges_{TIMESTAMP}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Graphique] Résumé gauges : {path}")
    return path


def plot_dual_model_comparison(stats1: dict, stats2: dict, save_dir: Path) -> Path:
    """Comparaison barres horizontales entre 2 modèles."""
    metrics = ["mAP@50", "mAP@50:95", "Précision", "Rappel", "F1-Score"]
    vals1   = [stats1["map50"], stats1["map50_95"], stats1["mean_precision"],
               stats1["mean_recall"], stats1["mean_f1"]]
    vals2   = [stats2["map50"], stats2["map50_95"], stats2["mean_precision"],
               stats2["mean_recall"], stats2["mean_f1"]]
    name1   = Path(stats1["model"]).stem
    name2   = Path(stats2["model"]).stem

    x     = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    b1 = ax.barh(x + width/2, vals1, width, label=name1,
                 color=COLORS["primary"], alpha=0.9)
    b2 = ax.barh(x - width/2, vals2, width, label=name2,
                 color=COLORS["teal"], alpha=0.9)

    ax.set_yticks(x)
    ax.set_yticklabels(metrics, color="white", fontsize=11)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Score", color="white", fontsize=11)
    ax.set_title("Comparaison des deux modèles", color="white",
                 fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#334155")
    ax.xaxis.grid(True, color="#334155", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    leg = ax.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="white")

    for bar in b1:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", va="center", color="white", fontsize=9)
    for bar in b2:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2,
                f"{w:.3f}", va="center", color="white", fontsize=9)

    plt.tight_layout()
    path = save_dir / f"model_comparison_{TIMESTAMP}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [Graphique] Comparaison : {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  RAPPORT HTML
# ══════════════════════════════════════════════════════════════════════════════

def generate_html_report(stats_list: list[dict],
                         img_paths: list[str],
                         save_dir: Path) -> Path:
    """Génère un rapport HTML complet avec tableaux et graphiques intégrés."""

    import base64

    def img_to_b64(path: str | Path | None) -> str:
        if path is None:
            return ""
        path = Path(path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def quality_badge(val: float) -> str:
        if val >= 0.85:
            return f'<span class="badge badge-ok">Excellent {val*100:.1f}%</span>'
        elif val >= 0.70:
            return f'<span class="badge badge-warn">Bon {val*100:.1f}%</span>'
        elif val >= 0.50:
            return f'<span class="badge badge-caution">Moyen {val*100:.1f}%</span>'
        else:
            return f'<span class="badge badge-bad">Faible {val*100:.1f}%</span>'

    imgs_b64 = {Path(p).name if p else f"img{i}": img_to_b64(p)
                for i, p in enumerate(img_paths) if p}

    # ── Table per-class pour chaque modèle ───────────────────────────────────
    tables_html = ""
    for stats in stats_list:
        model_name = Path(stats["model"]).stem
        names      = stats["class_names"]
        prec       = stats["precision_per_class"]
        rec        = stats["recall_per_class"]
        f1         = stats["f1_per_class"]
        ap50       = stats["ap50_per_class"]

        rows = ""
        for i, name in enumerate(names):
            p  = prec[i]  if i < len(prec)  else 0.0
            r  = rec[i]   if i < len(rec)   else 0.0
            f  = f1[i]    if i < len(f1)    else 0.0
            a  = ap50[i]  if i < len(ap50)  else 0.0
            row_class = "row-danger" if f < 0.50 else ("row-warn" if f < 0.70 else "")
            rows += f"""
            <tr class="{row_class}">
              <td><code>{name}</code></td>
              <td>{p:.3f}</td>
              <td>{r:.3f}</td>
              <td>{f:.3f}</td>
              <td>{a:.3f}</td>
              <td>{"▮" * int(a*10)}{"▯" * (10 - int(a*10))}</td>
            </tr>"""

        tables_html += f"""
        <section class="card">
          <h2>{model_name} — Métriques par classe</h2>
          <table>
            <thead>
              <tr>
                <th>Classe</th>
                <th>Précision</th>
                <th>Rappel</th>
                <th>F1-Score</th>
                <th>AP@50</th>
                <th>Barre AP</th>
              </tr>
            </thead>
            <tbody>
              {rows}
            </tbody>
          </table>
        </section>"""

    # ── Cartes de résumé ──────────────────────────────────────────────────────
    summary_cards = ""
    for stats in stats_list:
        model_name = Path(stats["model"]).stem
        summary_cards += f"""
        <div class="model-summary card">
          <h2>{model_name}</h2>
          <div class="kpi-grid">
            <div class="kpi">
              <div class="kpi-label">mAP@50</div>
              <div class="kpi-val">{stats['map50']*100:.1f}<span class="kpi-unit">%</span></div>
              {quality_badge(stats['map50'])}
            </div>
            <div class="kpi">
              <div class="kpi-label">mAP@50:95</div>
              <div class="kpi-val">{stats['map50_95']*100:.1f}<span class="kpi-unit">%</span></div>
              {quality_badge(stats['map50_95'])}
            </div>
            <div class="kpi">
              <div class="kpi-label">Précision</div>
              <div class="kpi-val">{stats['mean_precision']*100:.1f}<span class="kpi-unit">%</span></div>
              {quality_badge(stats['mean_precision'])}
            </div>
            <div class="kpi">
              <div class="kpi-label">Rappel</div>
              <div class="kpi-val">{stats['mean_recall']*100:.1f}<span class="kpi-unit">%</span></div>
              {quality_badge(stats['mean_recall'])}
            </div>
            <div class="kpi">
              <div class="kpi-label">F1-Score</div>
              <div class="kpi-val">{stats['mean_f1']*100:.1f}<span class="kpi-unit">%</span></div>
              {quality_badge(stats['mean_f1'])}
            </div>
            <div class="kpi">
              <div class="kpi-label">Vitesse</div>
              <div class="kpi-val">{stats['inference_ms']:.0f}<span class="kpi-unit">ms</span></div>
            </div>
          </div>
        </div>"""

    # ── Images ────────────────────────────────────────────────────────────────
    imgs_html = '<div class="img-grid">'
    for name, b64 in imgs_b64.items():
        if b64:
            imgs_html += f"""
            <figure>
              <img src="data:image/png;base64,{b64}" alt="{name}" />
              <figcaption>{name}</figcaption>
            </figure>"""
    imgs_html += "</div>"

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rapport de Validation PPE — {TIMESTAMP}</title>
<style>
  :root {{
    --bg:      #0F172A;
    --surface: #1E293B;
    --border:  #334155;
    --text:    #F1F5F9;
    --muted:   #94A3B8;
    --ok:      #22C55E;
    --warn:    #F59E0B;
    --bad:     #EF4444;
    --blue:    #3B82F6;
    --purple:  #8B5CF6;
    --teal:    #14B8A6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg);
          color: var(--text); padding: 2rem; line-height: 1.6; }}
  h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: .3rem; color: var(--blue); }}
  h2 {{ font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem; color: var(--text); }}
  .subtitle {{ color: var(--muted); font-size: .9rem; margin-bottom: 2rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
               gap: 1rem; margin-top: 1rem; }}
  .kpi {{ background: #0F172A; border: 1px solid var(--border); border-radius: 8px;
          padding: 1rem; text-align: center; }}
  .kpi-label {{ font-size: .75rem; color: var(--muted); text-transform: uppercase;
                letter-spacing: .05em; margin-bottom: .25rem; }}
  .kpi-val {{ font-size: 2rem; font-weight: 700; color: var(--text); }}
  .kpi-unit {{ font-size: 1rem; color: var(--muted); }}
  .badge {{ display: inline-block; padding: .2rem .6rem; border-radius: 999px;
            font-size: .72rem; font-weight: 600; margin-top: .4rem; }}
  .badge-ok      {{ background: #14532D; color: var(--ok); }}
  .badge-warn    {{ background: #451A03; color: var(--warn); }}
  .badge-caution {{ background: #1C1917; color: #FB923C; }}
  .badge-bad     {{ background: #450A0A; color: var(--bad); }}
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
  th {{ background: #0F172A; color: var(--muted); padding: .6rem .75rem;
        text-align: left; border-bottom: 1px solid var(--border);
        font-weight: 600; letter-spacing: .04em; font-size: .78rem; }}
  td {{ padding: .55rem .75rem; border-bottom: 1px solid var(--border); }}
  tr:last-child td {{ border-bottom: none; }}
  .row-danger td {{ color: #F87171; }}
  .row-warn   td {{ color: #FCD34D; }}
  code {{ background: #0F172A; border-radius: 4px; padding: .15rem .4rem;
          font-family: monospace; font-size: .82rem; color: var(--teal); }}
  .img-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px,1fr));
               gap: 1.5rem; }}
  figure {{ background: var(--surface); border: 1px solid var(--border);
            border-radius: 10px; overflow: hidden; }}
  figure img {{ width: 100%; display: block; }}
  figcaption {{ font-size: .78rem; color: var(--muted); padding: .5rem .75rem; }}
  .header-bar {{ display: flex; justify-content: space-between; align-items: flex-start;
                 margin-bottom: 2rem; flex-wrap: wrap; gap: 1rem; }}
  .models-list span {{ background: var(--surface); border: 1px solid var(--border);
                       border-radius: 6px; padding: .2rem .6rem; font-size: .82rem;
                       margin-left: .4rem; color: var(--purple); font-family: monospace; }}
  footer {{ margin-top: 3rem; text-align: center; color: var(--muted); font-size: .78rem; }}
</style>
</head>
<body>
<div class="header-bar">
  <div>
    <h1>Rapport de Validation — Modèles PPE</h1>
    <p class="subtitle">
      Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')} &nbsp;|&nbsp;
      Modèles :
      <span class="models-list">
        {"".join(f"<span>{Path(s['model']).name}</span>" for s in stats_list)}
      </span>
    </p>
  </div>
</div>

{summary_cards}

<section class="card">
  <h2>Visualisations</h2>
  {imgs_html}
</section>

{tables_html}

<footer>Système IA Sécurité PPE — Validation automatique — {TIMESTAMP}</footer>
</body>
</html>"""

    report_path = save_dir / f"rapport_validation_{TIMESTAMP}.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  [Rapport HTML] → {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
#  SAUVEGARDE JSON
# ══════════════════════════════════════════════════════════════════════════════

def save_json(stats: dict, save_dir: Path) -> Path:
    path = save_dir / f"stats_{Path(stats['model']).stem}_{TIMESTAMP}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  [JSON] Stats sauvegardées → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION SANS DATASET (sur images brutes)
# ══════════════════════════════════════════════════════════════════════════════

def quick_inference_test(model_path: str, images_dir: str,
                         conf: float = 0.25) -> dict:
    """
    Test rapide sur un dossier d'images SANS annotations.
    Produit des statistiques de détection (nb détections / classe).
    Utile quand on n'a pas encore de labels.
    """
    print(f"\n  [Quick Test] {model_path} sur {images_dir}")
    model  = YOLO(model_path)
    img_dir = Path(images_dir)
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]

    if not images:
        print("  Aucune image trouvée.")
        return {}

    counts: dict = {name: 0 for name in model.names.values()}
    total_dets = 0

    for img_path in images:
        import cv2
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        results = model.predict(frame, conf=conf, verbose=False)[0]
        for box in results.boxes:
            cls_name = model.names[int(box.cls)]
            counts[cls_name] += 1
            total_dets += 1

    print(f"  {len(images)} images, {total_dets} détections totales")
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        if cnt > 0:
            print(f"    {cls:<25} : {cnt:>5} détections")
    return counts


# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Validation complète des modèles PPE YOLO",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model",   default="best.pt",
                        help="Chemin vers le modèle .pt (default: best.pt)")
    parser.add_argument("--model2",  default=None,
                        help="Second modèle optionnel (ex: best_gloves.pt)")
    parser.add_argument("--data",    default=None,
                        help="Chemin vers le dataset.yaml YOLO")
    parser.add_argument("--split",   default="val",
                        choices=["val", "test", "train"],
                        help="Split à utiliser pour la validation (default: val)")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Seuil de confiance (default: 0.25)")
    parser.add_argument("--iou",     type=float, default=0.50,
                        help="Seuil IoU pour NMS (default: 0.50)")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="Taille image (default: 640)")
    parser.add_argument("--images",  default=None,
                        help="Dossier d'images pour test rapide sans labels")
    parser.add_argument("--all",     action="store_true",
                        help="Valide best.pt + best_gloves.pt ensemble")
    args = parser.parse_args()

    # ── Mode --all : valide les deux modèles PPE ──────────────────────────────
    if args.all:
        if args.data is None:
            print("ERREUR : --all nécessite --data chemin/dataset.yaml")
            sys.exit(1)
        s1 = validate_model("best.pt",        args.data, args.split, args.conf,
                            args.iou, args.imgsz)
        s2 = validate_model("best_gloves.pt", args.data, args.split, args.conf,
                            args.iou, args.imgsz)
        save_json(s1, OUTPUT_DIR)
        save_json(s2, OUTPUT_DIR)

        img_paths = [
            str(plot_per_class_metrics(s1, OUTPUT_DIR)),
            str(plot_per_class_metrics(s2, OUTPUT_DIR)),
            str(plot_confusion_matrix(s1, OUTPUT_DIR)),
            str(plot_confusion_matrix(s2, OUTPUT_DIR)),
            str(plot_pr_curve_summary(s1, OUTPUT_DIR)),
            str(plot_pr_curve_summary(s2, OUTPUT_DIR)),
            str(plot_dual_model_comparison(s1, s2, OUTPUT_DIR)),
        ]
        report = generate_html_report([s1, s2], img_paths, OUTPUT_DIR)
        print(f"\n{'═'*60}")
        print(f"  RAPPORT COMPLET → ouvrir dans votre navigateur:")
        print(f"  {report.resolve()}")
        print(f"{'═'*60}\n")
        return

    # ── Mode test rapide sur images brutes ────────────────────────────────────
    if args.images:
        quick_inference_test(args.model, args.images, args.conf)
        if args.model2:
            quick_inference_test(args.model2, args.images, args.conf)
        return

    # ── Mode validation avec dataset ──────────────────────────────────────────
    if args.data is None:
        print("ERREUR : vous devez fournir --data chemin/dataset.yaml OU --images dossier/")
        print("Exemple : python validate_model.py --model best.pt --data ppe_dataset/data.yaml")
        sys.exit(1)

    stats = validate_model(args.model, args.data, args.split,
                           args.conf, args.iou, args.imgsz)
    save_json(stats, OUTPUT_DIR)

    img_paths = [
        str(plot_per_class_metrics(stats, OUTPUT_DIR)),
        str(plot_confusion_matrix(stats, OUTPUT_DIR)),
        str(plot_pr_curve_summary(stats, OUTPUT_DIR)),
    ]

    all_stats = [stats]
    if args.model2:
        stats2 = validate_model(args.model2, args.data, args.split,
                                args.conf, args.iou, args.imgsz)
        save_json(stats2, OUTPUT_DIR)
        img_paths += [
            str(plot_per_class_metrics(stats2, OUTPUT_DIR)),
            str(plot_confusion_matrix(stats2, OUTPUT_DIR)),
            str(plot_pr_curve_summary(stats2, OUTPUT_DIR)),
            str(plot_dual_model_comparison(stats, stats2, OUTPUT_DIR)),
        ]
        all_stats.append(stats2)

    report = generate_html_report(all_stats, img_paths, OUTPUT_DIR)
    print(f"\n{'═'*60}")
    print(f"  RAPPORT COMPLET → ouvrir dans votre navigateur:")
    print(f"  {report.resolve()}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
