#!/usr/bin/env python3
"""P3 — Banc de mesure et accélération du pipeline EPI sur CPU.

Point de départ : ~3.5 FPS mesurés pour les trois modèles en cascade
(`yolov8n` personnes + `best.pt` + `best_gloves.pt`) à imgsz=480, dominés par
`best.pt` (YOLOv8m, 78.7 GFLOPs).

Le script mesure, sur les mêmes images et le même processus, quatre leviers :

1. **Résolution.** imgsz 640 / 512 / 480 / 416 — le gain était connu (~1.8x
   entre 640 et 480) mais l'impact sur la précision n'avait jamais été
   quantifié : c'est fait par `--valider-precision`, qui relance une validation
   à chaque résolution.
2. **ONNX + onnxruntime.** Même graphe, même précision numérique (fp32), un
   runtime différent : c'est le levier sans contrepartie, à tenter en premier.
3. **Composition de la cascade.** Coût de chaque modèle isolé, pour chiffrer
   ce que rapporte le retrait de `best_gloves.pt` (cf. P2 : son seul apport
   propre est la classe `safety_shoe`).
4. **Threads.** `torch.set_num_threads` : par défaut PyTorch prend tous les
   cœurs, ce qui n'est pas optimal quand plusieurs modèles tournent en parallèle
   dans le pipeline unifié.

Toutes les mesures sont faites après un préchauffage, et rapportées en médiane
(pas en moyenne) pour ne pas être polluées par la première inférence.
"""

from pathlib import Path
import argparse
import json
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/v2_results"
MODELES = {
    "personnes_yolov8n": ROOT / "ppe_detection/models/yolov8n.pt",
    "epi_best":          ROOT / "ppe_detection/models/best.pt",
    "epi_gloves":        ROOT / "ppe_detection/models/best_gloves.pt",
}


def charger_images(n: int):
    import cv2
    src = ROOT / "ppe_detection/data/extracted/ppe_dataset/val/images"
    chemins = sorted(src.iterdir())[:n]
    return [cv2.imread(str(p)) for p in chemins]


def mesurer(predict, images, repetitions=1):
    """Médiane du temps par image, préchauffage exclu."""
    predict(images[0])
    temps = []
    for _ in range(repetitions):
        for im in images:
            t0 = time.perf_counter()
            predict(im)
            temps.append(time.perf_counter() - t0)
    return {
        "ms_median": round(1000 * statistics.median(temps), 2),
        "ms_p90": round(1000 * sorted(temps)[int(0.9 * len(temps)) - 1], 2),
        "fps": round(1.0 / statistics.median(temps), 2),
        "n": len(temps),
    }


def bench_pytorch(images, tailles, threads):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(threads)

    res = {}
    modeles = {k: YOLO(str(v)) for k, v in MODELES.items()}
    for imgsz in tailles:
        par_modele = {}
        for nom, m in modeles.items():
            par_modele[nom] = mesurer(
                lambda im, m=m, s=imgsz: m.predict(im, imgsz=s, conf=0.25, verbose=False), images)
            print(f"  pytorch imgsz={imgsz:4} {nom:20} {par_modele[nom]['ms_median']:8.1f} ms")

        # Cascades : somme des coûts médians, la boucle vidéo étant séquentielle
        # par frame pour ces trois modèles.
        c3 = sum(par_modele[k]["ms_median"] for k in MODELES)
        c2 = par_modele["personnes_yolov8n"]["ms_median"] + par_modele["epi_best"]["ms_median"]
        par_modele["cascade_3_modeles"] = {"ms_median": round(c3, 2), "fps": round(1000 / c3, 2)}
        par_modele["cascade_2_modeles_sans_gloves"] = {"ms_median": round(c2, 2), "fps": round(1000 / c2, 2)}
        print(f"  -> cascade 3 modeles : {c3:.1f} ms ({1000 / c3:.2f} FPS)"
              f" | sans best_gloves : {c2:.1f} ms ({1000 / c2:.2f} FPS)")
        res[str(imgsz)] = par_modele
    return res


def exporter_onnx(tailles):
    from ultralytics import YOLO
    exports = {}
    for nom, poids in MODELES.items():
        for imgsz in tailles:
            cible = poids.with_name(f"{poids.stem}_{imgsz}.onnx")
            if not cible.exists():
                print(f"  export {nom} imgsz={imgsz} ...")
                produit = Path(YOLO(str(poids)).export(format="onnx", imgsz=imgsz,
                                                       dynamic=False, simplify=True, device="cpu"))
                produit.rename(cible)
            exports[(nom, imgsz)] = cible
    return exports


def bench_onnx(images, tailles, threads, exports):
    from ultralytics import YOLO
    res = {}
    for imgsz in tailles:
        par_modele = {}
        for nom in MODELES:
            m = YOLO(str(exports[(nom, imgsz)]), task="detect")
            par_modele[nom] = mesurer(
                lambda im, m=m, s=imgsz: m.predict(im, imgsz=s, conf=0.25, verbose=False), images)
            print(f"  onnx    imgsz={imgsz:4} {nom:20} {par_modele[nom]['ms_median']:8.1f} ms")
        c3 = sum(par_modele[k]["ms_median"] for k in MODELES)
        c2 = par_modele["personnes_yolov8n"]["ms_median"] + par_modele["epi_best"]["ms_median"]
        par_modele["cascade_3_modeles"] = {"ms_median": round(c3, 2), "fps": round(1000 / c3, 2)}
        par_modele["cascade_2_modeles_sans_gloves"] = {"ms_median": round(c2, 2), "fps": round(1000 / c2, 2)}
        print(f"  -> cascade 3 modeles : {c3:.1f} ms ({1000 / c3:.2f} FPS)"
              f" | sans best_gloves : {c2:.1f} ms ({1000 / c2:.2f} FPS)")
        res[str(imgsz)] = par_modele
    return res


def valider_precision(tailles):
    """Impact réel de la résolution sur la précision de `best.pt`.

    Mesuré sur le sous-ensemble gilet propre (779 images) : assez petit pour
    tenir sur CPU à quatre résolutions, et surtout exempt du mélange de lots qui
    fausse les AP sur le val complet.
    """
    from ultralytics import YOLO
    data = ROOT / "ppe_detection/data/extracted/ppe_vest_clean_14c/data.yaml"
    if not data.exists():
        print("  (sous-ensemble propre absent : lancer p1_build_vest_dataset.py)")
        return {}
    model = YOLO(str(MODELES["epi_best"]))
    res = {}
    for imgsz in tailles:
        m = model.val(data=str(data), split="val", imgsz=imgsz, device="cpu", batch=4,
                      project=str(OUT / "runs"), name=f"p3_prec_{imgsz}", exist_ok=True, plots=False)
        noms = model.names
        pc = {noms[int(ci)]: round(float(m.box.ap50[i]), 4) for i, ci in enumerate(m.ap_class_index)}
        res[str(imgsz)] = {"mAP50": round(float(m.box.map50), 4),
                           "Safety Vest": pc.get("Safety Vest"),
                           "NO-Safety Vest": pc.get("NO-Safety Vest")}
        print(f"  precision imgsz={imgsz}: mAP50={res[str(imgsz)]['mAP50']}  {pc.get('Safety Vest')} / {pc.get('NO-Safety Vest')}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=20)
    ap.add_argument("--tailles", default="640,512,480,416")
    ap.add_argument("--threads", type=int, default=0, help="0 = tous les coeurs")
    ap.add_argument("--sans-onnx", action="store_true")
    ap.add_argument("--valider-precision", action="store_true")
    args = ap.parse_args()

    import os
    tailles = [int(s) for s in args.tailles.split(",")]
    threads = args.threads or os.cpu_count()

    images = charger_images(args.images)
    print(f"{len(images)} images, {threads} threads torch\n")

    resultats = {"threads": threads, "n_images": len(images)}

    print("== PyTorch ==")
    resultats["pytorch"] = bench_pytorch(images, tailles, threads)

    if not args.sans_onnx:
        print("\n== Export ONNX ==")
        exports = exporter_onnx(tailles)
        print("\n== onnxruntime ==")
        resultats["onnx"] = bench_onnx(images, tailles, threads, exports)

    if args.valider_precision:
        print("\n== Precision vs resolution (best.pt, sous-ensemble propre) ==")
        resultats["precision_vs_resolution"] = valider_precision(tailles)

    dest = OUT / "p3_benchmark.json"
    dest.write_text(json.dumps(resultats, indent=2, ensure_ascii=False))
    print(f"\n-> {dest}")

    if "onnx" in resultats:
        print("\n== Synthese gain ONNX (cascade 3 modeles) ==")
        for t in tailles:
            pt = resultats["pytorch"][str(t)]["cascade_3_modeles"]["ms_median"]
            ox = resultats["onnx"][str(t)]["cascade_3_modeles"]["ms_median"]
            print(f"  imgsz={t:4} : {pt:7.1f} ms -> {ox:7.1f} ms  (x{pt / ox:.2f})")


if __name__ == "__main__":
    main()
