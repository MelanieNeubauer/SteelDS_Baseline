import os
import glob
import pandas as pd
import numpy as np

def summarize_results(dataset='_a2'):
    # Wir suchen tief in allen Unterordnern von "results" nach unseren End-CVSs
    csv_files = glob.glob(f"results{dataset}/**/*_evaluation_metrics_5_seeds.csv", recursive=True)
    
    # Falls die Ordnerstruktur flacher ist oder Dateien anders heißen, greifen wir zur Sicherheit alles ab was passt
    if not csv_files:
        csv_files = glob.glob(f"results{dataset}/**/*.csv", recursive=True)
        # Filtere nur die "zusammenfassende" Evaluierungs-CSVs heraus
        csv_files = [f for f in csv_files if "evaluation_metrics_5_seeds.csv" in f]

    if not csv_files:
        print("Keine CSV-Dateien mit Evaluierungsmetriken im Ordner 'results' gefunden!")
        return

    all_data = []

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # Modellname aus dem übergeordneten Ordnernamen extrahieren 
            # (Da du meintest: "Ordner mit den jeweiligen Modellnamen")
            model_name = os.path.basename(os.path.dirname(file_path))
            
            # Wenn der Dateipfad z.B. "results/yolo11n-seg.yaml_evaluation..." ist, nehmen wir den Dateinamen
            if model_name.lower() in [f"results{dataset}", ".", ".."]:
                model_name = os.path.basename(file_path).split('_evaluation_metrics')[0]
                
            df['Model'] = model_name
            all_data.append(df)
            
        except Exception as e:
            print(f"Konnte Datei {file_path} nicht lesen. Fehler: {e}")

    if not all_data:
        return

    # Alle gelesenen DataFrames zu einer großen Tabelle zusammenfügen
    combined_df = pd.concat(all_data, ignore_index=True)

    # Identifiziere alle numerischen Spalten (die wir runden und durchschnittlich berechnen können)
    num_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'seed' in num_cols:
        num_cols.remove('seed') # Seed soll natürlich nicht genittelt werden
        
    # Entferne YOLO-spezifische Metriken, die Mask R-CNN nicht (ohne PR-Curve) berechnet,
    # damit die Tabelle für beide exakt übereinstimmt:
    cols_to_drop = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'fitness']
    num_cols = [c for c in num_cols if c not in cols_to_drop]
        
    # Gruppiere nach 'Model' und berechne Durchschnitt (mean) und Standardabweichung (std)
    agg_df = combined_df.groupby('Model')[num_cols].agg(['mean', 'std'])

    # Ab hier formatieren wir die Tabelle, damit sie richtig hübsch und lesbar wird.
    # Wir vereinen Mean und Std in ein Feld pro Metrik: "0.95 ± 0.02" für die Lesbarkeit als Option:
    
    # Flacher Header für den einfachen Daten-Export (z.B. "metrics/mAP50(B)_mean")
    export_df = agg_df.copy()
    export_df.columns = [f"{col[0]}_{col[1]}" for col in export_df.columns]
    export_df = export_df.reset_index()

    # Lesbarer Header für die Anzeige mit +/- Zeichen
    pretty_df = pd.DataFrame()
    pretty_df['Model'] = agg_df.index
    for metric in num_cols:
        # Runde auf 4 Nachkommastellen
        mean = agg_df[(metric, 'mean')].values
        std = agg_df[(metric, 'std')].values
        
        # Erstelle den String "Mean ± Std" und fange nans sowie -1.0 (Torchmetrics Error-Code) ab
        pretty_strs = []
        for m, s in zip(mean, std):
            if np.isnan(m) or m < -0.5:
                pretty_strs.append("N/A")
            else:
                s_val = 0.0 if np.isnan(s) else s
                pretty_strs.append(f"{m:.4f} ± {s_val:.4f}")
                
        pretty_df[metric] = pretty_strs

    # Speichern der Rohdaten-Tabelle (nützlich für weitere Skripte)
    raw_output = f"results{dataset}/All_Models_Aggregated_Raw.csv"
    export_df.to_csv(raw_output, index=False)
    
    # Speichern der aufhübschen Tabelle (perfekt zum Anschauen, für Paper/Excel)
    pretty_output = f"results{dataset}/All_Models_Averaged_Pretty.csv"
    pretty_df.to_csv(pretty_output, index=False)
    
    print(f"\\n--- Zusammenfassung erfolgreich! ---")
    print(f"Rohdaten gespeichert in: {raw_output}")
    print(f"Formatierte Übersicht gespeichert in: {pretty_output}\\n")
    
    print("Vorschau der Ergebnisse:")
    print(pretty_df.to_string(index=False))

if __name__ == "__main__":
    summarize_results()
