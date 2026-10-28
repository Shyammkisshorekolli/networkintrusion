import csv
import html
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
REPORT = OUTPUTS / "research_results_report.html"
# Published accuracy and weighted F1 values transcribed from the paper's result tables.
DATASETS = {
    "cic": ("CIC-UNSW-NB15", [
        ("decision_tree", 93.03, 93.06), ("random_forest", 93.71, 93.65),
        ("naive_bayes", 81.64, 83.41), ("knn", 92.32, 92.18),
        ("bagging", 93.41, 93.36), ("stacking", 93.69, 93.62)]),
    "hikari": ("HIKARI-2021", [
        ("decision_tree", 77.06, 75.64), ("random_forest", 74.98, 74.58),
        ("naive_bayes", 34.63, 22.53), ("knn", 74.71, 73.17),
        ("bagging", 75.50, 74.22), ("stacking", 79.00, 78.11)]),
    "unr": ("UNR-IDD", [
        ("decision_tree", 92.45, 92.42), ("random_forest", 93.97, 93.92),
        ("naive_bayes", 69.06, 68.17), ("knn", 88.00, 87.80),
        ("svm", 75.18, 74.36), ("mlp", 85.69, 85.50),
        ("bagging", 91.31, 91.15), ("boosting", 90.75, 90.60),
        ("stacking", 95.56, 95.56)]),
}
LABELS = {"decision_tree":"Decision Tree", "random_forest":"Random Forest",
          "naive_bayes":"Naive Bayes", "knn":"K-Nearest Neighbors",
          "bagging":"Bagging Ensemble", "stacking":"Stacking Ensemble",
          "svm":"SVM", "mlp":"MLP", "boosting":"Gradient Boosting"}


def _latest_run(key):
    # A run writes metrics.csv as models finish; the newest file is the most recent result set.
    folder = OUTPUTS / key
    candidates = []
    if folder.exists():
        for metrics in folder.glob("*/metrics.csv"):
            try:
                with metrics.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if rows:
                    candidates.append((metrics.stat().st_mtime, metrics, rows))
            except (OSError, csv.Error):
                continue
    return max(candidates, default=None, key=lambda item: item[0])


def _num(value):
    # CSV cells such as an empty CV score for stacking should display as a dash.
    try:
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def generate_report():
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    try:
        manifest = json.loads((OUTPUTS / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {"runs": {}}
    datasets = {}
    for key, (title, paper_rows) in DATASETS.items():
        # Include whatever has finished so far; missing datasets get a clear empty state.
        latest = _latest_run(key)
        rows = {}
        run_id = ""
        updated = ""
        settings = {}
        run_status = ""
        if latest:
            modified, path, records = latest
            run_id = path.parent.name
            updated = datetime.fromtimestamp(modified).astimezone().strftime("%Y-%m-%d %H:%M")
            run_info = manifest.get("runs", {}).get(run_id, {})
            # Show the run settings beside its scores so a quick run isn't mistaken for final.
            settings = run_info.get("settings", {})
            run_status = run_info.get("status", "")
            for record in records:
                model = record.get("model", "")
                rows[model] = {field: _num(record.get(field)) for field in (
                    "accuracy", "precision", "recall", "f1_score", "missed_alarm_rate",
                    "false_alarm_rate", "inference_time_seconds", "test_rows", "best_cv_accuracy")}
        datasets[key] = {"title": title, "paper": [
            {"model": m, "accuracy": a, "f1_score": f} for m, a, f in paper_rows],
            "actual": rows, "run_id": run_id, "updated": updated,
            "settings": settings, "run_status": run_status}

    # Keep the report self-contained so it opens directly from File Explorer without a server.
    payload = json.dumps(datasets, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    document = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Network Intrusion Detection — Results</title>
<style>
:root{color-scheme:light;--ink:#142238;--muted:#586a80;--line:#d8e0e9;--paper:#fff;--canvas:#f3f6fa;--navy:#14385c;--blue:#2673b8;--teal:#149b8a;--gold:#d88927;--red:#b94a48;--soft:#e9f1f8;--good:#e7f5ef;--warn:#fff3df}*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:15px/1.5 "Segoe UI",Arial,sans-serif}.shell{max-width:1440px;margin:auto;padding:28px clamp(16px,3vw,44px) 48px}.masthead{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;border-bottom:2px solid var(--navy);padding-bottom:22px}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:700;color:var(--blue)}h1{font-size:clamp(27px,4vw,42px);line-height:1.1;margin:7px 0 8px;color:var(--navy)}.subtitle{color:var(--muted);margin:0}.stamp{text-align:right;color:var(--muted);font-size:13px;white-space:nowrap}.pill{display:inline-block;background:var(--soft);color:var(--navy);padding:5px 10px;border-radius:999px;font-size:12px;font-weight:700}.intro{margin:22px 0;color:var(--muted)}.tabs{display:flex;gap:8px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin:24px 0 20px}.tab{font:inherit;color:var(--muted);background:transparent;border:0;border-bottom:3px solid transparent;padding:12px 17px;font-weight:650;cursor:pointer}.tab[aria-selected=true]{color:var(--navy);border-color:var(--blue)}.tab:focus-visible{outline:3px solid #6eb5ed;outline-offset:2px}.summary{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:20px 0}.summary-item{background:var(--paper);border:1px solid var(--line);padding:14px 17px}.summary-item b{display:block;font-size:20px;color:var(--navy)}.summary-item span{color:var(--muted);font-size:13px}.section-heading{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin:28px 0 8px}.section-heading h2{margin:0;font-size:21px;color:var(--navy)}.section-heading p{margin:0;color:var(--muted);font-size:13px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:10px 0 24px}.chart{min-width:0;background:var(--paper);border:1px solid var(--line);padding:12px 12px 8px}.chart h3{font-size:15px;margin:3px 5px;color:var(--navy)}.chart svg{width:100%;height:auto;display:block}.gridline{stroke:#dfe5ec;stroke-width:1}.axis{fill:#54657a;font-size:11px}.bar-paper{fill:#a7b7c8}.bar-current{fill:var(--blue)}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:4px 7px 8px;color:var(--muted);font-size:12px}.swatch{display:inline-block;width:11px;height:11px;margin-right:6px;vertical-align:-1px}.table-wrap{overflow-x:auto;background:var(--paper);border:1px solid var(--line)}table{border-collapse:collapse;width:100%;min-width:950px;font-size:13px}thead{background:#eaf0f6;color:var(--navy)}th,td{padding:10px 9px;text-align:right;border-bottom:1px solid #e2e8ef;white-space:nowrap;font-variant-numeric:tabular-nums}th{font-weight:700}td:first-child,th:first-child{text-align:left;position:sticky;left:0;background:inherit}tbody tr:hover{background:#f4f8fc}.delta-positive{color:#087a55;font-weight:700}.delta-negative{color:#ad413e;font-weight:700}.delta-na{color:#7d8996}.status{font-size:12px;color:var(--muted);margin:9px 0}.empty{padding:20px;background:var(--warn);color:#62440f;margin:12px 0}.note{font-size:12px;color:var(--muted);margin:10px 0 20px}.footer{border-top:1px solid var(--line);margin-top:34px;padding-top:15px;color:var(--muted);font-size:12px}.panel[hidden]{display:none}@media(max-width:760px){.shell{padding-top:18px}.masthead{display:block}.stamp{text-align:left;margin-top:12px}.summary{grid-template-columns:1fr}.charts{grid-template-columns:1fr}.section-heading{display:block}.section-heading p{margin-top:4px}}@media print{body{background:#fff}.shell{max-width:none;padding:0}.tabs{display:none}.panel[hidden]{display:block!important;break-before:page}.panel{break-inside:avoid}.chart,.summary-item,.table-wrap{border-color:#aaa}.chart{break-inside:avoid}}
</style>
</head><body><main class="shell">
<header class="masthead"><div><div class="eyebrow">Research results dashboard</div><h1>Network Intrusion Detection</h1><p class="subtitle">Machine learning comparison across CIC-UNSW-NB15, HIKARI-2021 and UNR-IDD</p></div><div class="stamp"><span class="pill">Paper comparison</span><div id="generated" style="margin-top:8px"></div><div>Print or save as PDF from the browser</div></div></header>
<p class="intro">Compare this project’s held-out test results with the values reported in the paper. Metrics shown as percentages are converted from the CSV fractions. Each dataset tab shows its own model scores, alarm rates and inference time.</p>
<div class="summary" id="summary"></div>
<nav class="tabs" role="tablist" aria-label="Dataset results" id="tabs"></nav>
<div id="panels"></div>
<footer class="footer">Paper reference values transcribed from the paper’s model metric tables. Current values come from each dataset’s latest available metrics.csv in this project’s outputs folder. This is an implementation comparison; differences in preprocessing, data versions, hardware and run settings can affect results.</footer>
</main><script>
const data=__PAYLOAD__;
// Keep model names in one place so the plots and tables use the same labels.
const names={decision_tree:'Decision Tree',random_forest:'Random Forest',naive_bayes:'Naive Bayes',knn:'K-Nearest Neighbors',bagging:'Bagging Ensemble',stacking:'Stacking Ensemble',svm:'SVM',mlp:'MLP',boosting:'Gradient Boosting'};
const order=['cic','hikari','unr'];
const pct=x=>x==null?'—':(x*100).toFixed(2)+'%';
const sec=x=>x==null?'—':x.toFixed(x<1?3:2);
const safe=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
// Draw the paper and current scores side by side, using the same 0–100 scale.
function chartSvg(ds,metric){const paper=ds.paper, actual=ds.actual;const rows=paper.filter(p=>actual[p.model]);if(!rows.length)return '<div class="empty">Run this dataset to populate the comparison plot.</div>';const W=760,H=250,L=48,R=12,T=12,B=70,plotW=W-L-R,plotH=H-T-B;let out=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${metric==='accuracy'?'Accuracy':'F1 score'} comparison chart. Paper and project results in percent.">`;[0,25,50,75,100].forEach(v=>{const y=T+plotH*(1-v/100);out+=`<line class="gridline" x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"/><text class="axis" x="${L-7}" y="${y+4}" text-anchor="end">${v}</text>`});const group=plotW/rows.length,bw=Math.min(23,group*.25);rows.forEach((r,i)=>{const a=actual[r.model], p=r[metric],c=a[metric];const x=L+group*i+group/2;[[p,'bar-paper',-bw/2-1],[c,'bar-current',bw/2+1]].forEach(([val,cls,off])=>{if(val==null)return;const y=T+plotH*(1-val/100);out+=`<rect class="${cls}" x="${x+off}" y="${y}" width="${bw}" height="${T+plotH-y}" rx="2"><title>${safe(names[r.model])}: ${Number(val).toFixed(2)}%</title></rect>`});out+=`<text class="axis" x="${x}" y="${T+plotH+18}" text-anchor="middle">${safe(names[r.model]).replace(' Ensemble','')}</text>`});out+='</svg>';return out}
// Put the paper's reference values and the saved test metrics on matching model rows.
function table(ds){const paper=Object.fromEntries(ds.paper.map(x=>[x.model,x]));const models=[...new Set([...ds.paper.map(x=>x.model),...Object.keys(ds.actual)])];const heads=['Model','Accuracy','Paper acc.','Δ acc.','Precision','Recall','F1 score','Paper F1','Δ F1','MAR','FAR','Inference (s)'];let s='<div class="table-wrap"><table><thead><tr>'+heads.map(x=>`<th>${x}</th>`).join('')+'</tr></thead><tbody>';models.forEach(m=>{const a=ds.actual[m],p=paper[m];const delta=(x,y)=>x==null||y==null?'<span class="delta-na">—</span>':`<span class="${x-y>=0?'delta-positive':'delta-negative'}">${(x-y>=0?'+':'')+(x-y).toFixed(2)} pp</span>`;s+=`<tr><td>${safe(names[m]||m)}</td><td>${a?pct(a.accuracy):'—'}</td><td>${p?p.accuracy.toFixed(2)+'%':'—'}</td><td>${a&&p?delta(a.accuracy*100,p.accuracy):'<span class="delta-na">—</span>'}</td><td>${a?pct(a.precision):'—'}</td><td>${a?pct(a.recall):'—'}</td><td>${a?pct(a.f1_score):'—'}</td><td>${p?p.f1_score.toFixed(2)+'%':'—'}</td><td>${a&&p?delta(a.f1_score*100,p.f1_score):'<span class="delta-na">—</span>'}</td><td>${a?pct(a.missed_alarm_rate):'—'}</td><td>${a?pct(a.false_alarm_rate):'—'}</td><td>${a?sec(a.inference_time_seconds):'—'}</td></tr>`});s+='</tbody></table></div>';return s}
const total=order.filter(k=>Object.keys(data[k].actual).length).length;
document.getElementById('generated').textContent='Report generated '+new Date().toLocaleString();
document.getElementById('summary').innerHTML=order.map(k=>{const d=data[k],n=Object.keys(d.actual).length;return `<div class="summary-item"><b>${n?`${n} models reported`:'Awaiting run'}</b><span>${safe(d.title)} · ${d.updated?'updated '+safe(d.updated):'no metrics file yet'}</span></div>`}).join('');
document.getElementById('tabs').innerHTML=order.map((k,i)=>`<button class="tab" role="tab" id="tab-${k}" aria-controls="panel-${k}" aria-selected="${i===0}" type="button">${safe(data[k].title)}</button>`).join('');
document.getElementById('panels').innerHTML=order.map((k,i)=>{const d=data[k],s=d.settings||{};const cfg=[s.trials?`${s.trials} Optuna trials`:null,s.cv_folds?`${s.cv_folds}-fold CV`:null,s.seed!=null?`seed ${s.seed}`:null].filter(Boolean).join(' · ');const status=d.run_id?`Run ${safe(d.run_id)} · ${d.actual.stacking?`${d.actual.stacking.test_rows?.toLocaleString()||'—'} test rows`:'partial model results'}${cfg?' · '+safe(cfg):''}${d.run_status==='complete'?' · completed':d.run_status?' · '+safe(d.run_status):''}`:'No results yet. Run the training pipeline, then regenerate this report.';return `<section class="panel" id="panel-${k}" role="tabpanel" aria-labelledby="tab-${k}" ${i?'hidden':''}><div class="section-heading"><h2>${safe(d.title)}</h2><p>${status}</p></div>${!Object.keys(d.actual).length?'<div class="empty">No completed metrics were found for this dataset yet.</div>':''}<div class="charts"><div class="chart"><h3>Accuracy (%)</h3><div class="legend"><span><i class="swatch" style="background:#a7b7c8"></i>Paper</span><span><i class="swatch" style="background:#2673b8"></i>Your run</span></div>${chartSvg(d,'accuracy')}</div><div class="chart"><h3>Weighted F1 score (%)</h3><div class="legend"><span><i class="swatch" style="background:#a7b7c8"></i>Paper</span><span><i class="swatch" style="background:#2673b8"></i>Your run</span></div>${chartSvg(d,'f1_score')}</div></div><div class="section-heading"><h2>Model-by-model results</h2><p>Δ is your result minus the paper, in percentage points</p></div>${table(d)}<p class="note">MAR = 1 − weighted recall; FAR = 1 − weighted precision, as used in this project. Inference time is measured on this run’s test set; compare it cautiously because hardware and timing setup may differ from the paper.</p></section>`}).join('');
// Switching tabs only changes what is visible; the report works without an internet connection.
document.querySelectorAll('[role=tab]').forEach(btn=>btn.addEventListener('click',()=>{document.querySelectorAll('[role=tab]').forEach(b=>b.setAttribute('aria-selected',String(b===btn)));document.querySelectorAll('.panel').forEach(p=>p.hidden=p.id!==btn.getAttribute('aria-controls'))}));
</script></body></html>'''.replace("__PAYLOAD__", payload)
    # The result page is generated from saved CSVs; rebuilding it never reruns training.
    REPORT.write_text(document, encoding="utf-8")
    return REPORT

if __name__ == "__main__":
    print(generate_report())
