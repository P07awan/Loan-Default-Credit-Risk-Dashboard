"""Build docs/index.html: a self-contained static dashboard (works on GitHub Pages, no dependencies)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = json.load(open(os.path.join(ROOT, "reports", "results.json")))
TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Loan Default &amp; Credit Risk Dashboard</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--tx:#1b2430;--mu:#667085;--bd:#e4e7ec;--ac:#d9480f;--bl:#4f8cc9}
@media(prefers-color-scheme:dark){:root{--bg:#10141a;--card:#181e27;--tx:#e8ecf1;--mu:#98a2b3;--bd:#2a3342;--ac:#ff7a45;--bl:#6fa8e0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 48px}h1{font-size:24px;margin:0 0 4px}.sub{color:var(--mu);margin:0 0 20px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:16px}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--mu);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}h2{font-size:15px;margin:0 0 10px}
svg{width:100%;height:auto;display:block}svg text{fill:var(--mu);font-size:11px}
select{background:var(--card);color:var(--tx);border:1px solid var(--bd);border-radius:6px;padding:4px 8px;margin-bottom:8px}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:5px 8px;text-align:right;border-bottom:1px solid var(--bd)}th:first-child,td:first-child{text-align:left}
.note{color:var(--mu);font-size:12px;margin-top:6px}.wide{overflow-x:auto}
</style></head><body><main>
<h1>Loan Default &amp; Credit Risk Dashboard</h1>
<p class="sub">UCI Credit Card Default dataset &middot; 30,000 Taiwanese cardholders &middot; Apr–Sep 2005 behaviour predicting next-month default</p>
<div class="kpis" id="kpis"></div>
<div class="grid">
<div class="card"><h2>Default rate by model risk decile (test set)</h2><div id="dec"></div><div class="note">Dashed line = portfolio average. A good model rises steadily left to right.</div></div>
<div class="card"><h2>ROC curves (test set)</h2><div id="roc"></div></div>
<div class="card"><h2>Default rate by segment</h2><select id="seg"></select><div id="segc"></div></div>
<div class="card"><h2>Top drivers of default (permutation importance)</h2><div id="imp"></div></div>
<div class="card wide"><h2>Model comparison (test set)</h2><table id="mt"></table></div>
<div class="card wide"><h2>Risk-decile table</h2><table id="dt"></table></div>
</div>
<p class="note">SEX and MARRIAGE are excluded from the model as inputs. Selected model: <span id="bm"></span>.</p>
</main><script>
const D=__DATA__,$=id=>document.getElementById(id),pct=(x,d=1)=>(x*100).toFixed(d)+"%",fmt=n=>n.toLocaleString("en-US");
const S=D.summary,M=D.metrics[S.best_model],cf=D.confusion;
$("bm").textContent=S.best_model+" (decision threshold "+M.threshold+")";
$("kpis").innerHTML=[[fmt(S.customers),"Customers"],[pct(S.default_rate),"Default rate"],[fmt(Math.round(S.avg_limit)),"Avg credit limit (NT$)"],
[pct(S.avg_utilization,0),"Avg utilisation"],[M.roc_auc.toFixed(3),"Test ROC-AUC"],[M.ks.toFixed(3),"KS statistic"],[pct(M.recall,0),"Recall @ threshold"],[pct(M.precision,0),"Precision @ threshold"]]
.map(k=>`<div class="card kpi"><b>${k[0]}</b><span>${k[1]}</span></div>`).join("");
function bars(el,items,avg,color){const W=520,H=250,p={l:38,b:44,t:10,r:8},mx=Math.max(...items.map(i=>i.v),avg||0)*1.12,bw=(W-p.l-p.r)/items.length;
let s=`<svg viewBox="0 0 ${W} ${H}" role="img">`;for(let i=0;i<=4;i++){const y=H-p.b-(H-p.b-p.t)*i/4;s+=`<line x1="${p.l}" x2="${W-p.r}" y1="${y}" y2="${y}" stroke="var(--bd)"/><text x="${p.l-6}" y="${y+4}" text-anchor="end">${Math.round(mx*i/4*100)}%</text>`}
items.forEach((it,i)=>{const h=(H-p.b-p.t)*it.v/mx,x=p.l+i*bw+bw*.15;s+=`<rect x="${x}" y="${H-p.b-h}" width="${bw*.7}" height="${h}" rx="3" fill="${color}"><title>${it.l}: ${pct(it.v)}${it.n?" ("+fmt(it.n)+" customers)":""}</title></rect><text x="${x+bw*.35}" y="${H-p.b-h-4}" text-anchor="middle">${Math.round(it.v*100)}</text><text x="${x+bw*.35}" y="${H-p.b+16}" text-anchor="middle">${it.l}</text>`});
if(avg!=null){const y=H-p.b-(H-p.b-p.t)*avg/mx;s+=`<line x1="${p.l}" x2="${W-p.r}" y1="${y}" y2="${y}" stroke="var(--tx)" stroke-dasharray="4 3"/>`}
$(el).innerHTML=s+"</svg>"}
bars("dec",D.deciles.map(d=>({l:d.decile,v:d.default_rate,n:d.customers})),S.default_rate,"var(--ac)");
// ROC
(()=>{const W=520,H=300,p=40,col={"Logistic Regression":"#9aa5b1","Random Forest":"var(--bl)","Gradient Boosting":"var(--ac)"};
let s=`<svg viewBox="0 0 ${W} ${H}"><line x1="${p}" y1="${H-p}" x2="${W-10}" y2="${10}" stroke="var(--bd)" stroke-dasharray="4"/><line x1="${p}" y1="${H-p}" x2="${W-10}" y2="${H-p}" stroke="var(--bd)"/><line x1="${p}" y1="10" x2="${p}" y2="${H-p}" stroke="var(--bd)"/>`;
let ly=22;for(const [n,r] of Object.entries(D.roc)){s+=`<polyline fill="none" stroke="${col[n]}" stroke-width="2.2" points="${r.fpr.map((f,i)=>`${p+f*(W-p-10)},${H-p-r.tpr[i]*(H-p-10)}`).join(" ")}"/>`;
s+=`<rect x="${W-190}" y="${H-p-86+ly-22}" width="10" height="10" fill="${col[n]}"/><text x="${W-175}" y="${H-p-77+ly-22}">${n} (${D.metrics[n].roc_auc.toFixed(3)})</text>`;ly+=16}
s+=`<text x="${W/2}" y="${H-8}" text-anchor="middle">False positive rate</text><text x="12" y="${H/2}" transform="rotate(-90 12 ${H/2})" text-anchor="middle">True positive rate</text></svg>`;$("roc").innerHTML=s})();
// segments
const sel=$("seg");Object.keys(D.segments).forEach(k=>sel.add(new Option(k,k)));
const drawSeg=()=>bars("segc",D.segments[sel.value].map(x=>({l:x.segment,v:x.default_rate,n:x.customers})),S.default_rate,"var(--bl)");sel.onchange=drawSeg;drawSeg();
// importance
(()=>{const it=D.importance,W=520,rh=22,H=it.length*rh+8,mx=it[0].importance;let s=`<svg viewBox="0 0 ${W} ${H}">`;
it.forEach((x,i)=>{const w=(W-190)*x.importance/mx;s+=`<text x="150" y="${i*rh+16}" text-anchor="end">${x.feature}</text><rect x="158" y="${i*rh+5}" width="${w}" height="14" rx="3" fill="var(--bl)"/><text x="${164+w}" y="${i*rh+16}">${x.importance.toFixed(3)}</text>`});$("imp").innerHTML=s+"</svg>"})();
$("mt").innerHTML="<tr><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>KS</th></tr>"+Object.entries(D.metrics).map(([n,m])=>`<tr><td>${n}${n==S.best_model?" ★":""}</td><td>${m.roc_auc.toFixed(3)}</td><td>${m.pr_auc.toFixed(3)}</td><td>${m.ks.toFixed(3)}</td></tr>`).join("");
$("dt").innerHTML="<tr><th>Decile</th><th>Customers</th><th>Avg predicted PD</th><th>Actual default rate</th><th>Lift</th></tr>"+D.deciles.map(d=>`<tr><td>${d.decile}</td><td>${d.customers}</td><td>${pct(d.avg_pd)}</td><td>${pct(d.default_rate)}</td><td>${d.lift.toFixed(2)}x</td></tr>`).join("");
</script></body></html>"""
html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
open(os.path.join(ROOT, "docs", "index.html"), "w", encoding="utf-8").write(html)
print("docs/index.html", len(html) // 1024, "KB")
