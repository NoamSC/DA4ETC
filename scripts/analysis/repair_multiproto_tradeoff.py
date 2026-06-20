import os,sys,json,glob
import numpy as np
ROOT="/home/anatbr/students/noamshakedc/da4etc"; sys.path.insert(0,ROOT)
sys.path.insert(0,os.path.join(ROOT,"scripts/analysis"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import few_shot_repair_loop as F
names=sorted(json.load(open(F.LABEL_MAP)).keys()); n_classes=len(names)
src_emb,src_lab=F.load_week_embeddings(16)
src_protos,src_counts=F.class_prototypes(src_emb,src_lab,n_classes); src_valid=src_counts>0
# class -> (drift_week, teleported)
CLS={57:(18,True,"eset-edtd"),49:(28,True,"docker-registry"),140:(28,True,"skype"),
     76:(39,True,"google-fonts"),98:(22,False,"microsoft-defender")}
teleported={c for c,(_,t,_) in CLS.items() if t}
stable=np.array([c for c in range(n_classes) if c not in teleported and src_valid[c]])
K=50;SEEDS=F.SEEDS;M_LIST=[1,2,3,4]
def pick_eval(c,dw):
    files=sorted(glob.glob(os.path.join(F.INF_DIR,"WEEK-2022-*.npz")));best=None
    for f in files:
        w=int(os.path.basename(f).split("-")[-1].split(".")[0])
        if w<=dw:continue
        _,lab=F.load_week_embeddings(w)
        if (lab==c).sum()>=40 and w>=44:best=w
    return best
def pool(c,dw,ew):
    files=sorted(glob.glob(os.path.join(F.INF_DIR,"WEEK-2022-*.npz")));P=[]
    for f in files:
        w=int(os.path.basename(f).split("-")[-1].split(".")[0])
        if dw<=w<ew:e,l=F.load_week_embeddings(w);P.append(e[l==c])
    return np.concatenate(P) if P else np.zeros((0,600))
def predc(emb,c,subp):
    sidx=np.where(src_valid)[0];sidx=sidx[sidx!=c];Pst=src_protos[sidx]
    d_st=(-2*emb@Pst.T+(Pst*Pst).sum(1)[None,:]).min(1)
    d_sub=(-2*emb@subp.T+(subp*subp).sum(1)[None,:]).min(1)
    return d_sub<d_st
DATA={}
for c,(dw,tel,nm) in CLS.items():
    ew=pick_eval(c,dw);emb_e,lab_e=F.load_week_embeddings(ew);eval_c=lab_e==c
    sup=pool(c,dw,ew);npool=sup.shape[0]
    sm=np.isin(lab_e,stable[stable!=c]);nst=int(sm.sum())
    # before-fix: source proto of c
    pc0=predc(emb_e,c,src_protos[c:c+1]); rec0=float(pc0[eval_c].mean());poa0=100*pc0[sm].sum()/nst
    rows=[]
    for M in M_LIST:
        rs=[];ps=[]
        for s in SEEDS:
            rng=np.random.default_rng(s);sh=sup[rng.choice(npool,K,replace=False)]
            subp=sh.mean(0,keepdims=True) if M==1 else KMeans(M,n_init=10,random_state=s).fit(sh).cluster_centers_
            pc=predc(emb_e,c,subp);rs.append(float(pc[eval_c].mean()));ps.append(100*pc[sm].sum()/nst)
        rows.append(dict(M=M,recall=float(np.mean(rs)),poach=float(np.mean(ps))))
    DATA[nm]=dict(idx=c,teleported=tel,eval_wk=ew,n_stable=nst,
                  before=dict(recall=rec0,poach=float(poa0)),rows=rows)
json.dump(DATA,open(os.path.join(ROOT,"results/repair/repair_tradeoff_M_v2.json"),"w"),indent=2)

# ---------------- clean plot ----------------
plt.rcParams.update({"font.size":11})
fig,ax=plt.subplots(figsize=(9.2,6.2))
colors={"eset-edtd":"#1f77b4","docker-registry":"#2ca02c","skype":"#d62728",
        "google-fonts":"#9467bd","microsoft-defender":"#7f7f7f"}
for nm,d in DATA.items():
    col=colors[nm];ctrl=not d["teleported"]
    b=d["before"];pb=b["poach"]                      # baseline absolute poach of c's source proto
    xs=[r["poach"]-pb for r in d["rows"]]            # MARGINAL Δ-poach (repair-induced) vs source-only
    ys=[r["recall"] for r in d["rows"]]
    # before-fix -> M1 connector (the core repair jump); before-fix anchored at Δ=0 by definition
    ax.plot([0.0,xs[0]],[b["recall"],ys[0]],ls=":",color=col,lw=1.3,alpha=0.5,zorder=2)
    # fix trajectory M1..M4
    ax.plot(xs,ys,"-",color=col,lw=2.0,alpha=0.55 if ctrl else 0.95,zorder=3)
    ax.scatter(xs,ys,color=col,s=40,zorder=4,edgecolor="white",linewidth=0.5)
    # before-fix marker (hollow star) at Δ=0
    ax.scatter([0.0],[b["recall"]],marker="*",s=240,facecolor="none",
               edgecolor=col,linewidth=1.8,zorder=5)
    # endpoint labels: M1 and M4 only + class name
    ax.annotate("M1",(xs[0],ys[0]),textcoords="offset points",xytext=(4,5),fontsize=8,color=col)
    ax.annotate("M4",(xs[-1],ys[-1]),textcoords="offset points",xytext=(4,5),fontsize=8,color=col)
    lab=nm+(" (control)" if ctrl else "")
    ax.annotate(lab,(xs[-1],ys[-1]),textcoords="offset points",xytext=(9,-9),
                fontsize=9,color=col,fontweight="bold")
# aggregate over teleported
tele=[n for n,d in DATA.items() if d["teleported"]]
mx=[];my=[];dx=[];dy=[]
for i,M in enumerate(M_LIST):
    xs=[DATA[n]["rows"][i]["poach"]-DATA[n]["before"]["poach"] for n in tele]
    ys=[DATA[n]["rows"][i]["recall"] for n in tele]
    mx.append(np.mean(xs));my.append(np.mean(ys));dx.append(np.median(xs));dy.append(np.median(ys))
# aggregate before-fix anchor (Δ=0 by definition)
mb=float(np.mean([DATA[n]["before"]["recall"] for n in tele]))
db=float(np.median([DATA[n]["before"]["recall"] for n in tele]))
ax.plot([0.0,mx[0]],[mb,my[0]],ls=":",color="black",lw=1.8,alpha=0.7,zorder=6)
ax.plot([0.0,dx[0]],[db,dy[0]],ls=":",color="black",lw=1.3,alpha=0.5,zorder=6)
ax.plot(mx,my,"-o",color="black",lw=2.8,ms=7,zorder=6)
ax.plot(dx,dy,"--s",color="black",lw=1.6,ms=6,alpha=0.65,zorder=6)
ax.scatter([0.0],[mb],marker="*",s=300,facecolor="none",edgecolor="black",linewidth=2.4,zorder=7)
ax.scatter([0.0],[db],marker="*",s=190,facecolor="none",edgecolor="black",linewidth=1.5,alpha=0.7,zorder=7)
ax.annotate("mean",(0.0,mb),textcoords="offset points",xytext=(7,-2),fontsize=8,fontweight="bold")
ax.annotate("median",(0.0,db),textcoords="offset points",xytext=(7,-10),fontsize=8,alpha=0.8)
for M,x,y in zip(M_LIST,mx,my):
    ax.annotate(f"M{M}",(x,y),textcoords="offset points",xytext=(6,-12),fontsize=8,fontweight="bold")
# legend (semantics only, classes labelled inline)
from matplotlib.lines import Line2D
leg=[Line2D([0],[0],marker="*",ls="none",mfc="none",mec="k",ms=13,label="before fix (source-only, Δ=0)"),
     Line2D([0],[0],marker="o",color="k",lw=2.8,label="teleported mean (M1→M4)"),
     Line2D([0],[0],marker="s",color="k",ls="--",lw=1.6,alpha=0.7,label="teleported median")]
ax.legend(handles=leg,fontsize=9,loc="lower right",framealpha=0.92)
ax.axvline(0,color="#999",lw=1.0,alpha=0.6,zorder=1)
ax.set_xlabel("Additional stable data poached by the repair  (Δ% vs. source-only)   →   negative transfer")
ax.set_ylabel("Flagged-class recall@1   (the fix)")
ax.set_title("Few-shot prototype repair: recovery vs. repair-induced poaching\n"
             "★ broken source-only (Δ=0) → M=1…4 sub-prototypes (k=50 shots, frozen W16, forward-only)")
ax.grid(alpha=0.3);ax.set_ylim(-0.02,1.04)
fig.tight_layout()
out=os.path.join(ROOT,"figs/repair/fig_repair_recovery_vs_poach_M.png")
fig.savefig(out,dpi=160);print("wrote",out)
for n,d in DATA.items():
    b=d["before"];r=d["rows"]
    print(f"{n:20} before={b['recall']:.2f}/{b['poach']:.2f}%  "+
          "  ".join(f"M{x['M']}={x['recall']:.2f}/{x['poach']:.2f}%" for x in r))
