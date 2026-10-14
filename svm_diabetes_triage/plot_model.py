"""Plot the exported linear SVM and one input (optional matplotlib dependency)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Draw a linear SVM boundary and decision.")
    parser.add_argument("--model", type=Path, default=ROOT / "dist/model.json")
    parser.add_argument("--output", type=Path, default=ROOT / "graphs/linear_svm_decision.png")
    parser.add_argument("--glucose", type=float, default=105)
    parser.add_argument("--bmi", type=float, default=29)
    args = parser.parse_args()
    if not all(np.isfinite(v) and v>0 for v in (args.glucose,args.bmi)):
        parser.error("glucose and BMI must be finite and positive")
    model = json.loads(args.model.read_text(encoding="utf-8"))
    w,mean,scale = (np.array(model[key],dtype=float) for key in ("weights","means","scales"))
    b=model["bias"]
    z=(np.array([args.glucose,args.bmi])-mean)/scale
    score=float(w@z+b)
    norm=float(np.linalg.norm(w))
    projection=z-score*w/(norm**2) if norm else z
    rows=model["training_examples"]
    X=np.array([row["z"] for row in rows])
    y=np.array([row["numeric_label"] for row in rows])
    support=np.array([row["is_support_example"] for row in rows])
    lo=np.floor(min(X.min(),z.min(),projection.min(),-2)-0.5)
    hi=np.ceil(max(X.max(),z.max(),projection.max(),2)+0.5)
    axis=np.linspace(lo,hi,300)
    xx,yy=np.meshgrid(axis,axis)
    scores=w[0]*xx+w[1]*yy+b
    fig,ax=plt.subplots(figsize=(9,9))
    ax.contourf(xx,yy,scores,levels=[-np.inf,0,np.inf],colors=["#edf4fa","#fcf0f2"])
    ax.contourf(xx,yy,scores,levels=[-1,1],colors=["#fff2d0"])
    if norm:
        lines=ax.contour(xx,yy,scores,levels=[-1,0,1],colors=["#c18a27","#243e48","#c18a27"],linestyles=["--","-","--"],linewidths=[1.8,2.3,1.8])
        label_y = lo + 0.5
        if abs(w[0]) > 1e-10:
            manual = [((target-b-w[1]*label_y)/w[0],label_y) for target in (-1,0,1)]
        else:
            manual = [(lo+0.5,(target-b)/w[1]) for target in (-1,0,1)]
        ax.clabel(lines,fmt={-1:"score = -1",0:"score = 0",1:"score = +1"},fontsize=10,manual=manual)
    ax.scatter(X[y==-1,0],X[y==-1,1],c="#3a78a5",s=35,label="Class -1: lower reference",zorder=4)
    ax.scatter(X[y==1,0],X[y==1,1],c="#b84a5c",s=35,marker="s",label="Class +1: higher reference",zorder=4)
    ax.scatter(X[support,0],X[support,1],facecolors="none",edgecolors="#172f39",s=170,linewidths=1.6,label="Support vector (alpha > 0)",zorder=5)
    ax.scatter(*z,color="#dda531",edgecolor="#8b5f13",s=120,marker="D",label="Input",zorder=7)
    if norm:
        ax.plot([z[0],projection[0]],[z[1],projection[1]],"--",color="#bd7b0f",lw=2,zorder=6)
        ax.scatter(*projection,facecolor="white",edgecolor="#bd7b0f",s=40,zorder=6)
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel="Standardized fasting glucose (z1)",ylabel="Standardized BMI (z2)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left",fontsize=9)
    predicted=1 if score>=0 else -1
    fig.suptitle("Linear SVM: binary decision and maximum margin",fontsize=17,fontweight="bold",y=0.97)
    distance=f"{score/norm:.4f}" if norm else "undefined"
    ax.set_title(f"Glucose={args.glucose:g}, BMI={args.bmi:g} -> class {predicted:+d}\nscore={score:.4f}; signed distance={distance}; support vectors={support.sum()}",fontsize=11,pad=15)
    fig.text(0.5,0.025,"Margin planes are +/-1. Every input is classified by score sign, including points inside the margin.",ha="center",fontsize=9)
    fig.subplots_adjust(top=0.86,bottom=0.11,left=0.11,right=0.95)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)
    print(f"Saved {args.output} and {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
