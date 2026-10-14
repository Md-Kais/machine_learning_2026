"""Save an inspectable ID3 decision graph (optional matplotlib dependency)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from decision_tree_model import FEATURES, ID3DecisionTree

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Draw the exported ID3 tree and chosen path.")
    parser.add_argument("--model", type=Path, default=ROOT / "dist/model.json")
    parser.add_argument("--output", type=Path, default=ROOT / "graphs/id3_decision.png")
    for feature, value in zip(FEATURES, ("Sunny", "Mild", "High", "Strong")):
        parser.add_argument("--"+feature.lower(), default=value)
    args = parser.parse_args()
    data = json.loads(args.model.read_text(encoding="utf-8"))
    model = ID3DecisionTree()
    model.tree = data["tree"]
    values = {key: getattr(args,key.lower()) for key in FEATURES}
    result = model.predict(values)
    positions, nodes, edges, leaves = {}, [], [], []

    def layout(node, depth=0):
        if node["is_leaf"]:
            x = len(leaves)
            leaves.append(node)
        else:
            xs = []
            for value, child in node["children"].items():
                xs.append(layout(child,depth+1))
                edges.append((node,child,value))
            x = (xs[0]+xs[-1])/2
        positions[node["id"]] = (x,-depth)
        nodes.append(node)
        return x

    layout(data["tree"])
    active = {step["node_id"] for step in result["path"]} | {result["leaf_id"]}
    fig, ax = plt.subplots(figsize=(max(11,len(leaves)*2.4),6))
    fig.patch.set_facecolor("#f7faf8")
    for parent,child,value in edges:
        x,y=positions[parent["id"]]
        u,v=positions[child["id"]]
        used=any(s["node_id"]==parent["id"] and s["value"]==value and not s["fallback"] for s in result["path"])
        color="#c98217" if used else "#b4c6bf"
        ax.plot([x,u],[y,v],color=color,lw=3 if used else 1.5,zorder=1)
        ax.text((x+u)/2,(y+v)/2,value,ha="center",va="center",fontsize=11,
                bbox=dict(facecolor="white",edgecolor="none",pad=3),zorder=2)
    for node in nodes:
        x,y=positions[node["id"]]
        label="Play = "+node["prediction"] if node["is_leaf"] else node["feature"]
        label+=f"\nH = {node['entropy']:.3f} | n = {node['sample_count']}"
        label+=f"\nYes {node['class_counts']['Yes']} / No {node['class_counts']['No']}"
        if not node["is_leaf"]:
            label+=f"\nGain = {node['gain']:.3f}"
        used=node["id"] in active
        color="#c98217" if used else "#237658" if node["prediction"]=="Yes" else "#ac4653"
        ax.text(x,y,label,ha="center",va="center",fontsize=10,
                bbox=dict(boxstyle="round,pad=0.7",facecolor="#fff3d5" if used else "#eef4f0",edgecolor=color,lw=2),zorder=3)
    depth=max(node["depth"] for node in nodes)
    ax.set_xlim(-0.65,max(0,len(leaves)-1)+0.65)
    ax.set_ylim(-depth-0.5,0.5)
    ax.axis("off")
    fig.suptitle("ID3 Play Tennis: entropy and information gain",fontsize=19,fontweight="bold",y=0.96)
    path=" -> ".join(s["feature"]+"="+s["value"] for s in result["path"])
    fig.text(0.5,0.035,", ".join(key+"="+values[key] for key in FEATURES)+"\n"+path+" -> Play="+result["label"],ha="center",fontsize=11)
    fig.subplots_adjust(top=0.83,bottom=0.18,left=0.03,right=0.97)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180,facecolor=fig.get_facecolor())
    fig.savefig(args.output.with_suffix(".svg"),facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {args.output} and {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
