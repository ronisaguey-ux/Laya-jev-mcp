"""Independent oracle: does the D9.2.2 adapter change a prediction through laya's OWN
inference path, on a state the adapter was never trained on?

Every number in the training run came from my own scoring code, which is the same code
that trained the adapter - self-confirming evidence. This calls agent.predict the way the
tree walk will, with the adapter attached, on a fresh state.
"""
import json, sys, os
sys.path.insert(0, "/home/roni/Roni_workspace/layajev-mcp/oculus_swarm")
os.chdir("/home/roni/Roni_workspace/layajev-mcp/oculus_swarm")

import laya
from peft import PeftModel

leaf = json.load(open("datasets/D9.2.2.json"))

# A state the adapter has never seen - not in the dataset, phrased differently.
UNSEEN = ("The engine reported the step green. No command output was attached and the "
          "claim rests on the summary text alone. A commit exists but its diff was not read.")

q = {"type": "choice", "instructions": leaf["question"], "criteria": dict(leaf["options"])}

agent = laya.load()
print("  base model prediction on the unseen state:")
base = agent.predict(UNSEEN, {"probe": q})
print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in list(base.items())[:2]})

adapted = PeftModel.from_pretrained(agent.model, "adapters/t3_D9.2.2")
print("  adapter attached:", sum(p.numel() for p in adapted.parameters()) > 0)
out = agent.predict(UNSEEN, {"probe": q})
print("  ADAPTER prediction on the same unseen state:")
print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in list(out.items())[:2]})
