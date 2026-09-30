from __future__ import annotations
import json, os
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
R=Path(os.environ.get("BENCHFLOW_WORKSPACE","/root")); I=json.loads((R/"instance.json").read_text()); op=I["operation"]; rows=I["rows"]; variant=I["variant"]; audit={}
if variant=="baseline": audit={"records_used":len(rows)}
elif variant=="conflict":
 resolved={}
 for r in rows:
  k=r["logical_key"]
  if k not in resolved or r["source_priority"]>resolved[k]["source_priority"]:resolved[k]=r
 audit={"resolved_conflict_keys":sorted(k for k in resolved if sum(x["logical_key"]==k for x in rows)>1)}; rows=[resolved[k] for k in sorted(resolved)]
elif variant=="temporal":
 future=sorted(r["record_id"] for r in rows if r["effective_at"]>I["as_of"]); rows=[r for r in rows if r["effective_at"]<=I["as_of"]]; audit={"excluded_future_record_ids":future,"as_of":I["as_of"]}
elif variant=="adversarial":
 ignored=sorted(r["record_id"] for r in rows if r["is_distractor"]); rows=[r for r in rows if not r["is_distractor"]]; audit={"ignored_distractor_ids":ignored}
def q(x): return str(Decimal(str(x)).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP))
if op=="weighted_rollup":
 d=defaultdict(Decimal)
 for r in rows:d[r["group"]]+=Decimal(str(r["quantity"]))*Decimal(str(r["unit_value"]))
 totals={k:q(v) for k,v in sorted(d.items())}; ans={"totals":totals,"winner":max(totals,key=lambda k:(Decimal(totals[k]),k))}
elif op=="claim_coverage":
 valid={r["claim"] for r in rows if r["admissible"] and r["source_id"]}; claims=sorted({r["claim"] for r in rows}); ans={"supported":sorted(valid),"unsupported":sorted(set(claims)-valid)}
elif op=="source_precedence":
 d=defaultdict(list)
 for r in rows:d[r["claim"]].append(r)
 chosen={};unsupported=[]
 for k,v in sorted(d.items()):
  good=[r for r in v if r["admissible"]]
  if not good:unsupported.append(k)
  else:chosen[k]=min(good,key=lambda r:(r["authority_rank"],r["source_id"]))["statement"]
 ans={"chosen_statements":chosen,"unsupported":unsupported}
elif op=="trimmed_mean":
 d=defaultdict(list)
 for r in rows:d[r["sensor"]].append(Decimal(str(r["value"])))
 means={}
 for k,v in sorted(d.items()):
  v=sorted(v); cut=max(1,len(v)//10); kept=v[cut:-cut]; means[k]=q(sum(kept)/len(kept))
 vals=[Decimal(x) for x in means.values()]; ans={"trimmed_means":means,"spread":q(max(vals)-min(vals))}
elif op=="interval_overlap":
 ov=[]
 for a,b in zip(sorted(rows,key=lambda x:x["start"]),sorted(rows,key=lambda x:x["start"])[1:]):
  n=max(0,min(a["end"],b["end"])-max(a["start"],b["start"]));
  if n:ov.append({"left":a["id"],"right":b["id"],"duration":n})
 ans={"overlaps":ov,"total_overlap":sum(x["duration"] for x in ov)}
elif op=="policy_filter":
 ans={"violations":sorted(r["id"] for r in rows if r["granted"] not in I["allowed_by_role"][r["role"]])}
elif op=="deduplicate_latest":
 d={}
 for r in rows:
  if r["entity"] not in d or r["recorded_at"]>d[r["entity"]]["recorded_at"]:d[r["entity"]]=r
 ans={"survivors":{k:d[k]["value"] for k in sorted(d)}}
elif op=="greedy_capacity":
 rem=I["capacity"]; chosen=[]
 for r in sorted(rows,key=lambda x:(-x["priority"],x["id"])):
  if r["demand"]<=rem:chosen.append(r["id"]); rem-=r["demand"]
 ans={"selected":chosen,"unused_capacity":rem}
elif op=="deadline_status":
 asof=datetime.fromisoformat(I["as_of"]); ans={"statuses":{r["id"]:("breached" if datetime.fromisoformat(r["deadline"])<asof else "due-soon" if (datetime.fromisoformat(r["deadline"])-asof).total_seconds()<=86400 else "on-time") for r in sorted(rows,key=lambda x:x["id"])}}
elif op=="topological_order":
 nodes=sorted({r["node"] for r in rows}|{d for r in rows for d in r["depends_on"]}); deps={n:set() for n in nodes}
 for r in rows:deps[r["node"]]|=set(r["depends_on"])
 out=[]
 while deps:
  ready=sorted(n for n,v in deps.items() if not v)
  if not ready:raise RuntimeError("cycle")
  n=ready[0];out.append(n);del deps[n]
  for v in deps.values():v.discard(n)
 ans={"order":out}
elif op=="route_distance":
 edges={(r["from"],r["to"]):r["distance"] for r in rows}; route=I["route"]; missing=[]; total=0
 for a,b in zip(route,route[1:]):
  if (a,b) not in edges:missing.append(f"{a}->{b}")
  else:total+=edges[(a,b)]
 ans={"distance":q(total),"missing_legs":missing}
elif op=="join_reconcile":
 d=defaultdict(dict)
 for r in rows:d[r["key"]][r["side"]]=Decimal(str(r["amount"]))
 rec={};exc=[]
 for k in sorted(d):
  v=q(d[k].get("ledger",0)-d[k].get("control",0));rec[k]=v
  if abs(Decimal(v))>Decimal(str(I["tolerance"])):exc.append(k)
 ans={"variance":rec,"exceptions":exc}
else: raise RuntimeError(op)
if variant=="recovery": audit={"corrected_prior_fields":sorted(set(I["prior_answer"])-set(ans))}
ans["audit"]=audit
(R/"answer.json").write_text(json.dumps(ans,sort_keys=True,indent=2)+"\n")
