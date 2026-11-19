"""Full streaming audit for the IVA dataset registry. Aggregate-only output; never saves source rows."""
import argparse, hashlib, json, re, time
from pathlib import Path
from collections import Counter
from datasets import load_dataset

EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{8,}\d)(?!\w)")
CYR = re.compile(r"[А-Яа-яЁё]")
DEFAULTS = {
 "d0rj_math_dataset_ru":("default","ru"), "s85io_math_logic_ru":("default","train"),
 "openai_gsm8k_math":("main","train"), "ttech_tmath_ru_eval":("default","train")
}
def flatten(v):
 if isinstance(v,str): return v
 if isinstance(v,dict): return " ".join(flatten(x) for x in v.values())
 if isinstance(v,(list,tuple)): return " ".join(flatten(x) for x in v)
 return "" if v is None else str(v)
def run_one(entry, cap):
 repo=entry["hf_id"]; cfg,split=DEFAULTS.get(entry["id"],("","train"))
 kwargs={"split":split,"streaming":True}
 if cfg: kwargs["name"]=cfg
 stream=load_dataset(repo,**kwargs)
 n=empty=chars=letters=cyr=emails=phones=dups=0
 seen=set(); fields={}; lang=Counter(); start=time.time()
 for row in stream:
  if cap and n>=cap: break
  n+=1
  if not isinstance(row,dict): row={"text":row}
  if not row or not any(v not in (None,"",[],{}) for v in row.values()): empty+=1
  for k,v in row.items(): fields.setdefault(str(k),set()).add(type(v).__name__)
  l=row.get("language",row.get("lang",row.get("locale","")))
  if l: lang[str(l).lower()]+=1
  txt=flatten(row); chars+=len(txt)
  alpha=[x for x in txt if x.isalpha()]; letters+=len(alpha); cyr+=sum(bool(CYR.fullmatch(x)) for x in alpha)
  emails+=int(bool(EMAIL.search(txt))); phones+=int(bool(PHONE.search(txt)))
  h=hashlib.sha256(" ".join(txt.casefold().split()).encode()).digest()
  if h in seen: dups+=1
  else: seen.add(h)
 return {"id":entry["id"],"hf_id":repo,"registry_status":entry.get("status"),"declared_license":entry.get("declared_license"),"role":entry.get("role"),"config":cfg or None,"split":split,"rows_scanned":n,"row_cap":cap or None,"empty_rows":empty,"exact_duplicate_rows":dups,"possible_email_rows":emails,"possible_phone_rows":phones,"cyrillic_share":round(cyr/letters,4) if letters else None,"avg_chars":round(chars/n,1) if n else 0,"field_types":{k:sorted(v) for k,v in sorted(fields.items())},"language_counts":dict(lang),"duration_seconds":round(time.time()-start,1),"raw_text_saved":False,"training_approval_changed":False,"status":"full_stream_scanned" if not cap else "stream_scanned_with_cap","limitations":["Streaming scan only; no source text or cleaned derivatives retained.","Heuristic duplicate/PII counters are not complete guarantees.","Rights, factual correctness, safety and semantic quality are not certified.","Registry hold/excluded statuses remain unchanged; no training manifest modified."]}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--registry",default="data/dataset_candidates.json"); p.add_argument("--max-rows",type=int,default=0); p.add_argument("--output",default="reports/full-corpus-scan.json"); p.add_argument("--ids",default=""); a=p.parse_args()
 reg=json.loads(Path(a.registry).read_text(encoding="utf8")); wanted=set(a.ids.split(",")) if a.ids else None
 out={"started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"mode":"full_stream" if a.max_rows==0 else "capped_stream","max_rows_per_source":a.max_rows or None,"registry_count":len(reg["datasets"]),"results":[]}
 for e in reg["datasets"]:
  if wanted and e["id"] not in wanted: continue
  if e.get("status") not in ("candidate","hold"):
   out["results"].append({"id":e["id"],"status":"skipped_registry_status","raw_text_saved":False}); continue
  try: out["results"].append(run_one(e,a.max_rows))
  except Exception as ex: out["results"].append({"id":e["id"],"hf_id":e.get("hf_id"),"status":"scan_failed","error_type":type(ex).__name__,"error":str(ex)[:400],"raw_text_saved":False})
  print("completed",e["id"],out["results"][-1]["status"],flush=True)
 out["completed_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
 out["success_count"]=sum(x.get("status") in ("full_stream_scanned","stream_scanned_with_cap") for x in out["results"])
 out["failed_count"]=sum(x.get("status")=="scan_failed" for x in out["results"])
 Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
 print(json.dumps({"success":out["success_count"],"failed":out["failed_count"],"registry":out["registry_count"]}))
if __name__=="__main__": main()
