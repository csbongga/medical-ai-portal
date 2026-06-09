import sys, json
src = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
d = json.load(src)
top = d["top_prediction"]
print(f"Top: {top['code']} - {top['name']} ({top['probability_pct']}%)")
for p in d["all_predictions"]:
    print(f"  {p['rank']}. [{p['code']}] {p['name']} => {p['probability_pct']}%")
