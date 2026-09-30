import json
FPS=30; RATE=0.5333; TOTAL=3727; MAXEXT=0.60
cuts=sorted(float(x) for x in open('work2/cuts.txt'))
tx=json.load(open('work2/transcript.json'))
rows=json.load(open('work2/plan.json'))
def shot(t):
    return (max([c for c in cuts if c<=t],default=0.0),
            min([c for c in cuts if c> t],default=TOTAL/FPS))
print(f"{'#':3}{'spk':7}{'dialogue':>16}{'->':>3}{'extended':>16}{'fr':>6}{'cr':>6}  guard")
tot=0; out=[]
for r in rows:
    a,b=r['dialogue']; spk=r['spk']
    lo,hi=shot((a+b)/2)
    # never cross another voice: stop at the neighbouring transcript lines
    prev=max([t['end'] for t in tx if t['end']<=a+0.01],default=0.0)
    nxt =min([t['start'] for t in tx if t['start']>=b-0.01],default=TOTAL/FPS)
    na=max(a-MAXEXT, prev, lo)
    nb=min(b+MAXEXT, nxt, hi)
    sf,ef=int(na*FPS),int(nb*FPS)-1; n=ef-sf+1; tot+=n
    guard=[]
    if na==prev: guard.append("prev-voice")
    if nb==nxt: guard.append("next-voice")
    if na==lo: guard.append("shot-in")
    if nb==hi: guard.append("shot-out")
    # LAYER 1 assertion: nothing but this segment's own dialogue inside the window
    bad=[t['text'][:28] for t in tx if t['end']>na+0.02 and t['start']<nb-0.02
         and not (t['start']>=a-0.01 and t['end']<=b+0.01)]
    out.append({**r,"final":[sf,ef],"frames":n,"window":[round(na,2),round(nb,2)],
                "contamination":bad})
    flag="" if not bad else "!! "+str(bad)
    print(f"{r['n']:<3}{spk:7}{a:7.2f}-{b:<8.2f}{'->':>3}{na:7.2f}-{nb:<8.2f}{n:6d}{n*RATE:6.0f}  {','.join(guard)}{flag}")
print(f"\nC extended: {tot} frames = {tot/FPS:.1f}s = {tot*RATE:.0f} credits (${tot*RATE/100:.2f})")
bad=sum(1 for o in out if o['contamination'])
print(f"LAYER 1 (audio purity): {len(out)-bad}/{len(out)} segments clean")
json.dump(out,open('work2/plan_final.json','w'),indent=1)
