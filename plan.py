import json
FPS=30; RATE=0.5333  # credits per frame, measured from 98 past generations
cuts=sorted(float(x) for x in open('metadata/cuts.txt'))
def shot(t):
    lo=max([c for c in cuts if c<=t], default=0.0)
    hi=min([c for c in cuts if c> t], default=118.0)
    return lo,hi
# on-screen character dialogue only (narrator VO, phone woman and SFX excluded)
SEGS=[("s02",21.33,22.60,"Put this anywhere and people get power"),
      ("s03",29.24,30.58,"Did I just make scarcity obsolete?"),
      ("s04",56.52,57.74,"That wasn't me / I was here all night"),
      ("s05",58.82,59.24,"Who is this?"),
      ("s06",72.62,72.84,"Why?"),
      ("s07",78.30,78.70,"Who's coming for me?"),
      ("s08",79.42,79.64,"Who?"),
      ("s09",108.98,111.02,"Someone is willing to kill me..."),
      ("s10",113.47,114.40,"But I'm gonna make sure you do")]
MIN=30
out=[];tot=0
for sid,a,b,txt in SEGS:
    lo,hi=shot((a+b)/2)
    sf,ef=int(a*FPS),int(b*FPS)-1
    n=ef-sf+1
    if n<MIN:                                   # pad inside the shot, never across a cut
        need=MIN-n; l=need//2; r=need-l
        sf=max(int(lo*FPS), sf-l); ef=min(int(hi*FPS)-1, ef+r); n=ef-sf+1
    cr=n*RATE; tot+=n
    out.append({"id":sid,"start_frame":sf,"end_frame":ef,"frames":n,
                "sec":round(n/FPS,2),"credits":round(cr,1),"text":txt,
                "shot":[round(lo,2),round(hi,2)]})
print(f"{'id':5}{'frames':>14}{'sec':>7}{'cr':>7}  shot            line")
for o in out:
    print(f"{o['id']:5}{o['start_frame']:6d}-{o['end_frame']:<6d}{o['sec']:>7}{o['credits']:>7}  {str(o['shot']):15} {o['text'][:38]}")
print(f"\nNEW: {len(out)} segments, {tot} frames, {tot/FPS:.2f}s, {tot*RATE:.0f} credits (${tot*RATE/100:.2f})")
print(f"DONE already: seg001 110 frames, 59 credits")
print(f"Full video if sent whole: 3540 frames = {3540*RATE:.0f} credits (${3540*RATE/100:.2f})")
print(f"Saving: {(1-(tot+110)/3540)*100:.1f}%")
json.dump(out,open('metadata/segments.json','w'),indent=1)
