{
const normName=s=>String(s??"").normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase().replace(/\s+(county|parish|borough|census area|city and borough|municipality)$/,"").replace(/[^a-z0-9]/g,"");
const unitKey=s=>normName(s).replace(/^saint/,"st");
const partyCode = p => { const u=(p||"").toLowerCase().trim().replace(/\.$/,""); if(!u) return "?"; if(u==="d"||u==="dem"||u.startsWith("democrat")) return "D"; if(u==="r"||u==="rep"||u==="gop"||u.startsWith("republican")) return "R"; if(u==="l"||u.startsWith("lib")) return "L"; if(u==="g"||u.startsWith("gre")) return "G"; if(u==="i"||u.startsWith("ind")||u.startsWith("npa")||u.startsWith("nonpart")||u.startsWith("unaffil")||u.startsWith("no party")) return "I"; if(u==="y"||u==="yes") return "Y"; if(u==="n"||u==="no") return "N"; return "?"; };
/*PROJ_ENGINE_START*/
// ═══ County-level projection engine (shared by the page and the backtest) ═══
// Method after the Washington Post's election-night model: learn each county's change from its baseline ONLY from
// counties that have finished counting, predict the outstanding counties from their own features, size the ranges
// from the real misses of finished counties (conformal-style, turnout-weighted), and add counties up cautiously.
const PROJ_DEFAULTS={
  doneRpt:95,          // a county counts as finished at this % reporting (late-mail states use doneRptLate)
  doneRptLate:99,
  minDone:15,          // finished counties needed (nationally) before the night's swing is trusted at all
  k0:60,               // weight of the pre-election prior on the national intercept (in "average counties")
  kState:25, kRace:15, // shrinkage of state / race effects toward the national fit
  ridge:4,             // shrinkage of feature slopes toward 0
  wPow:0.75,           // county weight = expected vote ^ wPow (1 = pure turnout weighting)
  priorSd:3,           // sd (pts) of the pre-election national environment
  stateSd:2,         // sd (pts) of a state's own deviation before its counties report
  raceSd:2,            // sd (pts) of a race's own deviation (candidates) before its counties report
  raceMix:[[0.8,2],[0.17,9],[0.03,40]], // heavy-tailed prior on a race's own deviation: most races sit near the national/state swing, a few candidates run far ahead or behind (Phil Scott, Josh Stein, Larry Hogan)
  conf:1.0,            // multiplier on the conformal county-error quantile
  partialBias:8,       // how far (pts) a partly counted county's early batch can sit from its final result
  partialFloor:1.5,    // smallest allowance (pts) for a nearly finished county's last batch
  q:0.8                // nominal range (80%)
};
const PROJ_LATE_MAIL=new Set(["PA","WI","MI","NV","AZ","CA","OR","WA","UT","CO","NY","NJ","MD"]); // big mail shares or mail counted late/slow
const _clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function _solve(A,b){ const n=b.length, M=A.map((r,i)=>[...r,b[i]]);
  for(let i=0;i<n;i++){ let p=i; for(let k=i+1;k<n;k++) if(Math.abs(M[k][i])>Math.abs(M[p][i])) p=k; [M[i],M[p]]=[M[p],M[i]]; const d=M[i][i]||1e-9;
    for(let k=i+1;k<n;k++){ const f=M[k][i]/d; for(let j=i;j<=n;j++) M[k][j]-=f*M[i][j]; } }
  const x=new Array(n).fill(0); for(let i=n-1;i>=0;i--){ let s=M[i][n]; for(let j=i+1;j<n;j++) s-=M[i][j]*x[j]; x[i]=s/(M[i][i]||1e-9); } return x; }
const _feat=u=>[1,(u.b||0)/50,(u.s1==null?0:u.s1)/15,(u.s2==null?0:u.s2)/15,Math.log(Math.max(50,u.E||50))/10-1];
function _wq(vals,ws,q){ const idx=vals.map((v,i)=>i).sort((a,b)=>vals[a]-vals[b]); const W=ws.reduce((a,b)=>a+b,0)||1; let c=0; for(const i of idx){ c+=ws[i]; if(c>=q*W) return vals[i]; } return vals[idx[idx.length-1]]||0; }
const _ncdf=x=>{ const t=1/(1+0.2316419*Math.abs(x)), d=0.3989423*Math.exp(-x*x/2), p=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274)))); return x>0?1-p:p; };
// posterior of a race's own deviation δ given an estimate d with noise sd sd, under the mixture prior
function _mixPost(d,sd,mix){ if(!(sd<1e6)) { const v=mix.reduce((a,[w,s])=>a+w*s*s,0); return {m:0,sd:Math.sqrt(v)}; }
  const cs=mix.map(([w,s])=>{ const tv=s*s+sd*sd; const like=w*Math.exp(-d*d/(2*tv))/Math.sqrt(tv); const v=1/(1/(s*s)+1/(sd*sd)); return {like,m:v*d/(sd*sd),v}; });
  const Z=cs.reduce((a,c)=>a+c.like,0)||1; const m=cs.reduce((a,c)=>a+c.like*c.m,0)/Z; const v=cs.reduce((a,c)=>a+c.like*(c.v+c.m*c.m),0)/Z-m*m; return {m,sd:Math.sqrt(Math.max(0,v))}; }
// contests: [{id, st, units:[{c,b,s1,s2,E, C,Dv,Rv,rpt, bErr?}], ...}]  bErr: sd (pts) of an estimated baseline (new district lines: a county's
//   piece of a district is estimated) — such places don't teach the night's swing, and their own predictions carry that extra error
//   whole: the unit is a whole race (no county list) — it reads its own race but doesn't teach the night's county swing (one heavy unit
//   carrying a candidate's own pull would tilt every other race in its state)  prior: {mu} (D-minus-R swing vs baseline, pts)
function projectNight(contests,prior,params){
  const P={...PROJ_DEFAULTS,...(params||{})}; const mu=prior&&isFinite(prior.mu)?prior.mu:0;
  const isDone=(u,st)=>u.C>0&&u.rpt>=(PROJ_LATE_MAIL.has(st)?P.doneRptLate:P.doneRpt);
  // 1) learning set: finished counties with a baseline
  const L=[]; contests.forEach((ct,ci)=>ct.units.forEach(u=>{ if(u.b!=null&&u.E>0&&isDone(u,ct.st)&&!(u.bErr>0)&&!u.whole){ const m=100*(u.Dv-u.Rv)/u.C; L.push({ci,st:ct.st,x:_feat(u),y:m-u.b,t:Math.log(u.C/u.E),w:Math.pow(u.E,P.wPow)}); } }));
  const nDone=L.length, trusted=nDone>=P.minDone;
  const fit=(key,priorMean,passes)=>{ const p=5; const wbar=L.length?L.reduce((a,l)=>a+l.w,0)/L.length:1; let off=L.map(()=>0), st={}, rc={}, stW={}, rcW={}, beta=[priorMean,0,0,0,0], res=L.map(()=>0);
    for(let pass=0;pass<(passes||1);pass++){
      const A=[...Array(p)].map(()=>new Array(p).fill(0)), bv=new Array(p).fill(0);
      L.forEach((l,k)=>{ const yv=l[key]-off[k]; for(let i=0;i<p;i++){ bv[i]+=l.w*l.x[i]*yv; for(let j=0;j<p;j++) A[i][j]+=l.w*l.x[i]*l.x[j]; } });
      A[0][0]+=P.k0*wbar; bv[0]+=P.k0*wbar*priorMean; for(let i=1;i<p;i++) A[i][i]+=P.ridge*wbar*10;
      beta=_solve(A,bv); res=L.map(l=>l[key]-l.x.reduce((q,v,i)=>q+v*beta[i],0));
      const stE={}; stW={}; L.forEach((l,i)=>{ stE[l.st]=(stE[l.st]||0)+l.w*(res[i]-(rc[l.ci]||0)); stW[l.st]=(stW[l.st]||0)+l.w; });
      st={}; Object.keys(stE).forEach(q=>{ st[q]=stE[q]/(stW[q]+P.kState*wbar); });
      const rcE={}, rcW2={}; rcW={}; L.forEach((l,i)=>{ const r=res[i]-(st[l.st]||0); rcE[l.ci]=(rcE[l.ci]||0)+l.w*r; rcW[l.ci]=(rcW[l.ci]||0)+l.w; rcW2[l.ci]=(rcW2[l.ci]||0)+l.w*l.w; });
      rc={}; if(passes>1){ const sc=pass===0?6:Math.max(1.5,_wq(L.map((l,i)=>Math.abs(res[i]-(st[l.st]||0)-(rc0[l.ci]||0))),L.map(l=>l.w),0.6827));
        Object.keys(rcE).forEach(c=>{ const W=rcW[c], neff=W*W/rcW2[c]; rc[c]=_mixPost(rcE[c]/W,sc/Math.sqrt(neff),P.raceMix).m; }); }
      else Object.keys(rcE).forEach(c=>{ rc[c]=rcE[c]/(rcW[c]+P.kRace*wbar); });
      var rc0=rc; off=L.map(l=>(st[l.st]||0)+(rc[l.ci]||0)); }
    const fin=L.map((l,i)=>res[i]-(st[l.st]||0)-(rc[l.ci]||0));
    return {beta,st,rc,stW,rcW,fin,wbar,sumW:L.reduce((a,l)=>a+l.w,0)}; };
  const S=fit("y",mu,3), Tn=fit("t",Math.log(0.72),1);
  // conformal-style county error: turnout-weighted quantile of finished counties' leftover misses
  const qz=P.q===0.8?1.2816:1.6449;
  const nEff=Math.max(1,nDone); const qCounty=trusted?P.conf*_wq(S.fin.map(Math.abs),L.map(l=>l.w),P.q)*(1+5/nEff):P.conf*8;
  const qTurn=trusted?_wq(Tn.fin.map(Math.abs),L.map(l=>l.w),P.q)*(1+5/nEff):0.25;
  // shared (correlated) uncertainty: national intercept, then a state's and a race's own effect, shrinking as their counties finish
  const resSd=Math.max(1.5,(trusted?_wq(S.fin.map(Math.abs),L.map(l=>l.w),0.6827):8));
  // how much of the night's expected vote has finished — the shared (national) uncertainty shrinks with that, not with a count of small counties
  let eAll=0,eDone=0; const stAll={},stDone={}; contests.forEach(ct=>{ const seen=new Set(); ct.units.forEach(u=>{ const k=ct.st+'|'+u.c; if(seen.has(k)||!(u.E>0)) return; seen.add(k); eAll+=u.E; stAll[ct.st]=(stAll[ct.st]||0)+u.E; if(isDone(u,ct.st)){ eDone+=u.E; stDone[ct.st]=(stDone[ct.st]||0)+u.E; } }); });
  const fNat=eAll?eDone/eAll:0;
  const natSd=Math.sqrt((P.priorSd*(1-fNat))**2+(trusted?resSd*resSd/Math.max(1,nDone):P.priorSd*P.priorSd));
  const out=contests.map((ct,ci)=>{
    const stW=(S.stW[ct.st]||0)/S.wbar, rcW=(S.rcW[ci]||0)/S.wbar;
    const fSt=stAll[ct.st]?(stDone[ct.st]||0)/stAll[ct.st]:0; const stateSd=P.stateSd*(1-fSt)+P.stateSd*0.25*Math.sqrt(P.kState/(P.kState+stW));
    // the race's own deviation, read from everything it has counted so far — finished counties and partly counted ones
    // (a partly counted county's early batch can lean either way, so its reading carries that extra, county-to-county-correlated noise)
    let dW=0,dS=0,dB=0,dW2=0,dE2=0; const scC=Math.max(1.5,qCounty/qz);
    ct.units.forEach(u=>{ if(u.fixed||!(u.C>0)||u.b==null||!(u.E>0)) return; const x=_feat(u); const pred=u.b+x.reduce((q,v,i)=>q+v*S.beta[i],0)+(S.st[ct.st]||0);
      const th=x.reduce((q,v,i)=>q+v*Tn.beta[i],0)+(Tn.st[ct.st]||0); const tot=Math.max(u.C,u.E*Math.exp(th)); const pf=isDone(u,ct.st)?1:u.C/tot;
      const w=u.C; dW+=w; dW2+=w*w; dE2+=w*w*(u.bErr||0)**2; dS+=w*(100*(u.Dv-u.Rv)/u.C-pred); dB+=w*P.partialBias*(1-pf); });
    const dPost=dW>0?_mixPost(dS/dW,Math.sqrt((dB/dW)**2+(scC*scC*dW2+dE2)/(dW*dW)),P.raceMix):_mixPost(0,Infinity,P.raceMix);
    const rcHat=dPost.m, raceSd=dPost.sd;
    let cD=0,cR=0,cC=0, remM=0, remV=0, remErr=0, remTurnErr=0, noBase=0, unbE=0, hard=0; const outs=[];
    // places with no baseline on file get the race's average place (flagged, wider)
    const kn=ct.units.filter(u=>u.b!=null&&u.E>0); const kE=kn.reduce((a,u)=>a+u.E,0); const avgB=kE?kn.reduce((a,u)=>a+u.b*u.E,0)/kE:null; const avgE=kn.length?kE/kn.length:null;
    ct.units.forEach(u0=>{ let u=u0; cD+=u.Dv||0; cR+=u.Rv||0; cC+=u.C||0;
      if(u.fixed||isDone(u,ct.st)) return; // fixed: a place listed with the race that isn't expected to add votes (feed lists stray counties) — count what it reports, expect nothing more
      if(u.b==null||!(u.E>0)){ if(avgB==null){ noBase++; hard++; return; } u={...u,b:u.b==null?avgB:u.b,E:u.E>0?u.E:avgE,s1:null,s2:null}; noBase+=0.5; unbE+=u.E; }
      const x=_feat(u); const yh=x.reduce((s,v,i)=>s+v*S.beta[i],0)+(S.st[ct.st]||0)+rcHat;
      const th=x.reduce((s,v,i)=>s+v*Tn.beta[i],0)+(Tn.st[ct.st]||0)+(Tn.rc[ci]||0);
      const tot=Math.max(u.C||0,u.E*Math.exp(th)); const R=Math.max(0,tot-(u.C||0));
      // predict the county's FINAL margin, then the outstanding votes are whatever it takes to get there from what's counted
      // (so a partly counted county's early batch — mail-first or in-person-first — doesn't get projected onto the rest)
      const mModel=_clamp(u.b+yh,-100,100); const p=(u.C||0)/tot;
      // blend with what this county has counted so far: more counted → trust its own count more (its early batch may be skewed)
      const sdM=Math.sqrt(Math.max(1,qCounty/qz)**2+(u.bErr||0)**2), sdP=p>0?Math.max(P.partialFloor,(1-p)*P.partialBias)/Math.max(0.05,Math.sqrt(p)):1e9;
      const mObs=(u.C||0)>0?100*((u.Dv||0)-(u.Rv||0))/u.C:0; const wP=p>0?(1/(sdP*sdP))/(1/(sdP*sdP)+1/(sdM*sdM)):0;
      const mf=wP*mObs+(1-wP)*mModel, sdF=1/Math.sqrt(1/(sdM*sdM)+(p>0?1/(sdP*sdP):0));
      const need=tot*mf/100-((u.Dv||0)-(u.Rv||0)); const rm=_clamp(need,-R,R);
      remM+=rm; remV+=R; remErr+=tot*qz*sdF/100; remTurnErr+=R*(Math.exp(qTurn)-1); if(R>0) outs.push({c:u0.c,n:u0.n,R,mf,net:rm}); });
    const total=cC+remV; const unbShare=total>0?unbE/total:0; if(!(total>0)||hard>0||(noBase>=1&&unbShare>0.15&&cC<0.95*total)) return {id:ct.id,m:null}; // a place still counting with nothing to compare it to: no projection (leave it to the simpler estimate) // places with no baseline at all → leave this race to the simpler estimate
    const m=100*((cD-cR)+remM)/total, remShare=remV/total;
    const indH=100*remErr/total;                               // comonotonic sum of county ranges (cautious)
    const sharedSd=remShare*Math.sqrt(natSd*natSd+stateSd*stateSd+raceSd*raceSd);
    const H=Math.sqrt(indH*indH+(qz*sharedSd)**2)+Math.min(2,20*unbShare); // places guessed from the race's average place widen the range by how much of the vote they hold
    const sdTot=Math.max(0.05,H/qz); const pD=remV<=0?(m>0?1:m<0?0:0.5):_ncdf(m/sdTot);
    return {id:ct.id,m,lo:m-H,hi:m+H,H,pD,remShare,counted:cC,expected:total,natPart:remShare*natSd,statePart:remShare*stateSd,st:ct.st,trusted,noBase,raceEff:rcHat,raceSd,top:outs.sort((a,b)=>b.R-a.R).slice(0,5),nUnits:ct.units.length,nDoneUnits:ct.units.filter(u=>isDone(u,ct.st)).length};
  });
  const obsW=L.reduce((s,l)=>s+l.w,0); const obs=obsW?L.reduce((s,l)=>s+l.w*l.y,0)/obsW:null;
  return {races:out,nDone,trusted,swing:S.beta[0],obs,natSd,stateEff:S.st,stateDone:stDone,stateAll:stAll,fNat,qCounty,mu};
}
// seat ranges: correlated draws (national, state, race) — q10..q90 of seat counts per group
function projectSeats(res,groups,N=2000){ const rng=(()=>{ let s=987654321; return ()=>{ s=(s*1103515245+12345)%2147483648; return s/2147483648; }; })();
  const g=()=>{ let u=0,v=0; while(!u) u=rng(); while(!v) v=rng(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };
  const byId=Object.fromEntries(res.races.filter(r=>r.m!=null).map(r=>[r.id,r])); const out={};
  Object.entries(groups).forEach(([name,ids])=>{ const d=[]; const states=[...new Set(ids.map(i=>byId[i]&&byId[i].st).filter(Boolean))];
    for(let k=0;k<N;k++){ const zn=g(); const zs=Object.fromEntries(states.map(s=>[s,g()])); let D=0,R=0;
      ids.forEach(i=>{ const r=byId[i]; if(!r) return; const shared=r.natPart*zn+r.statePart*zs[r.st]; const ind=Math.sqrt(Math.max(0,(r.H/1.2816)**2-r.natPart**2-r.statePart**2)); const v=r.m+shared+ind*g(); if(v>0) D++; else R++; }); d.push(D); }
    d.sort((a,b)=>a-b); out[name]={d:[d[Math.floor(0.1*(N-1))],d[Math.floor(0.9*(N-1))]],dMed:d[Math.floor(0.5*(N-1))],n:ids.filter(i=>byId[i]).length}; });
  return out; }
if(typeof module!=="undefined"&&module.exports) module.exports={projectNight,projectSeats,PROJ_DEFAULTS,PROJ_LATE_MAIL};
/*PROJ_ENGINE_END*/
const UNIT_ALIAS={SD:{shannon:"oglalalakota"}}; // county renamed; the feed and the files don't always use the same name
const UNIT_MERGE={IL:{chicago:"cook"}}; // the feed lists the city of Chicago apart from the rest of Cook County in House races; the 2024 files have Cook whole
function modelUnits(r,det,cf,base24,fy){ const st=r.abbr, F=(cf&&cf[st])||{}; const cd=r.type==="House"?cdOf(r):null; const piece=cd&&cdBase(r,base24);
  const canon=k=>(UNIT_ALIAS[st]&&UNIT_ALIAS[st][k])||k;
  const pieceBy={}; if(piece&&piece.c){ const nm=base24[st].n||{}; Object.entries(piece.c).forEach(([f,[d,rr,t,be]])=>{ pieceBy[canon(unitKey(nm[f]||f))]={b:t?100*(d-rr)/t:0,E:t,bErr:be||0}; }); }
  const regs=det&&det.regions&&det.regions.length&&!multiParty(r)?det.regions:null; /* county rows carry only each county's top two */
  if(regs){ // the results feed's region lists vary by state and year: counties plus towns, old CT counties beside planning regions, districts or towns typed as counties
    const look=n=>{ const k0=unitKey(n); const ks=[UNIT_MERGE[st]&&UNIT_MERGE[st][k0],canon(k0),unitKey(String(n).replace(/\s+city$/i,"")),unitKey(String(n).replace(/&/g," and ")),unitKey(n+" planning region")].filter(Boolean); for(const k of ks) if(F[k]) return [k,F[k]]; return [canon(k0),null]; };
    const sets=[regs.filter(g=>!/town|planning|state/i.test(g.type||"")),regs.filter(g=>/planning/i.test(g.type||"")),regs.filter(g=>!/state|planning/i.test(g.type||""))].filter(a=>a.length);
    const score=a=>new Set(a.map(g=>look(g.name)).filter(x=>x[1]).map(x=>x[0])).size; // distinct places matched (towns named like counties don't count twice)
    const use=r.type==="House"?(sets[0]||[]):(sets.slice().sort((a,b)=>score(b)-score(a))[0]||[]);
    const house=r.type==="House", atLarge=cd==="AL";
    let units=use.map(g=>{ const k0=unitKey(g.name), merged=UNIT_MERGE[st]&&UNIT_MERGE[st][k0]; const [k,f]=look(g.name); const pc=pieceBy[k]||pieceBy[canon(k0)];
      let b=null,E=null,bErr=0; if(house){ if(pc){ b=pc.b; E=pc.E; bErr=pc.bErr||0; } else if(f&&atLarge){ b=f[0]; E=f[3]; } } else if(f){ b=f[0]; E=f[3]; } // a district inside one county is not the whole county
      return {c:merged||(house&&pc?(pieceBy[k]?k:canon(k0)):k),n:g.name,b,...(bErr?{bErr}:{}),s1:f?f[1]:null,s2:f?f[2]:null,E,C:g.total||0,Dv:g.D||0,Rv:g.R||0,rpt:g.rpt||0,merged:!!merged}; });
    // same place listed twice → keep the fuller one; a city listed apart from its county → add them together
    const by=new Map(); units.forEach(u=>{ const o=by.get(u.c); if(!o){ by.set(u.c,u); return; }
      if(u.merged||o.merged) by.set(u.c,{...o,n:o.merged?u.n:o.n,C:(o.C||0)+(u.C||0),Dv:(o.Dv||0)+(u.Dv||0),Rv:(o.Rv||0)+(u.Rv||0),rpt:Math.min(o.rpt||0,u.rpt||0),merged:false});
      else if((u.C||0)>(o.C||0)) by.set(u.c,u); }); units=[...by.values()];
    // every place on file present → anything else listed is another geography or a county outside the district, not extra votes
    if(!house){ const nF=Object.keys(F).length; if(nF&&units.filter(u=>u.b!=null).length===nF) units=units.filter(u=>u.b!=null); }
    else if(piece&&piece.c){ const nP=Object.keys(pieceBy).length; if(units.filter(u=>pieceBy[u.c]).length>=nP) units=units.filter(u=>u.b!=null);
      else units=units.map(u=>u.b==null?{...u,fixed:true}:u); } // a county the district's 2024 pieces don't include: count what it reports, expect nothing more from it
    // the counties must add up to the race: if they hold far more or fewer votes than the race total, the list is something else (seen in past feeds)
    const sumC=units.reduce((a,u)=>a+(u.C||0),0), raceC=det&&det.total!=null?det.total:counted(r); /* same fetch as the county list */ const fits=!(raceC>2000)||(sumC>=0.93*raceC&&sumC<=1.07*raceC);
    const based=units.filter(u=>u.b!=null&&!u.fixed).length, nFixed=units.filter(u=>u.fixed).length; // a list that's mostly something else (e.g. San Francisco by supervisor district) → don't trust it
    if(units.length&&fits&&based>0&&nFixed<=based) return units; }
  // no county breakdown yet (or an all-party race, followed by party totals): the whole race as one unit (state or district baseline)
  const b=baselineDR(r,base24); const ex=expectedVote(r,base24,fy); const E=ex&&ex.E||(piece&&piece.p?piece.p[2]:null);
  const mp=multiParty(r); return [{c:"all",whole:true,b,s1:null,s2:null,E,C:counted(r),Dv:mp?partyVotes(r,"D"):votesOf(r,"D"),Rv:mp?partyVotes(r,"R"):votesOf(r,"R"),rpt:r.rpt}]; }
const cdOf=r=>{ if(!r||r.type!=="House") return null; const d=String(r.district||"").trim(); let m=d.match(/(?:^|-)(AL|\d+)$/i); if(m) return /^al$/i.test(m[1])?"AL":String(+m[1]);
  if(/at[- ]large/i.test(r.name||"")) return "AL"; m=String(r.name||"").match(/(\d+)\s*$/); return m?String(+m[1]):null; };
const drPair=r=>(r.p1==="D"&&r.p2==="R")||(r.p1==="R"&&r.p2==="D");
const votesOf=(r,party)=>!r?0:r.p1===party?r.v1:r.p2===party?r.v2:0;
const counted=r=>(r.v1||0)+(r.v2||0)+(r.others||[]).reduce((a,o)=>a+(o.votes||0),0);
const multiParty=r=>{ const ps=[r.p1,r.p2,...(r.others||[]).map(o=>o.party)].filter(p=>p==="D"||p==="R"); return ps.length>new Set(ps).size&&ps.includes("D")&&ps.includes("R"); };
const partyVotes=(r,p)=>(r.p1===p?r.v1||0:0)+(r.p2===p?r.v2||0:0)+(r.others||[]).reduce((a,o)=>a+(o.party===p?(o.votes||0):0),0);

let NLSTATES=new Set(); const linesSince=r=>r.type==='House'&&NLSTATES.has(r.abbr)?2026:null; const newLines=r=>{ const y=linesSince(r); return !!y&&y>=2026; };
const cdBase=(r,base24)=>{ const st=base24&&base24[r.abbr]; const cd=cdOf(r); if(!st||!cd) return null; const ls=linesSince(r); return ls&&ls>=2026?(st.cd26&&st.cd26[cd])||null:(st.cd&&st.cd[cd])||null; };
const baselineDR=(r,base24)=>{ if(!base24) return null; const st=base24[r.abbr]; if(!st) return null;
  if(r.type==="House"){ const x=cdBase(r,base24); const p=x&&x.p; return p&&p[2]?100*(p[0]-p[1])/p[2]:null; } if((linesSince(r)||0)>2024) return null;
  if(r.level==="Federal"||r.level==="Statewide"){ const p=st.pres; return p&&p[2]?100*(p[0]-p[1])/p[2]:null; } return null; };
const expectedVote=()=>null;
function parseRegions(raw){ return Object.entries(raw.region_results||{}).map(([key,reg])=>{ const cs=[...(reg.candidates||[])].sort((a,b)=>(b.votes||0)-(a.votes||0)); const c1=cs[0]||{},c2=cs[1]||{};
  const fD=cs.filter(c=>partyCode(c.party)==="D").reduce((a,c)=>a+(c.votes||0),0), fR=cs.filter(c=>partyCode(c.party)==="R").reduce((a,c)=>a+(c.votes||0),0);
  const p1=partyCode(c1.party),p2=partyCode(c2.party); const tot=cs.reduce((s,c)=>s+(c.votes||0),0);
  return {key,name:reg.name||key,type:reg.type||"",total:tot,D:p1==="D"?c1.votes||0:p2==="D"?c2.votes||0:0,R:p1==="R"?c1.votes||0:p2==="R"?c2.votes||0:0,Dall:fD,Rall:fR}; }); }
window.__rh6=(async function(ids,seeds,prior,opts){ opts=opts||{}; NLSTATES=new Set(opts.newLines||[]);
  let cf=await fetch('county-features.json?v=1').then(r=>r.json()); let b24=(await fetch('base-2024.json').then(r=>r.json())).states; if(opts.est){ Object.entries(opts.est).forEach(([st,cd])=>{ if(b24[st]) b24[st]={...b24[st],cd26:cd}; }); }
  if(opts.asOf2022){ // features as they stood before 2022: baseline = 2020 president (2024 minus the 2020→24 swing), trend = 2016→20
    const cf2={}; Object.entries(cf).forEach(([st,F])=>{ cf2[st]={}; Object.entries(F).forEach(([k,v])=>{ cf2[st][k]=[v[0]-(v[1]||0),v[2],null,v[3]]; }); });
    const b2={}; Object.entries(b24).forEach(([st,S])=>{ const nm=S.n||{}; const cd={}; Object.entries(S.cd||{}).forEach(([n,p])=>{ const c={}; Object.entries(p.c||{}).forEach(([f,[dd,rr,t]])=>{ const k=unitKey(nm[f]||f); const s1=(cf[st]&&cf[st][k]&&cf[st][k][1])||0; const b=t?100*(dd-rr)/t-s1:0; const s=dd+rr; const d2=(s+t*b/100)/2; c[f]=[d2,s-d2,t]; }); cd[n]={...p,c}; }); b2[st]={...S,cd}; });
    cf=cf2; b24=b2; }
  const det={}; const q=[...ids]; const w=async()=>{ while(q.length){ const id=q.shift(); try{ det[id]=await fetch(`https://civicapi.org/api/v2/race/${id}`).then(r=>r.json()); }catch(e){} } }; await Promise.all([w(),w(),w(),w(),w(),w()]);
  const nl={}; Object.values(det).forEach(raw=>{ const n0=raw.election_name||''; if(!/us house|congress/i.test(n0)||!NLSTATES.has(raw.province)) return; const seen=new Set(); Object.values(raw.region_results||{}).filter(g=>!/town|planning|state/i.test(g.type||'')).forEach(g=>{ const k=unitKey(g.name); if(seen.has(k)) return; seen.add(k); (nl[raw.province]=nl[raw.province]||{})[k]=((nl[raw.province]||{})[k]||0)+1; }); });
  const ZONE={ET:0,CT:0.06,MT:0.12,PT:0.18}; const TZ=s=>({TX:"CT",IA:"CT",AL:"CT",AZ:"MT",CO:"MT",CA:"PT"})[s]||"ET";
  const MAIL_EARLY=new Set(["AZ","NV","FL","NC","GA","OH","CO","UT","OR","WA"]);
  function rngF(seed){ let s=seed>>>0||1; return ()=>{ s^=s<<13; s>>>=0; s^=s>>17; s^=s<<5; s>>>=0; return s/4294967296; }; }
  const gauss=r=>{ let u=0,v=0; while(!u) u=r(); while(!v) v=r(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };
  const _c=(x,a,b)=>Math.max(a,Math.min(b,x));
  const issues=[]; const races=[];
  ids.forEach(id=>{ const raw=det[id]; if(!raw){ issues.push(id+' fetch failed'); return; } const st=raw.province; const nm0=raw.election_name||''; const ty=/us house|congress/i.test(nm0)?"House":/governor/i.test(nm0)?"Governor":"Senate"; if(/council|lieutenant/i.test(nm0)) return; if(ty==="House"&&(opts.skipHouse||[]).includes(raw.province)) return;
    const cands=raw.candidates||[]; const D=cands.filter(c=>partyCode(c.party)==="D").reduce((a,c)=>a+(c.votes||0),0), R=cands.filter(c=>partyCode(c.party)==="R").reduce((a,c)=>a+(c.votes||0),0), T=cands.reduce((a,c)=>a+(c.votes||0),0);
    const tag=`${st} ${ty}${ty==="House"?' '+cdOf({type:"House",district:raw.district,name:raw.election_name}):''}`;
    if(!(D>0&&R>0)){ issues.push(tag+': not a D-vs-R race (skipped by the model)'); return; }
    const regs=parseRegions(raw); const cs=[...cands].sort((a,b)=>(b.votes||0)-(a.votes||0)); const T0=cs.reduce((a,c)=>a+(c.votes||0),0)||1;
    const r={abbr:st,type:ty,district:raw.district,name:raw.election_name,rpt:100,c1:(cs[0]||{}).name,c2:(cs[1]||{}).name||"—",p1:partyCode((cs[0]||{}).party),p2:partyCode((cs[1]||{}).party),v1:(cs[0]||{}).votes||0,v2:(cs[1]||{}).votes||0,
      others:cs.slice(2).filter(c=>100*(c.votes||0)/T0>=1).map(c=>({name:c.name,party:partyCode(c.party),votes:c.votes||0,pct:100*(c.votes||0)/T0}))}; const mp=multiParty(r); if(mp) issues.push(tag+': all-party race (several candidates from one party) → party totals, whole race');
    if(!mp&&!drPair(r)){ issues.push(tag+': top two are not one D and one R (skipped by the app)'); return; }
    const u0=modelUnits(r,{regions:regs.map(g=>({name:g.name,type:g.type,rpt:100,total:g.total,D:g.D,R:g.R}))},cf,b24,null,nl);
    const keep=new Set(u0.filter(u=>!u.fixed).map(u=>u.n)); const sumKeep=regs.filter(g=>keep.has(g.name)).reduce((a,g)=>a+g.total,0);
    const nb=u0.filter(u=>u.b==null&&!u.fixed&&u.c!=="all"), fx=u0.filter(u=>u.fixed);
    if(u0.length===1&&u0[0].c==="all"&&!mp) issues.push(tag+': no usable county list → whole-race fallback ('+regs.length+' regions: '+[...new Set(regs.map(g=>g.type))].join('/')+')');
    if(nb.length) issues.push(tag+': no baseline for '+nb.map(u=>u.n).join(', '));
    if(fx.length) issues.push(tag+': stray counties (count only) '+fx.map(u=>u.n).join(', '));
    if(Math.abs(sumKeep/T-1)>0.03) issues.push(tag+`: county votes ${(100*sumKeep/T).toFixed(1)}% of race total`);
    const twoParty=regs.filter(g=>keep.has(g.name)).reduce((a,g)=>a+(g.D+g.R),0), twoAll=regs.filter(g=>keep.has(g.name)).reduce((a,g)=>a+(g.Dall+g.Rall),0); if(twoAll&&twoParty/twoAll<0.98) issues.push(tag+`: D or R not top-two in some counties (${(100*twoParty/twoAll).toFixed(1)}% of D+R votes seen)`);
    races.push({id,st,ty,tag,regs,keep,T,final:100*(D-R)/T,r,mp,kk:T>0?sumKeep/T:1}); });
  const rows=[];
  for(const seed of seeds) for(const flip of [1,-1]){
    const rg=rngF(seed*7919+(flip>0?1:2)); const timing={};
    const sizes={}; races.forEach(rc=>rc.regs.forEach(g=>{ const k=rc.st+'|'+unitKey(g.name); sizes[k]=Math.max(sizes[k]||0,g.total); }));
    const bySt={}; Object.entries(sizes).forEach(([k,v])=>{ const st=k.split('|')[0]; (bySt[st]=bySt[st]||[]).push([k,v]); });
    Object.entries(bySt).forEach(([st,arr])=>{ arr.sort((a,b)=>a[1]-b[1]); const sb=(rg()-0.5)*10; arr.forEach(([k],i)=>{ const pct=i/Math.max(1,arr.length-1);
      const fin=_c(0.08+ZONE[TZ(st)]+0.45*rg()+0.3*pct*(0.5+rg()),0.05,0.98); const start=_c(fin-0.15-0.3*rg(),0,fin-0.01);
      const sign=(MAIL_EARLY.has(st)?1:-1)*flip; timing[k]={fin,start,bias:(opts.biasScale??1)*(sign*Math.abs(4+3*gauss(rg))+sb*0.3+gauss(rg)*3)}; }); });
    for(const t of [0.15,0.25,0.35,0.5,0.65,0.8,0.9]){
      const tcs=[];
      const contests=races.map(rc=>{ let cc=0,tt=0,ca=0,ta=0; const slim=rc.regs.map(g=>{ const tm=timing[rc.st+'|'+unitKey(g.name)]; const p=t>=tm.fin?1:t<=tm.start?0:(t-tm.start)/(tm.fin-tm.start);
          const T=g.total||1; const fm=100*(g.D-g.R)/T; const pm=_c(fm+(1-p)*tm.bias,-100,100); const two=(g.D+g.R)/T; const C=T*p; if(rc.keep.has(g.name)){ cc+=C; tt+=g.total; } ca+=C; ta+=g.total;
          return {name:g.name,type:g.type,rpt:p>=1?100:Math.floor(p*100),total:Math.round(C),D:Math.max(0,Math.round(C*(two/2+pm/200))),R:Math.max(0,Math.round(C*(two/2-pm/200)))}; });
        if(!tt){ cc=ca; tt=ta; } /* whole-race races: the count moves with all of the race's places */ tcs.push(tt?cc/tt:0); const tot=Math.round(rc.T*(tt?cc/tt:0));
        const fr=tt?cc/tt:0; let rs={...rc.r,rpt:Math.round(100*fr)}; if(true){ /* every race: the whole-race unit (fallback) must see the partial count, not the final */ /* party totals counted so far, with the same early-count lean as the counties */ const lean=rc.regs.reduce((a,g)=>{ const tm=timing[rc.st+'|'+unitKey(g.name)]; const p=t>=tm.fin?1:t<=tm.start?0:(t-tm.start)/(tm.fin-tm.start); return a+(g.total*p)*(1-p)*tm.bias; },0)/Math.max(1,cc);
          const sc=x=>Math.round(x*fr); const Dp=rc.r.p1==="D"?1:0; rs={...rs,v1:sc(rc.r.v1),v2:sc(rc.r.v2),others:rc.r.others.map(o=>({...o,votes:sc(o.votes)}))}; const t2=(rs.v1+rs.v2+rs.others.reduce((a,o)=>a+o.votes,0))||1; const shift=Math.round(t2*lean/200);
          /* move the lean between the biggest D and biggest R entries */ const bump=(p,dv)=>{ if(rs.p1===p){ rs.v1=Math.max(0,rs.v1+dv); return; } if(rs.p2===p){ rs.v2=Math.max(0,rs.v2+dv); return; } const o=rs.others.find(o=>o.party===p); if(o) o.votes=Math.max(0,o.votes+dv); }; bump("D",shift); bump("R",-shift); }
        return {id:rc.id,st:rc.st,units:modelUnits(rs,{regions:slim,total:tot},cf,b24,null,nl)}; });
      const res=projectNight(contests,{mu:prior},opts.params||{}); /* as the page does: measure against the race's own total when the county rows hold a few % more (write-ins) */ res.races.forEach((x,i)=>{ const k=races[i].kk; if(x.m==null||contests[i].units.length===1&&contests[i].units[0].c==="all") return; if(k>0.9&&k<1.1&&Math.abs(k-1)>0.002){ x.m*=k; x.lo*=k; x.hi*=k; } });
      res.races.forEach((x,i)=>{ const rc=races[i]; rows.push(x.m==null?{seed,flip,t,id:rc.id,none:true,tc:tcs[i]}:{seed,flip,t,id:rc.id,ty:rc.ty,final:rc.final,m:x.m,lo:x.lo,hi:x.hi,pD:x.pD,tc:tcs[i],trusted:x.trusted}); });
    } }
  const R=rows.filter(x=>!x.none&&x.tc<0.985&&x.tc>0);
  const score=a=>{ if(!a.length) return null; return {n:a.length,counted:Math.round(100*a.reduce((s,x)=>s+x.tc,0)/a.length),cover:Math.round(100*a.filter(x=>x.final>=x.lo&&x.final<=x.hi).length/a.length),mae:+(a.reduce((s,x)=>s+Math.abs(x.m-x.final),0)/a.length).toFixed(2),half:+(a.reduce((s,x)=>s+(x.hi-x.lo)/2,0)/a.length).toFixed(2),wrong:a.filter(x=>(x.pD>0.5)!==(x.final>0)).length}; };
  const ts=[0.15,0.25,0.35,0.5,0.65,0.8,0.9];
  const out={races:races.length,issues,notProjected:rows.filter(x=>x.none&&x.tc>0&&x.tc<0.985).length+' of '+rows.filter(x=>x.tc>0&&x.tc<0.985).length,
    all:ts.map(t=>score(R.filter(x=>x.t===t))), house:ts.map(t=>score(R.filter(x=>x.t===t&&x.ty==="House"))), statewide:ts.map(t=>score(R.filter(x=>x.t===t&&x.ty!=="House")))};
  const grp={}; R.forEach(x=>{ (grp[x.seed+'|'+x.flip+'|'+x.id]=grp[x.seed+'|'+x.flip+'|'+x.id]||[]).push(x); }); let fi=0,fn=0,mv=0,stp=0; Object.values(grp).forEach(a=>{ a.sort((p,q)=>p.t-q.t); fn++; if(a[0].final>=a[0].lo&&a[0].final<=a[0].hi) fi++; for(let i=1;i<a.length;i++){ stp++; if(Math.abs(a[i].m-a[i-1].m)>(a[i-1].hi-a[i-1].lo)/2) mv++; } });
  out.firstIn=Math.round(100*fi/Math.max(1,fn)); out.jump=Math.round(100*mv/Math.max(1,stp));
  out.worst=races.map(rc=>{ const a=R.filter(x=>x.id===rc.id); return {tag:rc.tag,final:+rc.final.toFixed(1),cover:a.length?Math.round(100*a.filter(x=>x.final>=x.lo&&x.final<=x.hi).length/a.length):null,wrong:a.filter(x=>(x.pD>0.5)!==(x.final>0)).length,n:a.length}; }).filter(x=>x.cover!=null).sort((a,b)=>a.cover-b.cover).slice(0,12);
  out.rows=opts.rows?rows:undefined; out.racesMeta=races.map(r=>({id:r.id,tag:r.tag,final:r.final,ty:r.ty,whole:false}));
  return out; });

}
