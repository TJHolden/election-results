{
const normName=s=>String(s??"").normalize("NFD").replace(/[\u0300-\u036f]/g,"").toLowerCase().replace(/\s+(county|parish|borough|census area|city and borough|municipality)$/,"").replace(/[^a-z0-9]/g,"");
const unitKey=s=>normName(s).replace(/^saint/,"st");
const partyCode = p => { const u=(p||"").toLowerCase().trim().replace(/\.$/,""); if(!u) return "?"; if(u==="d"||u==="dem"||u.startsWith("democrat")) return "D"; if(u==="r"||u==="rep"||u==="gop"||u.startsWith("republican")) return "R"; if(u==="l"||u.startsWith("lib")) return "L"; if(u==="g"||u.startsWith("gre")) return "G"; if(u==="i"||u.startsWith("ind")||u.startsWith("npa")||u.startsWith("nonpart")||u.startsWith("unaffil")||u.startsWith("no party")) return "I"; if(u==="y"||u==="yes") return "Y"; if(u==="n"||u==="no") return "N"; return "?"; };
/*PROJ_ENGINE_START*/
const PROJ_DEFAULTS={
  doneRpt:95,         
  doneRptLate:99,
  minDone:15,          // finished counties needed (nationally) before the night's swing is trusted at all
  k0:60,               // weight of the pre-election prior on the national intercept (in "average counties")
  kState:25, kRace:15,
  ridge:4,            
  wPow:0.75,          
  priorSd:3,          
  stateSd:2,         // sd (pts) of a state's own deviation before its counties report
  raceSd:2,            // sd (pts) of a race's own deviation (candidates) before its counties report
  raceMix:[[0.8,2],[0.17,9],[0.03,40]], // heavy-tailed prior on a race's own deviation: most races sit near the national/state swing, a few candidates run far ahead or behind (Phil Scott, Josh Stein, Larry Hogan)
  conf:1.0,           
  partialBias:8,       // how far (pts) a partly counted county's early batch can sit from its final result
  partialFloor:1.5,    // smallest allowance (pts) for a nearly finished county's last batch
  q:0.8               
};
const PROJ_LATE_MAIL=new Set(["PA","WI","MI","NV","AZ","CA","OR","WA","UT","CO","NY","NJ","MD"]);
const _clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function _solve(A,b){ const n=b.length, M=A.map((r,i)=>[...r,b[i]]);
  for(let i=0;i<n;i++){ let p=i; for(let k=i+1;k<n;k++) if(Math.abs(M[k][i])>Math.abs(M[p][i])) p=k; [M[i],M[p]]=[M[p],M[i]]; const d=M[i][i]||1e-9;
    for(let k=i+1;k<n;k++){ const f=M[k][i]/d; for(let j=i;j<=n;j++) M[k][j]-=f*M[i][j]; } }
  const x=new Array(n).fill(0); for(let i=n-1;i>=0;i--){ let s=M[i][n]; for(let j=i+1;j<n;j++) s-=M[i][j]*x[j]; x[i]=s/(M[i][i]||1e-9); } return x; }
const _feat=u=>[1,(u.b||0)/50,(u.s1==null?0:u.s1)/15,(u.s2==null?0:u.s2)/15,Math.log(Math.max(50,u.E||50))/10-1];
function _wq(vals,ws,q){ const idx=vals.map((v,i)=>i).sort((a,b)=>vals[a]-vals[b]); const W=ws.reduce((a,b)=>a+b,0)||1; let c=0; for(const i of idx){ c+=ws[i]; if(c>=q*W) return vals[i]; } return vals[idx[idx.length-1]]||0; }
const _ncdf=x=>{ const t=1/(1+0.2316419*Math.abs(x)), d=0.3989423*Math.exp(-x*x/2), p=d*t*(0.3193815+t*(-0.3565638+t*(1.781478+t*(-1.821256+t*1.330274)))); return x>0?1-p:p; };
function _mixPost(d,sd,mix){ if(!(sd<1e6)) { const v=mix.reduce((a,[w,s])=>a+w*s*s,0); return {m:0,sd:Math.sqrt(v)}; }
  const cs=mix.map(([w,s])=>{ const tv=s*s+sd*sd; const like=w*Math.exp(-d*d/(2*tv))/Math.sqrt(tv); const v=1/(1/(s*s)+1/(sd*sd)); return {like,m:v*d/(sd*sd),v}; });
  const Z=cs.reduce((a,c)=>a+c.like,0)||1; const m=cs.reduce((a,c)=>a+c.like*c.m,0)/Z; const v=cs.reduce((a,c)=>a+c.like*(c.v+c.m*c.m),0)/Z-m*m; return {m,sd:Math.sqrt(Math.max(0,v))}; }
function projectNight(contests,prior,params){
  const P={...PROJ_DEFAULTS,...(params||{})}; const mu=prior&&isFinite(prior.mu)?prior.mu:0;
  const isDone=(u,st)=>u.C>0&&u.rpt>=(PROJ_LATE_MAIL.has(st)?P.doneRptLate:P.doneRpt);
  const L=[]; contests.forEach((ct,ci)=>ct.units.forEach(u=>{ if(u.b!=null&&!u.rough&&u.E>0&&isDone(u,ct.st)){ const m=100*(u.Dv-u.Rv)/u.C; L.push({ci,st:ct.st,x:_feat(u),y:m-u.b,t:Math.log(u.C/u.E),w:Math.pow(u.E,P.wPow)}); } }));
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
  const qz=P.q===0.8?1.2816:1.6449;
  const nEff=Math.max(1,nDone); const qCounty=trusted?P.conf*_wq(S.fin.map(Math.abs),L.map(l=>l.w),P.q)*(1+5/nEff):P.conf*8;
  const qTurn=trusted?_wq(Tn.fin.map(Math.abs),L.map(l=>l.w),P.q)*(1+5/nEff):0.25;
  const resSd=Math.max(1.5,(trusted?_wq(S.fin.map(Math.abs),L.map(l=>l.w),0.6827):8));
  let eAll=0,eDone=0; const stAll={},stDone={}; contests.forEach(ct=>{ const seen=new Set(); ct.units.forEach(u=>{ const k=ct.st+'|'+u.c; if(seen.has(k)||!(u.E>0)) return; seen.add(k); eAll+=u.E; stAll[ct.st]=(stAll[ct.st]||0)+u.E; if(isDone(u,ct.st)){ eDone+=u.E; stDone[ct.st]=(stDone[ct.st]||0)+u.E; } }); });
  const fNat=eAll?eDone/eAll:0;
  const natSd=Math.sqrt((P.priorSd*(1-fNat))**2+(trusted?resSd*resSd/Math.max(1,nDone):P.priorSd*P.priorSd));
  const out=contests.map((ct,ci)=>{
    const stW=(S.stW[ct.st]||0)/S.wbar, rcW=(S.rcW[ci]||0)/S.wbar;
    const fSt=stAll[ct.st]?(stDone[ct.st]||0)/stAll[ct.st]:0; const stateSd=P.stateSd*(1-fSt)+P.stateSd*0.25*Math.sqrt(P.kState/(P.kState+stW));
    let dW=0,dS=0,dB=0,dW2=0; const scC=Math.max(1.5,qCounty/qz);
    ct.units.forEach(u=>{ if(u.fixed||u.rough||!(u.C>0)||u.b==null||!(u.E>0)) return; const x=_feat(u); const pred=u.b+x.reduce((q,v,i)=>q+v*S.beta[i],0)+(S.st[ct.st]||0);
      const th=x.reduce((q,v,i)=>q+v*Tn.beta[i],0)+(Tn.st[ct.st]||0); const tot=Math.max(u.C,u.E*Math.exp(th)); const pf=isDone(u,ct.st)?1:u.C/tot;
      const w=u.C; dW+=w; dW2+=w*w; dS+=w*(100*(u.Dv-u.Rv)/u.C-pred); dB+=w*P.partialBias*(1-pf); });
    const dPost=dW>0?_mixPost(dS/dW,Math.sqrt((dB/dW)**2+scC*scC*dW2/(dW*dW)),P.raceMix):_mixPost(0,Infinity,P.raceMix);
    const rcHat=dPost.m, raceSd=dPost.sd;
    let cD=0,cR=0,cC=0, remM=0, remV=0, remErr=0, remTurnErr=0, noBase=0, unbE=0; const outs=[];
    const kn=ct.units.filter(u=>u.b!=null&&u.E>0); const kE=kn.reduce((a,u)=>a+u.E,0); const avgB=kE?kn.reduce((a,u)=>a+u.b*u.E,0)/kE:null; const avgE=kn.length?kE/kn.length:null;
    ct.units.forEach(u0=>{ let u=u0; cD+=u.Dv||0; cR+=u.Rv||0; cC+=u.C||0;
      if(u.fixed||isDone(u,ct.st)) return; // fixed: a place listed with the race that isn't expected to add votes (feed lists stray counties) — count what it reports, expect nothing more
      if(u.b==null||!(u.E>0)){ if(avgB==null){ noBase++; return; } u={...u,b:u.b==null?avgB:u.b,E:u.E>0?u.E:avgE,s1:null,s2:null}; noBase+=0.5; unbE+=u.E; }
      if(u.rough) unbE+=0.5*u.E; // a split county's share of the district is a rough guess — widen for it
      const x=_feat(u); const yh=x.reduce((s,v,i)=>s+v*S.beta[i],0)+(S.st[ct.st]||0)+rcHat;
      const th=x.reduce((s,v,i)=>s+v*Tn.beta[i],0)+(Tn.st[ct.st]||0)+(Tn.rc[ci]||0);
      const tot=Math.max(u.C||0,u.E*Math.exp(th)); const R=Math.max(0,tot-(u.C||0));
      const mModel=_clamp(u.b+yh,-100,100); const p=(u.C||0)/tot;
      const sdM=Math.max(1,qCounty/qz), sdP=p>0?Math.max(P.partialFloor,(1-p)*P.partialBias)/Math.max(0.05,Math.sqrt(p)):1e9;
      const mObs=(u.C||0)>0?100*((u.Dv||0)-(u.Rv||0))/u.C:0; const wP=p>0?(1/(sdP*sdP))/(1/(sdP*sdP)+1/(sdM*sdM)):0;
      const mf=wP*mObs+(1-wP)*mModel, sdF=1/Math.sqrt(1/(sdM*sdM)+(p>0?1/(sdP*sdP):0));
      const need=tot*mf/100-((u.Dv||0)-(u.Rv||0)); const rm=_clamp(need,-R,R);
      remM+=rm; remV+=R; remErr+=tot*qz*sdF/100; remTurnErr+=R*(Math.exp(qTurn)-1); if(R>0) outs.push({c:u0.c,n:u0.n,R,mf,net:rm}); });
    const total=cC+remV; const unbShare=total>0?unbE/total:0; if(!(total>0)||(noBase>=1&&unbShare>0.15&&cC<0.95*total)) return {id:ct.id,m:null};
    const m=100*((cD-cR)+remM)/total, remShare=remV/total;
    const indH=100*remErr/total;                              
    const sharedSd=remShare*Math.sqrt(natSd*natSd+stateSd*stateSd+raceSd*raceSd);
    const H=Math.sqrt(indH*indH+(qz*sharedSd)**2)+Math.min(2,20*unbShare); // places guessed from the race's average place widen the range by how much of the vote they hold
    const sdTot=Math.max(0.05,H/qz); const pD=remV<=0?(m>0?1:m<0?0:0.5):_ncdf(m/sdTot);
    return {id:ct.id,m,lo:m-H,hi:m+H,H,pD,remShare,counted:cC,expected:total,natPart:remShare*natSd,statePart:remShare*stateSd,st:ct.st,trusted,noBase,raceEff:rcHat,raceSd,top:outs.sort((a,b)=>b.R-a.R).slice(0,5),nUnits:ct.units.length,nDoneUnits:ct.units.filter(u=>isDone(u,ct.st)).length};
  });
  const obsW=L.reduce((s,l)=>s+l.w,0); const obs=obsW?L.reduce((s,l)=>s+l.w*l.y,0)/obsW:null;
  return {races:out,nDone,trusted,swing:S.beta[0],obs,natSd,stateEff:S.st,stateDone:stDone,stateAll:stAll,fNat,qCounty,mu};
}

const UNIT_ALIAS={SD:{shannon:"oglalalakota"}}; // county renamed; the feed and the files don't always use the same name
const UNIT_MERGE={IL:{chicago:"cook"}};
function modelUnits(r,det,cf,base24,fy,nl){ const st=r.abbr, F=(cf&&cf[st])||{}; const cd=r.type==="House"?cdOf(r):null; const piece=cd&&!newLines(r)&&base24&&base24[st]&&base24[st].cd&&base24[st].cd[cd];
  const canon=k=>(UNIT_ALIAS[st]&&UNIT_ALIAS[st][k])||k;
  const pieceBy={}; if(piece&&piece.c){ const nm=base24[st].n||{}; Object.entries(piece.c).forEach(([f,[d,rr,t]])=>{ pieceBy[canon(unitKey(nm[f]||f))]={b:t?100*(d-rr)/t:0,E:t}; }); }
  const regs=det&&det.regions&&det.regions.length?det.regions:null;
  if(regs){ // the results feed's region lists vary by state and year: counties plus towns, old CT counties beside planning regions, districts or towns typed as counties
    const look=n=>{ const k0=unitKey(n); const ks=[UNIT_MERGE[st]&&UNIT_MERGE[st][k0],canon(k0),unitKey(String(n).replace(/\s+city$/i,"")),unitKey(String(n).replace(/&/g," and ")),unitKey(n+" planning region")].filter(Boolean); for(const k of ks) if(F[k]) return [k,F[k]]; return [canon(k0),null]; };
    const sets=[regs.filter(g=>!/town|planning|state/i.test(g.type||"")),regs.filter(g=>/planning/i.test(g.type||"")),regs.filter(g=>!/state|planning/i.test(g.type||""))].filter(a=>a.length);
    const score=a=>new Set(a.map(g=>look(g.name)).filter(x=>x[1]).map(x=>x[0])).size; // distinct places matched (towns named like counties don't count twice)
    const use=r.type==="House"?(sets[0]||[]):(sets.slice().sort((a,b)=>score(b)-score(a))[0]||[]);
    const house=r.type==="House", atLarge=cd==="AL";
    let units=use.map(g=>{ const k0=unitKey(g.name), merged=UNIT_MERGE[st]&&UNIT_MERGE[st][k0]; const [k,f]=look(g.name); const pc=pieceBy[k]||pieceBy[canon(k0)];
      let b=null,E=null,rough=false; if(house){ if(pc){ b=pc.b; E=pc.E; } else if(f&&(regs.length===1||atLarge)){ b=f[0]; E=f[3]; }
        else if(f&&newLines(r)&&nl&&nl[st]){ const n=nl[st][k]||nl[st][unitKey(g.name)]||1; b=f[0]; E=f[3]/n; rough=n>1; } }
      else if(f){ b=f[0]; E=f[3]; }
      return {c:merged||(house&&pc?(pieceBy[k]?k:canon(k0)):k),n:g.name,b,s1:f?f[1]:null,s2:f?f[2]:null,E,C:g.total||0,Dv:g.D||0,Rv:g.R||0,rpt:g.rpt||0,merged:!!merged,...(rough?{rough:true}:{})}; });
    const by=new Map(); units.forEach(u=>{ const o=by.get(u.c); if(!o){ by.set(u.c,u); return; }
      if(u.merged||o.merged) by.set(u.c,{...o,n:o.merged?u.n:o.n,C:(o.C||0)+(u.C||0),Dv:(o.Dv||0)+(u.Dv||0),Rv:(o.Rv||0)+(u.Rv||0),rpt:Math.min(o.rpt||0,u.rpt||0),merged:false});
      else if((u.C||0)>(o.C||0)) by.set(u.c,u); }); units=[...by.values()];
    if(!house){ const nF=Object.keys(F).length; if(nF&&units.filter(u=>u.b!=null).length===nF) units=units.filter(u=>u.b!=null); }
    else if(piece&&piece.c){ const nP=Object.keys(pieceBy).length; if(units.filter(u=>pieceBy[u.c]).length>=nP) units=units.filter(u=>u.b!=null);
      else units=units.map(u=>u.b==null?{...u,fixed:true}:u); } // a county the district's 2024 pieces don't include: count what it reports, expect nothing more from it
    const sumC=units.reduce((a,u)=>a+(u.C||0),0), raceC=det&&det.total!=null?det.total:counted(r); /* same fetch as the county list */ const fits=!(raceC>2000)||(sumC>=0.93*raceC&&sumC<=1.07*raceC);
    const based=units.filter(u=>u.b!=null&&!u.fixed).length, nFixed=units.filter(u=>u.fixed).length; // a list that's mostly something else (e.g. San Francisco by supervisor district) → don't trust it
    if(units.length&&fits&&based>0&&nFixed<=based) return units; }
  const b=baselineDR(r,base24); const ex=expectedVote(r,base24,fy); const E=ex&&ex.E||(piece&&piece.p?piece.p[2]:null);
  return [{c:"all",b,s1:null,s2:null,E,C:counted(r),Dv:votesOf(r,"D"),Rv:votesOf(r,"R"),rpt:r.rpt}]; }
let NLSTATES=new Set(); const newLines=r=>r.type==="House"&&NLSTATES.has(r.abbr);
const cdOf=r=>{ if(!r||r.type!=="House") return null; const d=String(r.district||"").trim(); let m=d.match(/(?:^|-)(AL|\d+)$/i); if(m) return /^al$/i.test(m[1])?"AL":String(+m[1]);
  if(/at[- ]large/i.test(r.name||"")) return "AL"; m=String(r.name||"").match(/(\d+)\s*$/); return m?String(+m[1]):null; };
const baselineDR=()=>null,expectedVote=()=>null,counted=()=>0,votesOf=()=>0;
function parseRegions(raw){ return Object.entries(raw.region_results||{}).map(([key,reg])=>{ const cs=[...(reg.candidates||[])].sort((a,b)=>(b.votes||0)-(a.votes||0)); const c1=cs[0]||{},c2=cs[1]||{};
  const fD=cs.filter(c=>partyCode(c.party)==="D").reduce((a,c)=>a+(c.votes||0),0), fR=cs.filter(c=>partyCode(c.party)==="R").reduce((a,c)=>a+(c.votes||0),0);
  const p1=partyCode(c1.party),p2=partyCode(c2.party); const tot=cs.reduce((s,c)=>s+(c.votes||0),0);
  return {key,name:reg.name||key,type:reg.type||"",total:tot,D:p1==="D"?c1.votes||0:p2==="D"?c2.votes||0:0,R:p1==="R"?c1.votes||0:p2==="R"?c2.votes||0:0,Dall:fD,Rall:fR}; }); }
window.__rh5=(async function(ids,seeds,prior,opts){ opts=opts||{}; NLSTATES=new Set(opts.newLines||[]);
  let cf=await fetch('county-features.json?v=1').then(r=>r.json()); let b24=(await fetch('base-2024.json').then(r=>r.json())).states;
  if(opts.asOf2022){
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
    const regs=parseRegions(raw); const r={abbr:st,type:ty,district:raw.district,name:raw.election_name,rpt:100};
    const u0=modelUnits(r,{regions:regs.map(g=>({name:g.name,type:g.type,rpt:100,total:g.total,D:g.D,R:g.R}))},cf,b24,null,nl);
    const keep=new Set(u0.filter(u=>!u.fixed).map(u=>u.n)); const sumKeep=regs.filter(g=>keep.has(g.name)).reduce((a,g)=>a+g.total,0);
    const nb=u0.filter(u=>u.b==null&&!u.fixed&&u.c!=="all"), fx=u0.filter(u=>u.fixed);
    if(u0.length===1&&u0[0].c==="all") issues.push(tag+': no usable county list → whole-race fallback ('+regs.length+' regions: '+[...new Set(regs.map(g=>g.type))].join('/')+')');
    if(nb.length) issues.push(tag+': no baseline for '+nb.map(u=>u.n).join(', '));
    if(fx.length) issues.push(tag+': stray counties (count only) '+fx.map(u=>u.n).join(', '));
    if(Math.abs(sumKeep/T-1)>0.03) issues.push(tag+`: county votes ${(100*sumKeep/T).toFixed(1)}% of race total`);
    const twoParty=regs.filter(g=>keep.has(g.name)).reduce((a,g)=>a+(g.D+g.R),0), twoAll=regs.filter(g=>keep.has(g.name)).reduce((a,g)=>a+(g.Dall+g.Rall),0); if(twoAll&&twoParty/twoAll<0.98) issues.push(tag+`: D or R not top-two in some counties (${(100*twoParty/twoAll).toFixed(1)}% of D+R votes seen)`);
    races.push({id,st,ty,tag,regs,keep,T,final:100*(D-R)/T,r}); });
  const rows=[];
  for(const seed of seeds) for(const flip of [1,-1]){
    const rg=rngF(seed*7919+(flip>0?1:2)); const timing={};
    const sizes={}; races.forEach(rc=>rc.regs.forEach(g=>{ const k=rc.st+'|'+unitKey(g.name); sizes[k]=Math.max(sizes[k]||0,g.total); }));
    const bySt={}; Object.entries(sizes).forEach(([k,v])=>{ const st=k.split('|')[0]; (bySt[st]=bySt[st]||[]).push([k,v]); });
    Object.entries(bySt).forEach(([st,arr])=>{ arr.sort((a,b)=>a[1]-b[1]); const sb=(rg()-0.5)*10; arr.forEach(([k],i)=>{ const pct=i/Math.max(1,arr.length-1);
      const fin=_c(0.08+ZONE[TZ(st)]+0.45*rg()+0.3*pct*(0.5+rg()),0.05,0.98); const start=_c(fin-0.15-0.3*rg(),0,fin-0.01);
      const sign=(MAIL_EARLY.has(st)?1:-1)*flip; timing[k]={fin,start,bias:sign*Math.abs(4+3*gauss(rg))+sb*0.3+gauss(rg)*3}; }); });
    for(const t of [0.15,0.25,0.35,0.5,0.65,0.8,0.9]){
      const tcs=[];
      const contests=races.map(rc=>{ let cc=0,tt=0; const slim=rc.regs.map(g=>{ const tm=timing[rc.st+'|'+unitKey(g.name)]; const p=t>=tm.fin?1:t<=tm.start?0:(t-tm.start)/(tm.fin-tm.start);
          const T=g.total||1; const fm=100*(g.D-g.R)/T; const pm=_c(fm+(1-p)*tm.bias,-100,100); const two=(g.D+g.R)/T; const C=T*p; if(rc.keep.has(g.name)){ cc+=C; tt+=g.total; }
          return {name:g.name,type:g.type,rpt:p>=1?100:Math.floor(p*100),total:Math.round(C),D:Math.max(0,Math.round(C*(two/2+pm/200))),R:Math.max(0,Math.round(C*(two/2-pm/200)))}; });
        tcs.push(tt?cc/tt:0); const tot=Math.round(rc.T*(tt?cc/tt:0));
        return {id:rc.id,st:rc.st,units:modelUnits({...rc.r,rpt:Math.round(100*(tt?cc/tt:0))},{regions:slim,total:tot},cf,b24,null,nl)}; });
      const res=projectNight(contests,{mu:prior},{});
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
  return out; });

}
"ok"