var CLAUDE_KEYS=["fable","opus","sonnet","haiku"];
/* Every target value names its provider with a prefix. Custom provider names
   join the built-ins once the mapping payload arrives. */
var BUILTIN_ROUTE_PROVIDERS=["codex","kimi","grok"];
var ROUTE_PROVIDERS=BUILTIN_ROUTE_PROVIDERS.slice();
var CUSTOM_PROVIDERS=[];
/* Live model IDs per provider. Purely autocomplete for the add-node box —
   the gateway never validates map values against a catalog, so an
   unreachable one only costs suggestions. */
var CATALOG={codex:[],kimi:[],grok:[]};
/* Curated compaction reroute targets, in the dashboard's <select> order.
   "claude-haiku" is deliberately never a curated choice here. */
var COMPACTION_CURATED_MODELS=["claude-sonnet-5","claude-opus-5","claude-fable-5"];
/* Reads config.parse_route_target's shape: the first colon separates the
   provider prefix from the complete model portion, including later colons. */
function routeOf(value){
  var at=String(value).indexOf(":");
  var prefix=value.slice(0,at);
  return{provider:prefix,model:value.slice(at+1)};
}
function routeValue(provider,model){return provider+":"+model}
/* Target nodes staged on the board but not wired yet. The board no longer
   dumps the catalogs, so a brand-new route is authored by placing its node
   here first and wiring it after. */
var addedTargets=[];
/* The single Claude Code → provider map. Source keys are the fixed Claude
   tiers; targets are only what the map references plus staged nodes. */
var DIR={
  envName:"CLAUDEX_MODEL_MAP",locked:false,
  LIVE:{},mapping:{},sources:[],targets:[],sel:null
};
function uniq(ids){var seen={};return ids.filter(function(id){if(!id||seen[id])return false;seen[id]=true;return true})}
/* Codex answers {models:[slug]}; Kimi relays its backend's catalog verbatim,
   which uses the Anthropic {data:[{id}]} shape. Accept either, ignore the rest. */
function catalogIds(body){
  var list=body&&(Array.isArray(body.models)?body.models:Array.isArray(body.data)?body.data:null);
  if(!list)return[];
  return uniq(list.map(function(m){
    return typeof m==="string"?m:(m&&(m.id||m.slug||m.name))||"";
  }));
}
/* Keep the position of every node that survives the rebuild so adding or
   removing one never rearranges a board the user laid out by hand; new nodes
   stack below the lowest existing one. */
function rebuildColumn(existing,ids,step){
  var kept={},bottom=28-step;
  existing.forEach(function(item){kept[item.id]=item;bottom=Math.max(bottom,item.y)});
  return ids.map(function(id){
    if(kept[id])return kept[id];
    bottom+=step;
    return{id:id,y:bottom};
  });
}
function buildColumns(){
  DIR.sources=rebuildColumn(DIR.sources,
    uniq(CLAUDE_KEYS.concat(Object.keys(DIR.LIVE),Object.keys(DIR.mapping))),88);
  /* LIVE first so a route the draft removed keeps its node (and its ghost
     wire and undo affordance) until Apply. */
  DIR.targets=rebuildColumn(DIR.targets,
    uniq(Object.values(DIR.LIVE).concat(Object.values(DIR.mapping),addedTargets)),88);
}
function claudeIcon(s){return '<svg viewBox="0 0 24 24" width="'+s+'" height="'+s+'" aria-hidden="true">'+
  '<g stroke="#D97757" stroke-width="2.6" stroke-linecap="round">'+
  '<line x1="12" y1="3" x2="12" y2="8"/><line x1="12" y1="16" x2="12" y2="21"/>'+
  '<line x1="3" y1="12" x2="8" y2="12"/><line x1="16" y1="12" x2="21" y2="12"/>'+
  '<line x1="5.6" y1="5.6" x2="9.2" y2="9.2"/><line x1="14.8" y1="14.8" x2="18.4" y2="18.4"/>'+
  '<line x1="18.4" y1="5.6" x2="14.8" y2="9.2"/><line x1="9.2" y1="14.8" x2="5.6" y2="18.4"/></g></svg>'}
function codexIcon(s){return '<svg viewBox="0 0 24 24" width="'+s+'" height="'+s+'" aria-hidden="true">'+
  '<rect x="2" y="2" width="20" height="20" rx="5" fill="var(--text)"/>'+
  '<g stroke="var(--canvas)" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">'+
  '<polyline points="7,9 11,12 7,15"/><line x1="13" y1="15.5" x2="17" y2="15.5"/></g></svg>'}
function kimiIcon(s){return '<svg viewBox="0 0 24 24" width="'+s+'" height="'+s+'" aria-hidden="true">'+
  '<rect x="2" y="2" width="20" height="20" rx="5" fill="var(--text)"/>'+
  '<path transform="translate(4.5 4.5) scale(.625)" fill="var(--canvas)" '+
  'd="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>'}
function grokIcon(s){return '<svg viewBox="0 0 24 24" width="'+s+'" height="'+s+'" aria-hidden="true">'+
  '<rect x="2" y="2" width="20" height="20" rx="5" fill="var(--text)"/>'+
  '<g stroke="var(--canvas)" stroke-width="2.4" stroke-linecap="round">'+
  '<line x1="7" y1="7" x2="17" y2="17"/><line x1="17" y1="7" x2="7" y2="17"/></g></svg>'}
function customProviderIcon(s){return '<svg viewBox="0 0 24 24" width="'+s+'" height="'+s+'" aria-hidden="true">'+
  '<rect x="2" y="2" width="20" height="20" rx="5" fill="var(--text)"/>'+
  '<g fill="var(--canvas)"><circle cx="7" cy="12" r="1.5"/><circle cx="12" cy="12" r="1.5"/>'+
  '<circle cx="17" cy="12" r="1.5"/></g></svg>'}
function providerIcon(provider,s){return provider==="kimi"?kimiIcon(s):provider==="grok"?grokIcon(s):
  provider==="codex"?codexIcon(s):customProviderIcon(s)}
var X_ICON='<svg viewBox="0 0 24 24" width="10" height="10" fill="none" stroke="#fff" stroke-width="3.5" '+
  'stroke-linecap="round" aria-hidden="true"><line x1="6" y1="6" x2="18" y2="18"/><line x1="18" y1="6" x2="6" y2="18"/></svg>';
var drag=null,move=null,panning=null,pan={x:0,y:0},zoom=1;
var board=document.getElementById("board"),svg=document.getElementById("wires"),layer=document.getElementById("layer");
function applyView(){
  layer.style.transform="translate("+pan.x+"px,"+pan.y+"px) scale("+zoom+")";
  document.getElementById("z-pct").textContent=Math.round(zoom*100)+"%";
}
function setZoom(next){
  next=Math.max(.4,Math.min(2.5,next));
  // Zoom around the board center so the visible middle stays put.
  var b=board.getBoundingClientRect(),cx=b.width/2,cy=b.height/2;
  pan.x=cx-(cx-pan.x)/zoom*next;
  pan.y=cy-(cy-pan.y)/zoom*next;
  zoom=next;
  applyView();drawWires();
}
function esc(s){return String(s).replace(/</g,"&lt;")}
function stateOf(k){
  if(!(k in DIR.mapping))return(k in DIR.LIVE)?"REMOVED":"UNWIRED";
  if(!(k in DIR.LIVE))return"ADDED";
  return DIR.LIVE[k]!==DIR.mapping[k]?"MODIFIED":"LIVE";
}
/* Draft changes in board vocabulary: ADDED -> wired, MODIFIED -> rewired, REMOVED -> unwired. */
function draftCounts(){
  var c={wired:0,rewired:0,unwired:0};
  Object.keys(DIR.mapping).forEach(function(k){
    var s=stateOf(k);
    if(s==="ADDED")c.wired++;else if(s==="MODIFIED")c.rewired++;
  });
  Object.keys(DIR.LIVE).forEach(function(k){if(stateOf(k)==="REMOVED")c.unwired++});
  return c;
}
function dirtyCount(){
  var c=draftCounts();
  return c.wired+c.rewired+c.unwired;
}
function portXY(el,side){
  var b=board.getBoundingClientRect(),r=el.getBoundingClientRect();
  return{x:(side==="r"?r.right:r.left)-b.left,y:r.top-b.top+r.height/2};
}
function bez(a,b){
  var dx=Math.max(60,(b.x-a.x)/2);
  return"M"+a.x+","+a.y+" C"+(a.x+dx)+","+a.y+" "+(b.x-dx)+","+b.y+" "+b.x+","+b.y;
}
function nodeEl(k,cls){return board.querySelector('.node.'+cls+'[data-id="'+CSS.escape(k)+'"]')}
var WIRE_DEFS='<defs>'+
  ["wire","accent","warn"].map(function(c){
    return '<marker id="ah-'+c+'" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" '+
      'refX="9" refY="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="var(--'+c+')"/></marker>';
  }).join("")+'</defs>';
function renderChrome(){
  document.getElementById("lock-env").textContent=DIR.envName;
  document.body.classList.toggle("env-locked",!!DIR.locked);
}
function render(){
  // The target column follows the map, so it is rebuilt on every render.
  buildColumns();
  layer.querySelectorAll(".node").forEach(function(n){n.remove()});
  var srcIcon=claudeIcon(14);
  DIR.sources.forEach(function(s){
    var st=stateOf(s.id);
    var isDraft=st==="ADDED"||st==="MODIFIED"||st==="REMOVED";
    var el=document.createElement("div");
    el.className="node src"+(st==="LIVE"?" live":"");
    el.dataset.id=s.id;el.style.top=s.y+"px";
    el.innerHTML='<span class="nicon">'+srcIcon+'</span><span class="name">'+esc(s.id)+'</span>'+
      '<span class="st">'+(s.id in DIR.mapping?"WIRED":"UNWIRED")+
        (isDraft?' <span class="dr">(draft)</span> <span class="undo" data-k="'+esc(s.id)+'">undo</span>':"")+'</span>'+
      '<span class="port" data-k="'+esc(s.id)+'"></span>'+
      (s.id in DIR.mapping?'<button class="x" data-k="'+esc(s.id)+'" title="연결 끊기">'+X_ICON+"</button>":"");
    layer.appendChild(el);
  });
  DIR.targets.forEach(function(t){
    var wired=Object.keys(DIR.mapping).some(function(k){return DIR.mapping[k]===t.id});
    var liveWired=Object.keys(DIR.LIVE).some(function(k){return DIR.LIVE[k]===t.id});
    var liveIn=Object.keys(DIR.mapping).some(function(k){return DIR.mapping[k]===t.id&&stateOf(k)==="LIVE"});
    var route=routeOf(t.id);
    var el=document.createElement("div");
    el.className="node tgt"+(liveIn?" live":"");
    el.dataset.id=t.id;el.style.top=t.y+"px";
    if(t.x!=null){el.style.left=t.x+"px";el.style.right="auto"}
    el.innerHTML='<span class="port'+(wired?"":" idle")+'" data-t="'+esc(t.id)+'"></span>'+
      '<span class="nicon">'+providerIcon(route.provider,14)+'</span>'+
      '<span class="name">'+esc(route.model)+'</span>'+
      '<span class="st">'+(wired?"WIRED":"UNWIRED")+
        (wired!==liveWired?' <span class="dr">(draft)</span>':"")+'</span>'+
      // A ghost node (live wire already cut in this draft) has nothing left
      // to delete; every other target node gets the remove button.
      (wired||!liveWired?'<button class="x" data-t="'+esc(t.id)+'" title="노드 제거">'+X_ICON+"</button>":"");
    layer.appendChild(el);
  });
  drawWires();
  var counts=draftCounts(),dirty=counts.wired+counts.rewired+counts.unwired;
  document.querySelector(".applybar").classList.toggle("dirty",dirty>0);
  var breakdown=["wired","rewired","unwired"].filter(function(k){return counts[k]>0})
    .map(function(k){return counts[k]+" "+k}).join(" · ");
  document.getElementById("cnt").innerHTML=
    "<b>"+dirty+(dirty===1?" change":" changes")+"</b>"+
    (breakdown?'<span class="dim">'+breakdown+"</span>":"");
}
/* rebuilt flow paths resume mid-cycle so redraws during drag do not freeze the animation */
function flowPhase(){return (-(performance.now()/1000%1.4)).toFixed(3)+"s"}
function drawWires(){
  svg.innerHTML=WIRE_DEFS;
  /* ghost wires: live routes that Apply will remove (REMOVED keys, MODIFIED old targets) */
  Object.keys(DIR.LIVE).forEach(function(k){
    var st=stateOf(k);
    if(st!=="REMOVED"&&st!=="MODIFIED")return;
    var se=nodeEl(k,"src"),te=nodeEl(DIR.LIVE[k],"tgt");
    if(!se||!te)return;
    var a=portXY(se.querySelector(".port"),"r"),b=portXY(te.querySelector(".port"),"l");
    var p=document.createElementNS("http://www.w3.org/2000/svg","path");
    var g=bez(a,b);
    p.setAttribute("d",g);p.setAttribute("class","w ghost");
    p.setAttribute("marker-end","url(#ah-wire)");
    svg.appendChild(p);
    var fl=document.createElementNS("http://www.w3.org/2000/svg","path");
    fl.setAttribute("d",g);fl.setAttribute("class","flow ghostflow");
    fl.style.animationDelay=flowPhase();
    svg.appendChild(fl);
  });
  Object.keys(DIR.mapping).forEach(function(k){
    var se=nodeEl(k,"src"),te=nodeEl(DIR.mapping[k],"tgt");
    if(!se||!te)return;
    // A picked-up wire stays on the board, faded: the temp path shows where
    // it would land while the ghost keeps showing where it still sits, so
    // letting go over empty space visibly puts it back.
    var held=!!(drag&&drag.keys.indexOf(k)>=0);
    var a=portXY(se.querySelector(".port"),"r"),b=portXY(te.querySelector(".port"),"l");
    var p=document.createElementNS("http://www.w3.org/2000/svg","path");
    var st=stateOf(k);
    var cls="w"+(st!=="LIVE"?" chg":"")+(held?" ghost":"");
    if(DIR.sel===k)cls+=" sel";
    var wired=bez(a,b);
    p.setAttribute("d",wired);p.setAttribute("class",cls);p.dataset.k=k;
    p.setAttribute("marker-end","url(#ah-"+(cls.indexOf("sel")>=0?"accent":cls.indexOf("chg")>=0?"warn":"wire")+")");
    p.addEventListener("click",function(ev){ev.stopPropagation();selectWire(k,(a.x+b.x)/2,(a.y+b.y)/2)});
    svg.appendChild(p);
    if(st==="LIVE"){
      var fl=document.createElementNS("http://www.w3.org/2000/svg","path");
      fl.setAttribute("d",wired);fl.setAttribute("class","flow"+(held?" ghostflow":""));
      fl.style.animationDelay=flowPhase();
      svg.appendChild(fl);
    }
  });
}
function selectWire(k,mx,my){
  DIR.sel=k;drawWires();
  var d=document.getElementById("wdel");
  d.classList.add("show");d.style.left=mx+"px";d.style.top=my+"px";
}
document.getElementById("wdel").addEventListener("click",function(){
  if(DIR.sel){delete DIR.mapping[DIR.sel];DIR.sel=null;this.classList.remove("show");render()}
});
board.addEventListener("click",function(ev){
  if(suppressClick){suppressClick=false;return}
  if(ev.target===board||ev.target===svg){DIR.sel=null;document.getElementById("wdel").classList.remove("show");drawWires()}
});
board.addEventListener("pointerdown",function(ev){
  var port=ev.target.closest?ev.target.closest(".node.src .port"):null;
  if(port){
    // Source port: drag rightward to a target node.
    drag={keys:[port.dataset.k],from:[portXY(port,"r")]};
    document.body.classList.add("wiring");
    ev.preventDefault();return;
  }
  var tport=ev.target.closest?ev.target.closest(".node.tgt .port"):null;
  if(tport){
    // A wire is owned by its source key, so every wire is authored left to
    // right and a target port can only pick up what already feeds it: the
    // incoming wires come loose at their far end and follow the cursor while
    // still hanging off their source ports. A target gathers N of them, so
    // the whole fan travels together. With nothing to pick up this is not a
    // wiring gesture at all — it falls through to a node drag, so a target
    // whose wires were cut can never sprout a new one from this side.
    var incoming=Object.keys(DIR.mapping).filter(function(k){
      return DIR.mapping[k]===tport.dataset.t&&nodeEl(k,"src");
    });
    if(incoming.length){
      drag={keys:incoming,from:incoming.map(function(k){
        return portXY(nodeEl(k,"src").querySelector(".port"),"r");
      })};
      document.body.classList.add("wiring");
      ev.preventDefault();return;
    }
  }
  var xbtn=ev.target.closest?ev.target.closest(".x"):null;
  if(xbtn){
    if(xbtn.dataset.t!=null){
      // Dropping a node cuts every wire feeding it: live ones stay as REMOVED
      // drafts (ghost + undo), session wires just vanish.
      var tid=xbtn.dataset.t;
      Object.keys(DIR.mapping).forEach(function(k){if(DIR.mapping[k]===tid)delete DIR.mapping[k]});
      addedTargets=addedTargets.filter(function(id){return id!==tid});
    }else{
      delete DIR.mapping[xbtn.dataset.k];   // Fixed key: disconnect only (if LIVE, REMOVED + undo).
    }
    if(DIR.sel&&!(DIR.sel in DIR.mapping))DIR.sel=null;
    document.getElementById("wdel").classList.remove("show");
    render();return;
  }
  var undo=ev.target.closest(".undo");
  if(undo){
    var k=undo.dataset.k;
    if(k in DIR.LIVE)DIR.mapping[k]=DIR.LIVE[k];else delete DIR.mapping[k];
    render();return;
  }
  // Only target nodes move; source nodes stay fixed in their column.
  var node=ev.target.closest(".node.tgt");
  if(node){
    var item=DIR.targets.find(function(x){return x.id===node.dataset.id});
    var rect=node.getBoundingClientRect(),brect=board.getBoundingClientRect();
    if(item.x==null)item.x=(rect.left-brect.left-pan.x)/zoom;
    move={item:item,off:ev.clientY-rect.top,offX:ev.clientX-rect.left};
    ev.preventDefault();return;
  }
  if(ev.target===board||ev.target===svg){
    panning={sx:ev.clientX,sy:ev.clientY,ox:pan.x,oy:pan.y,moved:false};
    board.classList.add("panning");
    ev.preventDefault();
  }
});
document.addEventListener("pointermove",function(ev){
  if(drag){
    var b=board.getBoundingClientRect();
    var cur2={x:ev.clientX-b.left,y:ev.clientY-b.top};
    drawWires();
    drag.from.forEach(function(a){
      var t=document.createElementNS("http://www.w3.org/2000/svg","path");
      t.setAttribute("d",bez(a,cur2));t.setAttribute("class","temp");
      t.setAttribute("marker-end","url(#ah-accent)");
      svg.appendChild(t);
    });
    // Wires only ever land on a target, so that is the only side that lights up.
    board.querySelectorAll(".node.tgt").forEach(function(n){n.classList.remove("droppable")});
    var el=document.elementFromPoint(ev.clientX,ev.clientY);
    var over=el&&el.closest?el.closest(".node.tgt"):null;
    if(over)over.classList.add("droppable");
  }else if(move){
    var b=board.getBoundingClientRect();
    move.item.y=Math.max(-260,Math.min(860,(ev.clientY-b.top-move.off-pan.y)/zoom));
    var el=nodeEl(move.item.id,"tgt");
    if(el){
      el.style.top=move.item.y+"px";
      move.item.x=Math.max(-300,Math.min(1600,(ev.clientX-b.left-move.offX-pan.x)/zoom));
      el.style.left=move.item.x+"px";el.style.right="auto";
      drawWires();
    }
  }else if(panning){
    var dx=ev.clientX-panning.sx,dy=ev.clientY-panning.sy;
    if(Math.abs(dx)+Math.abs(dy)>3)panning.moved=true;
    pan.x=panning.ox+dx;pan.y=panning.oy+dy;
    applyView();drawWires();
  }
});
var suppressClick=false;
document.addEventListener("pointerup",function(ev){
  if(drag){
    var el=document.elementFromPoint(ev.clientX,ev.clientY);
    var tgt=el&&el.closest?el.closest(".node.tgt"):null;
    if(tgt)drag.keys.forEach(function(k){DIR.mapping[k]=tgt.dataset.id});
    drag=null;document.body.classList.remove("wiring");render();
  }
  if(panning){
    suppressClick=panning.moved;
    board.classList.remove("panning");
    panning=null;
  }
  move=null;
});
document.getElementById("z-in").addEventListener("click",function(){setZoom(zoom*1.2)});
document.getElementById("z-out").addEventListener("click",function(){setZoom(zoom/1.2)});
document.getElementById("z-pct").addEventListener("click",function(){setZoom(1)});
document.getElementById("z-reset").addEventListener("click",fitView);
// Wheel zoom anchored at the cursor so the point under it stays put; the
// passive:false is required to preventDefault and stop the page scroll.
board.addEventListener("wheel",function(ev){
  ev.preventDefault();
  if(drag)return;   // don't zoom the wires out from under an active drag
  var next=Math.max(.4,Math.min(2.5,zoom*Math.exp(-ev.deltaY*0.0018)));
  if(next===zoom)return;
  var b=board.getBoundingClientRect();
  var mx=ev.clientX-b.left,my=ev.clientY-b.top;
  pan.x=mx-(mx-pan.x)/zoom*next;
  pan.y=my-(my-pan.y)/zoom*next;
  zoom=next;
  applyView();drawWires();
},{passive:false});
/* --- staging a target node ------------------------------------------------ */
var addProvider="codex";
document.getElementById("add-prov").innerHTML=ROUTE_PROVIDERS.map(function(p){
  // Optional providers start hidden; setProviderVisibility reveals them
  // once /health confirms a local login (or a map route that needs one).
  return'<button data-p="'+p+'"'+(p==="codex"?"":' class="provider-hidden"')+">"+p+"</button>";
}).join("");
/* The provider buttons are built once and only toggled: re-rendering them
   would detach the clicked one mid-event, and the outside-click listener
   below would then see a node with no .addnode ancestor and close the form. */
function renderAddForm(){
  document.querySelectorAll("#add-prov button").forEach(function(b){
    b.classList.toggle("on",b.dataset.p===addProvider);
  });
  document.getElementById("add-catalog").innerHTML=CATALOG[addProvider].map(function(id){
    return'<option value="'+esc(id).replace(/"/g,"&quot;")+'">';
  }).join("");
}
function openAddForm(open){
  document.getElementById("add-form").classList.toggle("open",open);
  if(open){renderAddForm();document.getElementById("add-model").focus()}
}
function addTargetNode(){
  var input=document.getElementById("add-model"),model=input.value.trim();
  if(!model)return;
  // The catalog only suggests: a typed ID the gateway has never heard of is
  // still valid, exactly as a hand-written model_map value would be.
  var value=routeValue(addProvider,model);
  if(addedTargets.indexOf(value)<0)addedTargets.push(value);
  input.value="";openAddForm(false);render();
}
document.getElementById("add-open").addEventListener("click",function(){
  openAddForm(!document.getElementById("add-form").classList.contains("open"));
});
document.getElementById("add-prov").addEventListener("click",function(ev){
  var b=ev.target.closest("button");
  if(!b||b.dataset.p===addProvider)return;
  addProvider=b.dataset.p;renderAddForm();
  var input=document.getElementById("add-model");
  /* The datalist filters its suggestions by the input's current value, so
     keeping another provider's model id would hide every fresh option —
     the dropdown looks dead until the text is deleted by hand. */
  input.value="";
  input.focus();
});
document.getElementById("add-go").addEventListener("click",addTargetNode);
document.getElementById("add-model").addEventListener("keydown",function(ev){
  if(ev.key==="Enter")addTargetNode();
  else if(ev.key==="Escape")openAddForm(false);
});
document.addEventListener("click",function(ev){
  if(!ev.target.closest||!ev.target.closest(".addnode"))openAddForm(false);
});
var toastTimer=null;
function showToast(html,isErr){
  var t=document.getElementById("toast");
  if(!t){
    t=document.createElement("div");t.id="toast";
    t.addEventListener("click",function(){t.remove()});
    document.body.appendChild(t);
  }else if(!t.isConnected){document.body.appendChild(t)}
  t.className="toast"+(isErr?" err":"");
  t.innerHTML=html;
  clearTimeout(toastTimer);
  toastTimer=setTimeout(function(){t.remove()},4500);
}
/* --- live data plumbing ------------------------------------------------- */
/* CLAUDEX_LOCAL_TOKEN lives only in these closure variables for the page's
   lifetime: never in HTML, URLs, storage, or logs. */
var authRequired=false,localToken=null;
function promptLocalToken(){
  var entered=window.prompt("이 게이트웨이는 CLAUDEX_LOCAL_TOKEN 인증이 필요합니다.\n토큰을 입력하세요:");
  localToken=entered?entered:null;
  return localToken!==null;
}
function rawFetch(url,opts){
  if(url.indexOf("/admin/")===0&&localToken){
    opts=Object.assign({},opts||{});
    opts.headers=Object.assign({},opts.headers||{},{"Authorization":"Bearer "+localToken});
  }
  return fetch(url,opts).then(function(r){
    return r.json().catch(function(){return{}}).then(function(b){return{ok:r.ok,status:r.status,body:b}});
  });
}
function jfetch(url,opts){
  return rawFetch(url,opts).then(function(r){
    if(r.status!==401||url.indexOf("/admin/")!==0||!authRequired)return r;
    localToken=null;   // wrong or missing token: ask once, retry once, no loop
    if(!promptLocalToken())return r;
    return rawFetch(url,opts);
  });
}
function errDetail(body){
  return body&&body.error&&body.error.message?body.error.message:"request failed";
}
function renderHealth(h){
  var box=document.getElementById("health");
  var overallClass={ok:"ok",degraded:"dg",error:"er"}[h.status]||"er";
  var chips=[
    {cls:overallClass,label:"Gateway "+String(h.status||"error").toUpperCase(),go:"map",title:""}
  ];
  box.innerHTML=chips.map(function(c){
    return '<span class="'+c.cls+'" data-go="'+c.go+'" title="'+esc(c.title).replace(/"/g,"&quot;")+'">'+esc(c.label)+"</span>";
  }).join("");
  box.querySelectorAll("span").forEach(function(s){
    s.addEventListener("click",function(){setTab(this.dataset.go)});
  });
}
var COPY_ICON='<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" '+
  'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'+
  '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';
var COPIED_ICON='<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="var(--ok)" stroke-width="2.5" '+
  'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="4,12 10,18 20,6"/></svg>';
function wireCopy(btn,text){
  btn.innerHTML=COPY_ICON;
  btn.addEventListener("click",function(){
    navigator.clipboard.writeText(text).then(function(){
      btn.innerHTML=COPIED_ICON;
      setTimeout(function(){btn.innerHTML=COPY_ICON},1500);
    });
  });
}
/* The box carries the state colour and the plan line; only the status line
   inside it is rewritten, so this never wipes the plan the usage probe put
   there. `title` keeps the raw upstream detail reachable on hover. */
function renderStatLine(provider,state,title,html){
  var box=document.getElementById(provider+"-stat");
  box.className="stat"+(state?" "+state:"");
  if(title)box.title=title;else box.removeAttribute("title");
  document.getElementById(provider+"-statline").innerHTML=html;
}
function configureCustomProviders(providers){
  CUSTOM_PROVIDERS=Array.isArray(providers)?providers.filter(function(p){return p&&p.name}):[];
  ROUTE_PROVIDERS=BUILTIN_ROUTE_PROVIDERS.concat(CUSTOM_PROVIDERS.map(function(p){return p.name}));
  CUSTOM_PROVIDERS.forEach(function(provider){
    var name=provider.name;
    CATALOG[name]=[];
    PROVIDER_VISIBLE[name]=true;
    if(!document.querySelector('#add-prov button[data-p="'+name+'"]')){
      document.getElementById("add-prov").insertAdjacentHTML("beforeend",
        '<button data-p="'+name+'" class="provider-hidden">'+esc(name)+"</button>");
    }
    if(!document.getElementById("card-"+name)){
      var card=document.createElement("div");
      card.className="card provider-hidden";
      card.id="card-"+name;
      card.innerHTML='<div class="ucard-h"><h2>'+esc(name)+' Status</h2></div>'+
        '<div class="sub">설정된 OpenAI 호환 Responses API 프로바이더의 연결 상태입니다.</div>'+
        '<div class="stat" id="'+name+'-stat"><div class="statline" id="'+name+'-statline">'+
        '<span class="sk">OK · Responses API 연결됨</span></div></div>';
      document.getElementById("tab-status").appendChild(card);
    }
    jfetch("/admin/providers/custom/"+encodeURIComponent(name)+"/models").then(function(r){
      if(r.ok)CATALOG[name]=catalogIds(r.body);
      if(addProvider===name&&document.getElementById("add-form").classList.contains("open"))renderAddForm();
    }).catch(function(){
      CATALOG[name]=[];
    });
  });
}
/* Purely cosmetic gating for first-run clarity: kimi/grok are extensions, so
   they appear only when detected or required. A custom provider was explicitly
   configured and therefore remains visible even when its health probe fails. */
var PROVIDER_VISIBLE={kimi:false,grok:false};
function setProviderVisibility(h){
  ROUTE_PROVIDERS.filter(function(p){return p!=="codex"}).forEach(function(p){
    var info=((h||{}).providers||{})[p]||{};
    var visible=BUILTIN_ROUTE_PROVIDERS.indexOf(p)<0||info.status==="ok"||info.required===true;
    PROVIDER_VISIBLE[p]=visible;
    var card=document.getElementById("card-"+p);
    if(card)card.classList.toggle("provider-hidden",!visible);
    var btn=document.querySelector('#add-prov button[data-p="'+p+'"]');
    if(btn)btn.classList.toggle("provider-hidden",!visible);
  });
}
function renderProviderCards(h){
  var codex=(h.providers||{}).codex||{};
  if(codex.status==="ok"){
    renderStatLine("codex","okv",null,'● OK <span class="detail">'+
      (codex.auth_mode==="api_key"?"API 키로 인증됨":"ChatGPT 계정으로 로그인됨")+"</span>"+
      (codex.auth_mode!=="api_key"&&(codex.email||codex.account)
        ?'<div class="codeblock">'+esc(String(codex.email||codex.account))+"</div>":""));
  }else{
    renderStatLine("codex","err",codex.detail||"",
      '● ERROR <span class="detail">Codex 로그인이 필요합니다</span>'+
      '<div class="codeblock"><span class="tx">$ codex login</span>'+
      '<button class="cp" id="codex-login-copy" title="복사"></button></div>');
    wireCopy(document.getElementById("codex-login-copy"),"codex login");
  }
  var kimi=(h.providers||{}).kimi||{};
  if(kimi.status==="ok"){
    renderStatLine("kimi","okv",null,'● OK <span class="detail">Kimi 계정으로 로그인됨</span>'+
      (kimi.account?'<div class="codeblock">'+esc(String(kimi.account))+"</div>":""));
  }else if(kimi.required===false){
    // Kimi login does not affect readiness without a route, so use a neutral state.
    renderStatLine("kimi","",kimi.detail||"",
      '● 미사용 <span class="detail">모델 맵에 kimi: 타겟이 없어 로그인하지 않아도 됩니다</span>'+
      '<div class="codeblock"><span class="tx">$ kimi login</span>'+
      '<button class="cp" id="kimi-login-copy" title="복사"></button></div>');
    wireCopy(document.getElementById("kimi-login-copy"),"kimi login");
  }else{
    renderStatLine("kimi","err",kimi.detail||"",
      '● ERROR <span class="detail">Kimi 로그인이 필요합니다</span>'+
      '<div class="codeblock"><span class="tx">$ kimi login</span>'+
      '<button class="cp" id="kimi-login-copy" title="복사"></button></div>');
    wireCopy(document.getElementById("kimi-login-copy"),"kimi login");
  }
  var grok=(h.providers||{}).grok||{};
  if(grok.status==="ok"){
    renderStatLine("grok","okv",null,'● OK <span class="detail">'+
      (grok.auth_mode==="api_key"?"API 키로 인증됨":"Grok 계정으로 로그인됨")+"</span>"+
      (grok.auth_mode!=="api_key"&&grok.account
        ?'<div class="codeblock">'+esc(String(grok.account))+"</div>":""));
  }else if(grok.required===false){
    // Grok login does not affect readiness without a route, so use a neutral state.
    renderStatLine("grok","",grok.detail||"",
      '● 미사용 <span class="detail">모델 맵에 grok: 타겟이 없어 로그인하지 않아도 됩니다</span>'+
      '<div class="codeblock"><span class="tx">$ grok login</span>'+
      '<button class="cp" id="grok-login-copy" title="복사"></button></div>');
    wireCopy(document.getElementById("grok-login-copy"),"grok login");
  }else{
    renderStatLine("grok","err",grok.detail||"",
      '● ERROR <span class="detail">Grok 로그인이 필요합니다</span>'+
      '<div class="codeblock"><span class="tx">$ grok login</span>'+
      '<button class="cp" id="grok-login-copy" title="복사"></button></div>');
    wireCopy(document.getElementById("grok-login-copy"),"grok login");
  }
  CUSTOM_PROVIDERS.forEach(function(provider){
    var name=provider.name,info=(h.providers||{})[name]||{};
    if(info.status==="ok"){
      renderStatLine(name,"okv",null,'● OK <span class="detail">Responses API 연결됨</span>');
    }else if(info.required===false){
      renderStatLine(name,"",info.detail||"",'● 미사용 <span class="detail">모델 맵에 '+esc(name)+
        ': 타겟이 없어 연결 실패가 준비 상태와 무관합니다</span>');
    }else{
      renderStatLine(name,"err",info.detail||"",'● ERROR <span class="detail">Responses API 연결 실패</span>');
    }
  });
}
function renderFacts(p){
  document.getElementById("kv-codex-home").textContent=p.codex_home||"—";
  document.getElementById("kv-kimi-auth").textContent=p.kimi_code_home||"—";
  document.getElementById("kv-grok-home").textContent=p.grok_home||"—";
}
function renderLogLevel(p){
  var box=document.getElementById("loglevel");
  var locked=!!p.env_locked;
  box.innerHTML=(p.choices||[]).map(function(l){
    return '<button data-l="'+esc(l)+'"'+(l===p.log_level?' class="on"':"")+(locked?" disabled":"")+
      (locked?' title="'+esc(p.env_locked)+' 환경변수가 우선합니다"':"")+'>'+esc(l.toUpperCase())+"</button>";
  }).join("");
  box.querySelectorAll("button").forEach(function(b){
    b.addEventListener("click",function(){setLogLevel(this.dataset.l)});
  });
}
var logTimer=null,logLive=false;
/* The poll runs only while the Log tab is visible and Live is toggled on. */
function syncLogTimer(){
  var active=logLive&&document.body.dataset.tab==="log";
  if(active&&!logTimer)logTimer=setInterval(fetchLogs,2000);
  if(!active&&logTimer){clearInterval(logTimer);logTimer=null}
}
function fmtLogTs(ts){
  var d=new Date(ts*1000);
  function pad(n){return(n<10?"0":"")+n}
  return pad(d.getHours())+":"+pad(d.getMinutes())+":"+pad(d.getSeconds());
}
function renderLogs(entries){
  var box=document.getElementById("logbox");
  if(!entries.length){box.innerHTML='<div class="logempty">로그가 아직 없습니다.</div>';return}
  var pinned=box.scrollTop+box.clientHeight>=box.scrollHeight-8;
  box.innerHTML=entries.map(function(e){
    return '<div class="logline"><span class="lts">'+fmtLogTs(e.ts)+'</span><span class="llv '+esc(e.level)+'">'+
      esc(e.level)+"</span>"+esc(e.message)+"</div>";
  }).join("");
  if(pinned)box.scrollTop=box.scrollHeight;
}
function fetchLogs(){
  jfetch("/admin/logs").then(function(r){
    if(r.ok&&Array.isArray(r.body.logs))renderLogs(r.body.logs);
  }).catch(function(){});
}
function setLogLevel(level){
  jfetch("/admin/settings/log-level",{
    method:"PUT",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({log_level:level})
  }).then(function(r){
    if(!r.ok){
      showToast('<span class="chip chip-err">'+(r.status===409?"LOCKED":"ERROR")+'</span><span class="lat">'+r.status+
        "</span><br>log level"+'<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      jfetch("/admin/settings/log-level").then(function(g){if(g.ok)renderLogLevel(g.body)});
      return;
    }
    renderLogLevel(r.body);
    showToast('<span class="chip chip-ok">APPLIED</span><br>log level → '+esc(r.body.log_level),false);
  }).catch(function(){
    showToast('<span class="chip chip-err">ERROR</span><br>log level<br><span class="dim">gateway unreachable</span>',true);
  });
}
/* --- compaction reroute (Claude Code's own compaction requests, sent
   directly to Anthropic instead of the mapped provider) ------------------ */
var COMP={
  envName:"CLAUDEX_COMPACTION_MODEL",
  locked:false,
  LIVE:{model:null},
  // draftKind: "disabled" | one of COMPACTION_CURATED_MODELS | "custom"
  draftKind:"disabled",
  draftCustom:""   // raw id typed into the custom input, untrimmed
};
/* Strips the pinned "claude:" prefix parse_compaction_model expects; a bare
   or unprefixed value (should never reach here from the server) still
   renders as its own id rather than throwing. */
function compactionModelId(model){
  var at=String(model).indexOf(":");
  return at<0?model:model.slice(at+1);
}
function compactionDraftFromModel(model){
  if(model==null)return{kind:"disabled",custom:""};
  var id=compactionModelId(model);
  return COMPACTION_CURATED_MODELS.indexOf(id)>=0
    ?{kind:id,custom:""}
    // Not one of the three curated ids: still render it, as Custom with the
    // raw id filled in, rather than silently dropping it.
    :{kind:"custom",custom:id};
}
function renderCompaction(){
  document.getElementById("compaction-card").classList.toggle("locked",COMP.locked);
  document.getElementById("comp-lock-env").textContent=COMP.envName;
  var select=document.getElementById("comp-select");
  select.value=COMP.draftKind;
  select.disabled=COMP.locked;
  document.getElementById("comp-custom-wrap").classList.toggle("show",COMP.draftKind==="custom");
  var input=document.getElementById("comp-custom-input");
  input.value=COMP.draftCustom;
  input.disabled=COMP.locked;
  var btn=document.getElementById("comp-apply");
  btn.disabled=COMP.locked;
  btn.textContent="적용";
}
/* Adopts a fresh /admin/settings/compaction GET/PUT envelope as the new live state
   and re-derives the draft selection from it. */
function renderCompactionState(body){
  COMP.LIVE.model=body.model;
  COMP.locked=!!body.env_locked;
  var draft=compactionDraftFromModel(body.model);
  COMP.draftKind=draft.kind;
  COMP.draftCustom=draft.custom;
  renderCompaction();
}
function applyCompaction(){
  var btn=document.getElementById("comp-apply");
  if(COMP.locked||btn.disabled)return;
  var model;
  if(COMP.draftKind==="disabled"){
    model=null;
  }else if(COMP.draftKind==="custom"){
    var input=document.getElementById("comp-custom-input");
    var raw=input.value.trim();
    if(!raw)return;
    model="claude:"+raw;
  }else{
    model="claude:"+COMP.draftKind;
  }
  btn.disabled=true;btn.textContent="…";
  jfetch("/admin/settings/compaction",{
    method:"PUT",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({model:model})
  }).then(function(r){
    if(r.status===409){
      COMP.locked=true;
      renderCompaction();
      showToast('<span class="chip chip-err">LOCKED</span><span class="lat">409</span><br>'+esc(COMP.envName)+
        '<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      // The 409 body is only the admin error envelope, never state: model,
      // env_locked and last_reroute are only ever read from this fresh GET,
      // never from r.body above. And if that refresh GET itself fails or
      // returns a malformed envelope, its body must not be rendered as
      // state either — the card stays locked with its previous live state.
      jfetch("/admin/settings/compaction").then(function(g){
        if(g.ok&&g.body&&typeof g.body.env_locked==="boolean"&&"model" in g.body&&"last_reroute" in g.body){
          renderCompactionState(g.body);
        }else{
          COMP.locked=true;
          renderCompaction();
          showToast('<span class="chip chip-err">ERROR</span><br>Compaction refresh<br><span class="dim">could not load current state</span>',true);
        }
      });
      return;
    }
    if(!r.ok){
      renderCompaction();
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+r.status+"</span><br>Compaction"+
        '<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      return;
    }
    renderCompactionState(r.body);
    showToast('<span class="chip chip-ok">APPLIED</span><br>Compaction → '+esc(r.body.model||"Disabled"),false);
  }).catch(function(){
    renderCompaction();
    showToast('<span class="chip chip-err">ERROR</span><br>Compaction<br><span class="dim">gateway unreachable</span>',true);
  });
}
document.getElementById("comp-select").addEventListener("change",function(){
  COMP.draftKind=this.value;
  COMP.draftCustom="";
  renderCompaction();
});
document.getElementById("comp-custom-input").addEventListener("input",function(){
  COMP.draftCustom=this.value;
});
document.getElementById("comp-apply").addEventListener("click",applyCompaction);
/* --- Codex Fast service tier --------------------------------------------- */
var CODEX={envName:"CLAUDEX_CODEX_SERVICE_TIER",locked:false,serviceTier:null,draft:false};
function isCodexEnvelope(body){
  return !!body&&typeof body.env_locked==="boolean"&&
    (body.service_tier===null||body.service_tier==="fast");
}
function renderCodex(){
  document.getElementById("codex-card").classList.toggle("locked",CODEX.locked);
  document.getElementById("codex-lock-env").textContent=CODEX.envName;
  var checkbox=document.getElementById("codex-fast");
  checkbox.checked=CODEX.draft;
  checkbox.disabled=CODEX.locked;
  var btn=document.getElementById("codex-apply");
  btn.disabled=CODEX.locked||CODEX.draft===(CODEX.serviceTier==="fast");
  btn.textContent="Apply";
}
function renderCodexState(body){
  CODEX.serviceTier=body.service_tier;
  CODEX.locked=!!body.env_locked;
  CODEX.draft=body.service_tier==="fast";
  renderCodex();
}
function applyCodex(){
  var btn=document.getElementById("codex-apply");
  if(CODEX.locked||btn.disabled)return;
  btn.disabled=true;btn.textContent="…";
  jfetch("/admin/settings/codex",{
    method:"PUT",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({service_tier:CODEX.draft?"fast":null})
  }).then(function(r){
    if(r.status===409){
      CODEX.locked=true;
      renderCodex();
      showToast('<span class="chip chip-err">LOCKED</span><span class="lat">409</span><br>'+esc(CODEX.envName)+
        '<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      jfetch("/admin/settings/codex").then(function(g){
        if(g.ok&&isCodexEnvelope(g.body)){
          renderCodexState(g.body);
        }else{
          CODEX.locked=true;
          renderCodex();
          showToast('<span class="chip chip-err">ERROR</span><br>Codex Fast refresh<br><span class="dim">could not load current state</span>',true);
        }
      });
      return;
    }
    if(!r.ok){
      renderCodex();
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+r.status+"</span><br>Codex Fast"+
        '<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      return;
    }
    renderCodexState(r.body);
    showToast('<span class="chip chip-ok">APPLIED</span><br>Codex Fast → '+(r.body.service_tier==="fast"?"Enabled":"Disabled"),false);
  }).catch(function(){
    renderCodex();
    showToast('<span class="chip chip-err">ERROR</span><br>Codex Fast<br><span class="dim">gateway unreachable</span>',true);
  });
}
document.getElementById("codex-fast").addEventListener("change",function(){
  CODEX.draft=this.checked;
  renderCodex();
});
document.getElementById("codex-apply").addEventListener("click",applyCodex);
/* --- account-pool routing mode (claude_account.routing) -------------------
   Same envelope discipline as the compaction card: adopt {mode, env_locked}
   from every successful response, 409 flips the local lock and re-syncs from
   a fresh GET. Apply stays inert while the draft equals the live mode. */
var ROUTING={envName:"CLAUDEX_CLAUDE_ACCOUNT_ROUTING",locked:false,mode:"disabled",draft:"disabled"};
var ROUTING_LABELS={disabled:"Disabled",fallback:"Fallback",balanced:"Balanced"};
var ROUTING_HINTS={
  disabled:"현재: Disabled — 단일 서빙 계정만 사용하고 429는 그대로 전달합니다.",
  fallback:"현재: Fallback — ready 계정으로 순차 폴백하고 쿨다운 후 자동 복귀합니다.",
  balanced:"현재: Balanced — ready 계정 풀 전체에 세션을 사용량 기반으로 고르게 분산합니다."
};
function isRoutingEnvelope(body){
  return !!body&&typeof body.env_locked==="boolean"&&
    (body.mode==="disabled"||body.mode==="fallback"||body.mode==="balanced");
}
function renderRouting(){
  document.getElementById("routing-card").classList.toggle("locked",ROUTING.locked);
  document.getElementById("routing-lock-env").textContent=ROUTING.envName;
  var select=document.getElementById("routing-select");
  select.value=ROUTING.draft;
  select.disabled=ROUTING.locked;
  var btn=document.getElementById("routing-apply");
  btn.disabled=ROUTING.locked||ROUTING.draft===ROUTING.mode;
  btn.textContent="적용";
  document.getElementById("routing-current").textContent=ROUTING_HINTS[ROUTING.mode];
}
function renderRoutingState(body){
  ROUTING.mode=body.mode;
  ROUTING.locked=!!body.env_locked;
  ROUTING.draft=body.mode;
  renderRouting();
}
function applyRouting(){
  var btn=document.getElementById("routing-apply");
  if(ROUTING.locked||btn.disabled)return;
  btn.disabled=true;btn.textContent="…";
  jfetch("/admin/providers/claude/pool/routing",{
    method:"PUT",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({mode:ROUTING.draft})
  }).then(function(r){
    if(r.status===409){
      ROUTING.locked=true;
      renderRouting();
      showToast('<span class="chip chip-err">LOCKED</span><span class="lat">409</span><br>'+esc(ROUTING.envName)+
        '<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      jfetch("/admin/providers/claude/pool/routing").then(function(g){
        if(g.ok&&isRoutingEnvelope(g.body)){
          renderRoutingState(g.body);
        }else{
          ROUTING.locked=true;
          renderRouting();
          showToast('<span class="chip chip-err">ERROR</span><br>라우팅 새로고침<br><span class="dim">could not load current state</span>',true);
        }
      });
      return;
    }
    if(!r.ok){
      renderRouting();
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+r.status+"</span><br>계정 라우팅"+
        '<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      return;
    }
    renderRoutingState(r.body);
    showToast('<span class="chip chip-ok">APPLIED</span><br>라우팅 → '+(ROUTING_LABELS[r.body.mode]||"Disabled"),false);
  }).catch(function(){
    renderRouting();
    showToast('<span class="chip chip-err">ERROR</span><br>계정 라우팅<br><span class="dim">gateway unreachable</span>',true);
  });
}
document.getElementById("routing-select").addEventListener("change",function(){
  ROUTING.draft=this.value;
  renderRouting();
});
document.getElementById("routing-apply").addEventListener("click",applyRouting);
/* The Settings rail switches the visible category card (.scard) and pins a
   deep-linkable #settings/<cat> hash; entering the accounts category fetches
   its data. */
var SETTINGS_CATS=["general","accounts"];
function setSettingsCat(cat){
  if(SETTINGS_CATS.indexOf(cat)<0)cat="general";
  var section=document.getElementById("tab-settings");
  var changed=section.dataset.cat!==cat;
  section.dataset.cat=cat;
  document.querySelectorAll("#tab-settings .rail-item").forEach(function(a){
    a.classList.toggle("on",a.getAttribute("href")==="#settings/"+cat);
  });
  if(document.body.dataset.tab==="settings"&&location.hash!=="#settings/"+cat)
    history.replaceState(null,"","#settings/"+cat);
  if(changed&&cat==="accounts")fetchAccounts();
}
document.querySelectorAll("#tab-settings .rail-item").forEach(function(a){
  a.addEventListener("click",function(ev){
    ev.preventDefault();
    setSettingsCat((this.getAttribute("href")||"").split("/")[1]);
  });
});
/* --- Claude accounts (Settings > Claude Accounts) -------------------------
   The registry GET paints rows immediately; per-account usage and the local
   login summary fill in asynchronously. Rows expand independently — opening
   one never closes another — and
   the open set survives re-renders. No interval polling: fetch on entry,
   the hero's Refresh button, and after a login succeeds. */
var ACCT_ENV="CLAUDEX_CLAUDE_ACCOUNT_ID";
var ACCT={rows:[],serving:null,local:null,locked:false,usage:{},open:{},removeArmed:{},localUsage:null,routing:{},usageFreshness:null};
function attr(s){return esc(s).replace(/"/g,"&quot;")}
function planLabel(planType){
  // claude_max -> MAX, claude_pro -> PRO; an absent plan renders as a dash.
  return planType?String(planType).replace(/^claude_/,"").toUpperCase():"—";
}
function fmtAcctDate(epochMs){
  if(typeof epochMs!=="number")return"—";
  var d=new Date(epochMs);
  function pad(n){return(n<10?"0":"")+n}
  return d.getFullYear()+"-"+pad(d.getMonth()+1)+"-"+pad(d.getDate());
}
function fmtAgo(epochSec){
  if(!epochSec)return"—";
  var m=Math.max(0,Math.round((Date.now()/1000-epochSec)/60));
  return m<1?"방금":m<60?m+"분 전":Math.floor(m/60)+"시간 "+(m%60)+"분 전";
}
/* Uses the same relative-time phrasing as fmtAgo, but for a pool/usage
   window's own age_seconds duration rather than an epoch timestamp — this is
   the per-window observation age, not the whole envelope's updated_at. */
function fmtAge(ageSeconds){
  if(typeof ageSeconds!=="number")return"—";
  var m=Math.max(0,Math.round(ageSeconds/60));
  return m<1?"방금":m<60?m+"분 전":Math.floor(m/60)+"시간 "+(m%60)+"분 전";
}
function fmtCooldownUntil(epochMs){
  if(typeof epochMs!=="number")return"—";
  var d=new Date(epochMs);
  function pad(n){return(n<10?"0":"")+n}
  return d.getFullYear()+"-"+pad(d.getMonth()+1)+"-"+pad(d.getDate())+" "+pad(d.getHours())+":"+pad(d.getMinutes());
}
function routingBadgeHtml(accountId){
  var m=ACCT.routing[accountId];
  if(!m)return"";
  if(m.routing_state==="cooldown")
    return'<span class="pill cool">쿨다운 · '+esc(fmtCooldownUntil(m.cooldown_until))+'</span>';
  if(m.routing_state==="ready")return'<span class="pill ready">라우팅 준비</span>';
  if(m.routing_state==="unavailable")
    return'<span class="pill unavailable">라우팅 불가'+(m.reason==="needs-reauth"?" · 토큰 만료":"")+'</span>';
  return"";
}
var ACCT_USAGE_SKELETON='<div class="uwin" aria-hidden="true"><div class="uwin-h"><span class="sk">세션 윈도우</span></div>'+
  '<div class="ubar"></div><div class="ureset"><span class="sk">0요일 00:00 리셋 (0시간 0분 후)</span></div></div>';
function usageWindowsHtml(u){
  // Outside active balanced mode, pool/usage carries no windows map. Each
  // metadata lookup therefore falls back to "no metadata".
  var w=u.windows||{};
  return(u.session?usageWindowHtml("세션 윈도우",u.session,true,w.session):"")+
    (u.weekly?usageWindowHtml("주간 윈도우",u.weekly,true,w.weekly):"")+
    (u.fable_weekly?usageWindowHtml("Fable 주간",u.fable_weekly,true,w.fable_weekly):"")+
    (u.monthly?usageWindowHtml("월간 윈도우",u.monthly,true,w.monthly):"");
}
function acctUsageHtml(row){
  if(row.state!=="ready")
    return'<div class="unavail">사용량을 불러올 수 없습니다.'+
      '<span class="hint">토큰이 만료됐습니다. 다시 로그인하면 이 자리에서 갱신됩니다.</span></div>';
  var u=ACCT.usage[row.id];
  if(!u)return ACCT_USAGE_SKELETON+ACCT_USAGE_SKELETON;
  if(u.status!=="ok")
    return'<div class="unavail">사용량을 불러올 수 없습니다.'+
      (u.error?'<span class="hint">'+esc(u.error)+"</span>":"")+"</div>";
  var rows=usageWindowsHtml(u);
  // A queued manual refresh never fetches inline. It renders as "queued"
  // until the coordinator's next poll completes the refresh.
  var queuedHtml=u.queued?'<div class="ureset queued">사용량 새로고침 대기 중 — 다음 폴링에 반영됩니다</div>':"";
  return(rows||'<div class="ureset">표시되는 윈도우가 없습니다</div>')+queuedHtml+
    '<div class="age">사용량 <b>'+esc(fmtAgo(u.updated_at))+'</b> 기준</div>';
}
function acctDetailHtml(row,serving){
  var isLocalSame=!!(ACCT.local&&ACCT.local.accountUuid&&ACCT.local.accountUuid===row.id);
  var kv='<div class="kv">'+
    (row.planType?'<span class="k">플랜</span><span class="v">'+esc(planLabel(row.planType))+'</span>':"")+
    '<span class="k">조직</span>'+(row.organizationName
      ?'<span class="v">'+esc(row.organizationName)+'</span>'
      :'<span class="v" style="color:var(--muted);font-style:italic">없음</span>')+
    '<span class="k">추가</span><span class="v">'+fmtAcctDate(row.createdAt)+'</span>'+
    '<span class="k">마지막 인증</span><span class="v">'+fmtAcctDate(row.lastAuthenticatedAt)+'</span>'+
    (isLocalSame?'<span class="k">로컬 CLI</span><span class="v"><span class="pill same">동일 계정</span></span>':"")+
    "</div>";
  var primary=row.state!=="ready"
    ?'<button type="button" class="primary" data-act="relogin">다시 로그인</button>'
    :serving
      ?'<button type="button" class="ghost" data-act="unserve"'+(ACCT.locked?" disabled":"")+'>서빙 해제</button>'
      :'<button type="button" class="primary" data-act="serve"'+(ACCT.locked?" disabled":"")+'>이 계정으로 서빙</button>';
  // Account removal is available unless the account is the active serving pin.
  var member=ACCT.routing[row.id];
  var coolnote=member&&member.routing_state==="cooldown"
    ?'<div class="coolnote">사용량 한도(429)로 쿨다운 중입니다.'+
      '<span class="hint">'+esc(fmtCooldownUntil(member.cooldown_until))+' 이후 자동으로 다시 참여합니다.</span></div>'
    :"";
  return coolnote+'<div class="ad-grid"><div>'+acctUsageHtml(row)+'</div><div>'+kv+'</div>'+
    '<div class="actions">'+primary+'<span class="spacer"></span>'+
    '<button type="button" class="ghost danger" data-act="remove"'+
      (serving?' disabled title="서빙 핀을 먼저 해제해야 제거할 수 있습니다"':"")+'>제거</button></div></div>';
}
function acctRowHtml(row){
  var serving=row.id===ACCT.serving;
  var open=!!ACCT.open[row.id];
  var st=serving?'<span class="st serve">서빙 중</span>'
    :row.state!=="ready"?'<span class="st reauth">⚠ 재로그인 필요</span>':"";
  return'<div class="arow'+(open?" open":"")+'" data-id="'+attr(row.id)+'" tabindex="0" role="button" aria-expanded="'+(open?"true":"false")+'">'+
    '<span class="chev" aria-hidden="true">▶</span>'+
    '<span class="em">'+esc(row.email)+'</span>'+st+routingBadgeHtml(row.id)+
    '<span class="spacer"></span>'+
    '<span class="plan-txt">'+esc(planLabel(row.planType))+'</span></div>'+
    '<div class="adetail'+(open?" show":"")+'" data-for="'+attr(row.id)+'">'+acctDetailHtml(row,serving)+'</div>';
}
/* Pool-wide usage_freshness (pool/status, balanced mode only — null in every
   other mode): fresh means every ready account's every window is at most 5
   minutes old, degraded means persistence is degraded or nothing is within
   30 minutes, partial is everything between. The manual-refresh button only
   does anything useful while balanced routing is actually polling, so it
   shares this chip's visibility. */
var POOL_FRESHNESS_LABEL={fresh:"사용량 최신",partial:"사용량 일부 지연",degraded:"사용량 지연됨"};
function renderPoolFreshness(){
  var pill=document.getElementById("pool-fresh-pill");
  var refreshBtn=document.getElementById("btn-usage-refresh");
  var label=POOL_FRESHNESS_LABEL[ACCT.usageFreshness];
  pill.hidden=!label;
  refreshBtn.hidden=!label;
  if(label){pill.className="pill "+ACCT.usageFreshness;pill.textContent=label}
}
function renderAcctList(){
  ACCT.removeArmed={};
  document.getElementById("acct-count").textContent=String(ACCT.rows.length);
  document.getElementById("acct-lock-env").textContent=ACCT_ENV;
  document.getElementById("scard-accounts").classList.toggle("locked",ACCT.locked);
  renderPoolFreshness();
  var list=document.getElementById("acct-list");
  if(!ACCT.rows.length){
    list.innerHTML='<div class="ureset" style="padding:12px 0">등록된 계정이 없습니다. 계정 추가로 브라우저 로그인을 시작하세요.</div>';
    return;
  }
  list.innerHTML=ACCT.rows.map(acctRowHtml).join("");
}
function renderLocalHero(){
  var box=document.getElementById("local-body");
  var local=ACCT.local;
  if(!local){
    box.innerHTML='<div class="org">로컬 Claude Code 로그인이 없습니다 · 게이트웨이 서빙과 무관</div>';
    return;
  }
  var u=ACCT.localUsage,usage;
  if(!u)usage=ACCT_USAGE_SKELETON+ACCT_USAGE_SKELETON;
  else if(u.status!=="ok")
    usage='<div class="unavail">사용량을 불러올 수 없습니다.'+
      (u.error?'<span class="hint">'+esc(u.error)+"</span>":"")+"</div>";
  else usage=usageWindowsHtml(u);
  box.innerHTML='<div class="who"><span class="em">'+esc(local.email)+'</span>'+
    '<span class="plan-lg">'+esc(planLabel(local.planType))+'</span></div>'+
    '<div class="org">'+(local.organizationName?esc(local.organizationName)+" · ":"")+
    "게이트웨이 서빙과 무관"+(u&&u.status==="ok"?' · <b>'+esc(fmtAgo(u.updated_at))+'</b> 기준':"")+"</div>"+usage;
}
function fetchLocalHeroUsage(){
  return jfetch("/admin/usage?provider=claude").then(function(r){
    ACCT.localUsage=r.ok&&r.body.claude?r.body.claude
      :{status:"error",error:errDetail(r.body),updated_at:Date.now()/1000};
    renderLocalHero();
  }).catch(function(){
    ACCT.localUsage={status:"error",error:"gateway unreachable",updated_at:Date.now()/1000};
    renderLocalHero();
  });
}
function fetchAccountUsage(){
  if(!ACCT.rows.length)return;
  var settle=function(message){
    ACCT.rows.forEach(function(row){
      if(row.state==="ready"&&!ACCT.usage[row.id])
        ACCT.usage[row.id]={status:"error",error:message,updated_at:Date.now()/1000};
    });
    renderAcctList();
  };
  jfetch("/admin/providers/claude/pool/usage").then(function(r){
    if(!r.ok){settle(errDetail(r.body));return}
    ACCT.usage=r.body.accounts||{};
    renderAcctList();
  }).catch(function(){settle("gateway unreachable")});
}
/* In balanced mode, `?refresh` enqueues a coalesced, rate-limited poll on
   the balanced coordinator and returns
   whatever is already cached immediately either way — it never fetches
   inline. A queued request renders via acctUsageHtml's own u.queued check;
   this only needs to adopt the response and surface it wasn't ignored. */
function refreshAccountUsage(){
  if(!ACCT.rows.length)return Promise.resolve();
  return jfetch("/admin/providers/claude/pool/usage?refresh").then(function(r){
    if(!r.ok){
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+r.status+
        '</span><br>사용량 새로고침<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      return;
    }
    ACCT.usage=r.body.accounts||{};
    renderAcctList();
    if(r.body.queued){
      showToast('<span class="chip chip-ok">QUEUED</span><br>사용량 새로고침이 대기열에 들어갔습니다'+
        '<br><span class="dim">다음 폴링에 반영됩니다</span>',false);
    }
  }).catch(function(){
    showToast('<span class="chip chip-err">ERROR</span><br>사용량 새로고침<br><span class="dim">gateway unreachable</span>',true);
  });
}
function fetchAccounts(){
  /* The accounts screen paints from four independent resources: the
     registry collection, the local-login hero, the serving pin, and the
     per-member routing status. */
  Promise.all([
    jfetch("/admin/providers/claude/accounts"),
    jfetch("/admin/providers/claude/local"),
    jfetch("/admin/providers/claude/pool/serving"),
    jfetch("/admin/providers/claude/pool/status")
  ]).then(function(rs){
    var listResp=rs[0],localResp=rs[1],servingResp=rs[2],statusResp=rs[3];
    if(!listResp.ok){
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+listResp.status+
        '</span><br>GET /admin/providers/claude/accounts<br><span class="dim">'+esc(errDetail(listResp.body))+"</span>",true);
      return;
    }
    ACCT.rows=listResp.body.accounts||[];
    ACCT.local=localResp.ok?(localResp.body.local||null):null;
    if(servingResp.ok){
      ACCT.serving=servingResp.body.account_id||null;
      ACCT.locked=!!servingResp.body.env_locked;
    }else{
      // Never paint fresh rows against a stale pin from a previous
      // refresh: an unknown pin state renders as none, with the error.
      ACCT.serving=null;ACCT.locked=false;
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+servingResp.status+
        '</span><br>GET /admin/providers/claude/pool/serving<br><span class="dim">'+
        esc(errDetail(servingResp.body))+"</span>",true);
    }
    // Same stance as the serving pin: an unknown routing status renders as no
    // badges, with the error — fresh rows never wear stale badges.
    ACCT.routing={};
    ACCT.usageFreshness=statusResp.ok?(statusResp.body.usage_freshness||null):null;
    if(statusResp.ok){
      (statusResp.body.members||[]).forEach(function(m){ACCT.routing[m.account_id]=m});
    }else{
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+statusResp.status+
        '</span><br>GET /admin/providers/claude/pool/status<br><span class="dim">'+
        esc(errDetail(statusResp.body))+"</span>",true);
    }
    renderAcctList();
    renderLocalHero();
    fetchAccountUsage();
    fetchLocalHeroUsage();
  }).catch(function(){
    showToast('<span class="chip chip-err">ERROR</span><br>GET /admin/providers/claude/accounts'+
      '<br><span class="dim">gateway unreachable</span>',true);
  });
}
function setServingAccount(accountId){
  /* Selecting pins uses PUT; clearing uses DELETE because a null PUT is
     refused. Both answer with the same {account_id, env_locked} envelope. */
  (accountId
    ?jfetch("/admin/providers/claude/pool/serving",{
      method:"PUT",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({account_id:accountId})
    })
    :jfetch("/admin/providers/claude/pool/serving",{method:"DELETE"})
  ).then(function(r){
    if(r.status===409){
      ACCT.locked=true;
      renderAcctList();
      showToast('<span class="chip chip-err">LOCKED</span><span class="lat">409</span><br>'+ACCT_ENV+
        '<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      return;
    }
    if(!r.ok){
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+r.status+"</span><br>서빙 계정 변경"+
        '<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      return;
    }
    ACCT.serving=r.body.account_id||null;
    renderAcctList();
    var servingRow=ACCT.rows.find(function(row){return row.id===ACCT.serving});
    showToast('<span class="chip chip-ok">APPLIED</span><br>서빙 계정 → '+
      (servingRow?esc(servingRow.email):"해제"),false);
  }).catch(function(){
    showToast('<span class="chip chip-err">ERROR</span><br>서빙 계정 변경<br><span class="dim">gateway unreachable</span>',true);
  });
}
function removeAccount(accountId){
  var account=ACCT.rows.find(function(row){return row.id===accountId});
  jfetch("/admin/providers/claude/accounts/"+encodeURIComponent(accountId),{method:"DELETE"}).then(function(r){
    if(!r.ok){
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+r.status+
        '</span><br>계정 제거<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      if(r.status===404)fetchAccounts();
      else renderAcctList();
      return;
    }
    showToast('<span class="chip chip-ok">REMOVED</span><br>'+esc(account?account.email:accountId)+" 제거됨",false);
    fetchAccounts();
  }).catch(function(){
    renderAcctList();
    showToast('<span class="chip chip-err">ERROR</span><br>계정 제거<br><span class="dim">gateway unreachable</span>',true);
  });
}
document.getElementById("acct-list").addEventListener("click",function(e){
  var btn=e.target.closest("button");
  if(btn){
    var det=btn.closest(".adetail");
    if(!det)return;
    var id=det.getAttribute("data-for");
    if(btn.dataset.act==="serve")setServingAccount(id);
    else if(btn.dataset.act==="unserve")setServingAccount(null);
    else if(btn.dataset.act==="relogin")openLoginModal();
    else if(btn.dataset.act==="remove"){
      if(!ACCT.removeArmed[id]){
        ACCT.removeArmed[id]=true;
        btn.classList.add("armed");btn.textContent="정말 제거";
      }else{
        delete ACCT.removeArmed[id];
        btn.disabled=true;btn.textContent="…";
        removeAccount(id);
      }
    }
    return;
  }
  var row=e.target.closest(".arow");
  if(!row)return;
  // Each row's expansion is independent state — opening one never closes another.
  var id=row.getAttribute("data-id");
  ACCT.open[id]=!ACCT.open[id];
  row.classList.toggle("open",!!ACCT.open[id]);
  row.setAttribute("aria-expanded",ACCT.open[id]?"true":"false");
  var det=this.querySelector('.adetail[data-for="'+CSS.escape(id)+'"]');
  if(det)det.classList.toggle("show",!!ACCT.open[id]);
});
document.getElementById("acct-list").addEventListener("keydown",function(e){
  if(e.key!=="Enter"&&e.key!==" ")return;
  var row=e.target.closest(".arow");
  if(!row)return;
  e.preventDefault();row.click();
});
document.getElementById("btn-add-account").addEventListener("click",function(){openLoginModal()});
document.getElementById("btn-local-refresh").addEventListener("click",function(){
  var btn=this;
  if(btn.disabled)return;
  btn.disabled=true;btn.textContent="…";
  ACCT.localUsage=null;renderLocalHero();
  fetchLocalHeroUsage().finally(function(){btn.disabled=false;btn.textContent="갱신"});
});
document.getElementById("btn-usage-refresh").addEventListener("click",function(){
  var btn=this;
  if(btn.disabled)return;
  btn.disabled=true;btn.textContent="…";
  refreshAccountUsage().finally(function(){btn.disabled=false;btn.textContent="사용량 새로고침"});
});
/* --- dashboard login (browser OAuth through the gateway's claude CLI) -----
   POST starts the session (or attaches to a running one on 409 login-active);
   a 1s poll drives the modal. The body re-renders only when the session
   status materially changes — an unconditional re-render would wipe the code
   input every second; the countdown ticks via textContent. */
var LOGIN={timer:null,lastKey:null,failures:0,attemptId:null,existingAccountId:null,openGeneration:0};
function loginStateKey(st){
  return[st.status,st.url||"",st.email||"",st.error||"",st.detail||"",st.code_prompt_detected?"1":"0"].join("|");
}
function isStaleLogin(r){
  return r.status===409&&r.body&&r.body.error&&r.body.error.code==="stale_login";
}
/* Async responses are pinned to the attempt they were dispatched for: a
   delayed callback from attempt A must never render, close, or cancel a
   session after the tab adopted attempt B (or closed). Capture
   LOGIN.attemptId at dispatch and drop the response when it no longer
   matches. */
function isCurrentAttempt(attempt){
  return attempt!==null&&attempt===LOGIN.attemptId;
}
function attemptHeaders(attempt,extra){
  return Object.assign({},extra||{},{"X-Login-Attempt":attempt});
}
function stopLoginPolling(){
  if(LOGIN.timer){clearInterval(LOGIN.timer);LOGIN.timer=null}
}
function closeLoginModal(){
  stopLoginPolling();
  // Invalidate any in-flight modal-open callback: an adoption that lands
  // after the close must not resurrect the modal with an old attempt.
  LOGIN.openGeneration++;
  LOGIN.attemptId=null;LOGIN.existingAccountId=null;
  document.getElementById("login-modal").classList.remove("open");
}
function adoptLoginAttempt(attemptId){
  /* Adopt only a well-formed attempt id — a malformed envelope must not
     leave the tab polling as bare unattached GETs. */
  if(typeof attemptId!=="string"||!attemptId){
    showToast('<span class="chip chip-err">ERROR</span><br>계정 추가'+
      '<br><span class="dim">로그인 세션 응답에 attempt_id가 없습니다.</span>',true);
    return;
  }
  LOGIN.attemptId=attemptId;
  LOGIN.existingAccountId=null;
  LOGIN.lastKey=null;LOGIN.failures=0;
  document.getElementById("login-reconnect").textContent="";
  document.getElementById("login-modal").classList.add("open");
  renderLoginModal({status:"starting"});
  stopLoginPolling();
  LOGIN.timer=setInterval(pollLogin,1000);
  pollLogin();
}
function openLoginModal(){
  // Adoption callbacks are pinned to this open operation: a delayed
  // response from an overlapping (or since-closed) open must not adopt
  // its attempt over the one the latest open established.
  var generation=++LOGIN.openGeneration;
  jfetch("/admin/providers/claude/login",{
    method:"POST",headers:{"Content-Type":"application/json"},body:"{}"
  }).then(function(r){
    if(generation!==LOGIN.openGeneration)return;
    var code=r.body&&r.body.error&&r.body.error.code;
    if(r.ok){
      adoptLoginAttempt(r.body&&r.body.attempt_id);
      return;
    }
    // login-active: another dashboard tab (or this one) already runs a
    // session — attach to it via a bare discovery GET instead of erroring.
    if(r.status===409&&code==="login-active"){
      jfetch("/admin/providers/claude/login").then(function(g){
        if(generation!==LOGIN.openGeneration)return;
        if(g.ok&&g.body&&g.body.attempt_id){
          adoptLoginAttempt(g.body.attempt_id);
          return;
        }
        showToast('<span class="chip chip-err">ERROR</span><br>계정 추가'+
          '<br><span class="dim">활성 로그인 세션에 연결할 수 없습니다.</span>',true);
      }).catch(function(){
        if(generation!==LOGIN.openGeneration)return;
        showToast('<span class="chip chip-err">ERROR</span><br>계정 추가<br><span class="dim">gateway unreachable</span>',true);
      });
      return;
    }
    showToast('<span class="chip chip-err">'+(r.status===409?"LOCKED":"ERROR")+'</span><span class="lat">'+r.status+
      '</span><br>계정 추가<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
  }).catch(function(){
    if(generation!==LOGIN.openGeneration)return;
    showToast('<span class="chip chip-err">ERROR</span><br>계정 추가<br><span class="dim">gateway unreachable</span>',true);
  });
}
function pollLogin(){
  var attempt=LOGIN.attemptId;
  if(!attempt){stopLoginPolling();return}
  jfetch("/admin/providers/claude/login",{headers:attemptHeaders(attempt)}).then(function(r){
    if(!isCurrentAttempt(attempt))return;
    if(isStaleLogin(r)){
      // A newer login superseded this tab's attempt: bow out quietly.
      closeLoginModal();fetchAccounts();
      return;
    }
    if(!r.ok){loginPollFailed();return}
    LOGIN.failures=0;
    document.getElementById("login-reconnect").textContent="";
    var st=r.body||{};
    if(st.status==="succeeded"){
      closeLoginModal();
      showToast('<span class="chip chip-ok">ADDED</span><br>'+
        esc((st.account&&st.account.email)||"계정")+" 등록됨",false);
      fetchAccounts();
      return;
    }
    if(st.status==="cancelled"||st.status==="idle"){closeLoginModal();return}
    renderLoginModal(st);
    if(st.status==="failed")stopLoginPolling();
    updateLoginCountdown(st);
  }).catch(function(){
    if(isCurrentAttempt(attempt))loginPollFailed();
  });
}
function loginPollFailed(){
  LOGIN.failures++;
  if(LOGIN.failures>=5)
    document.getElementById("login-reconnect").textContent="게이트웨이 연결을 다시 시도하는 중…";
}
function renderLoginModal(st){
  var key=loginStateKey(st);
  if(key===LOGIN.lastKey)return;
  LOGIN.lastKey=key;
  var body=document.getElementById("login-modal-body");
  var actions=document.getElementById("login-modal-actions");
  var codeInput=document.getElementById("login-code-input");
  var codeValue=codeInput?codeInput.value:"";
  var cancelBtn='<button class="cancel" data-lact="cancel">취소</button>';
  if(st.status==="awaiting-browser"){
    var safeUrl=/^https:\/\//.test(st.url||"")?st.url:null;
    body.innerHTML='<p>브라우저에서 Anthropic 로그인을 완료하세요. 로그인이 끝나면 이 창이 자동으로 진행됩니다.</p>'+
      (safeUrl
        ?'<div class="lgurl"><span class="tx">'+esc(safeUrl)+'</span><button class="cp" id="login-url-copy" title="복사"></button></div>'+
          '<a class="lgopen" href="'+attr(safeUrl)+'" target="_blank" rel="noopener">브라우저에서 열기 ↗</a>'
        :'<p><span class="sk">로그인 URL을 기다리는 중…</span></p>')+
      '<div class="lgrow"><input id="login-code-input" spellcheck="false" placeholder="브라우저가 표시한 코드 붙여넣기">'+
      '<button type="button" class="primary" data-lact="code">제출</button></div>'+
      '<div class="lghint">브라우저가 자동으로 돌아오지 않으면 표시된 코드를 붙여넣으세요. <span id="login-countdown"></span></div>';
    if(safeUrl)wireCopy(document.getElementById("login-url-copy"),safeUrl);
    document.getElementById("login-code-input").value=codeValue;
    actions.innerHTML=cancelBtn;
  }else if(st.status==="completing"){
    body.innerHTML='<p><span class="sk">로그인을 완료하는 중…</span></p>';
    actions.innerHTML=cancelBtn;
  }else if(st.status==="awaiting-replace"){
    // Exactly two buttons, mirroring the CLI's [y/N] replace prompt. The
    // confirmation names the record being replaced; declining cancels.
    LOGIN.existingAccountId=st.existing_account_id||null;
    body.innerHTML='<p><b>'+esc(st.email||"")+'</b> 계정이 이미 등록되어 있습니다.</p>'+
      '<p class="warn">교체하면 기존 자격증명을 이번 로그인으로 덮어씁니다.</p>';
    actions.innerHTML='<button class="cancel" data-lact="decline">교체 안 함</button>'+
      '<button class="go" data-lact="replace">교체</button>';
  }else if(st.status==="failed"){
    body.innerHTML='<div class="unavail">로그인에 실패했습니다.'+
      (st.error?'<span class="hint">'+esc(st.error)+"</span>":"")+"</div>";
    actions.innerHTML='<button class="cancel" data-lact="close">닫기</button>';
  }else{
    // starting (server) or the optimistic local cancelling state.
    body.innerHTML='<p><span class="sk">'+
      (st.status==="cancelling"?"취소하는 중…":"로그인 세션을 시작하는 중…")+"</span></p>";
    actions.innerHTML=st.status==="cancelling"?"":cancelBtn;
  }
}
function updateLoginCountdown(st){
  var el=document.getElementById("login-countdown");
  if(!el)return;
  if(!st.expires_at){el.textContent="";return}
  var s=Math.max(0,Math.round(st.expires_at-Date.now()/1000));
  el.textContent="남은 시간 "+Math.floor(s/60)+"분 "+(s%60)+"초";
}
function submitLoginCode(){
  var input=document.getElementById("login-code-input");
  if(!input)return;
  var code=input.value.trim();
  if(!code)return;
  var attempt=LOGIN.attemptId;
  if(!attempt)return;
  jfetch("/admin/providers/claude/login/code",{
    method:"POST",headers:attemptHeaders(attempt,{"Content-Type":"application/json"}),
    body:JSON.stringify({code:code})
  }).then(function(r){
    if(!isCurrentAttempt(attempt))return;
    if(isStaleLogin(r)){closeLoginModal();fetchAccounts();return}
    if(!r.ok)
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+r.status+
        '</span><br>코드 제출<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
  }).catch(function(){
    if(!isCurrentAttempt(attempt))return;
    showToast('<span class="chip chip-err">ERROR</span><br>코드 제출<br><span class="dim">gateway unreachable</span>',true);
  });
}
function cancelLoginSession(){
  var attempt=LOGIN.attemptId;
  if(!attempt){closeLoginModal();return}
  jfetch("/admin/providers/claude/login",{
    method:"DELETE",headers:attemptHeaders(attempt)
  }).then(function(r){
    if(!isCurrentAttempt(attempt))return;
    if(isStaleLogin(r)){closeLoginModal();fetchAccounts();return}
    renderLoginModal({status:"cancelling"});
  }).catch(function(){});
}
document.getElementById("login-modal").addEventListener("click",function(e){
  var btn=e.target.closest("button");
  if(!btn||!btn.dataset.lact)return;
  var act=btn.dataset.lact;
  if(act==="close"){closeLoginModal();return}
  if(act==="code"){submitLoginCode();return}
  if(act==="cancel"||act==="decline"){
    // Explicit cancel only: backdrop/Escape never close a live login.
    // Declining a replacement IS cancelling — the daemon has no decline verb.
    cancelLoginSession();
    return;
  }
  if(act==="replace"){
    var attempt=LOGIN.attemptId;
    if(!attempt)return;
    jfetch("/admin/providers/claude/login/replace",{
      method:"POST",headers:attemptHeaders(attempt,{"Content-Type":"application/json"}),
      body:JSON.stringify({existing_account_id:LOGIN.existingAccountId})
    }).then(function(r){
      if(!isCurrentAttempt(attempt))return;
      if(isStaleLogin(r)){closeLoginModal();fetchAccounts();return}
      if(!r.ok)
        showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+r.status+
          '</span><br>교체 확인<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
    }).catch(function(){});
  }
});
document.getElementById("login-modal").addEventListener("keydown",function(e){
  if(e.key==="Enter"&&e.target.id==="login-code-input"){e.preventDefault();submitLoginCode()}
});
function apply(){
  if(DIR.locked)return;
  var count=dirtyCount();
  jfetch("/admin/settings/mapping",{
    method:"PUT",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({model_map:DIR.mapping})
  }).then(function(r){
    if(r.status===409){
      DIR.locked=true;renderChrome();
      showToast('<span class="chip chip-err">LOCKED</span><span class="lat">409</span><br>'+esc(DIR.envName)+
        '<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      return;
    }
    if(!r.ok){
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+r.status+"</span><br>Apply Claude → Codex"+
        '<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      return;
    }
    DIR.LIVE=r.body.model_map||{};
    DIR.mapping=Object.assign({},DIR.LIVE);
    DIR.sel=null;
    render();
    showToast('<span class="chip chip-ok">APPLIED</span><br>Claude → Codex map · '+count+" changes",false);
  }).catch(function(){
    showToast('<span class="chip chip-err">ERROR</span><br>Apply Claude → Codex'+
      '<br><span class="dim">gateway unreachable</span>',true);
  });
}
document.getElementById("applybtn").addEventListener("click",apply);
document.getElementById("discardbtn").addEventListener("click",function(){
  DIR.mapping=Object.assign({},DIR.LIVE);
  DIR.sel=null;
  document.getElementById("wdel").classList.remove("show");
  // back to the live board: drop staged nodes and re-stack from scratch
  addedTargets=[];DIR.sources=[];DIR.targets=[];
  render();
});
function runConnTest(){
  var n=document.getElementById("ct-in").value.trim();
  var btn=document.getElementById("ct-btn");
  if(!n||btn.disabled)return;
  btn.disabled=true;btn.textContent="…";
  var label='"'+esc(n)+'"';
  jfetch("/admin/test",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({target:n})
  }).then(function(r){
    btn.disabled=false;btn.textContent="Test";
    var b=r.body||{};
    if(r.ok&&b.ok){
      showToast('<span class="chip chip-ok">OK</span><span class="lat">'+(b.latency_ms!=null?b.latency_ms+"ms":"")+
        "</span><br>"+label+' 응답 확인<br><span class="dim">response.model: '+esc(b.response_model||n)+"</span>",false);
    }else{
      var code=b.status!=null?b.status:r.status;
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+esc(code)+"</span><br>"+label+
        '<br><span class="dim">'+esc(b.detail||errDetail(b))+"</span>",true);
    }
  }).catch(function(){
    btn.disabled=false;btn.textContent="Test";
    showToast('<span class="chip chip-err">ERROR</span><br>'+label+
      '<br><span class="dim">gateway unreachable</span>',true);
  });
}
document.getElementById("ct-btn").addEventListener("click",runConnTest);
document.getElementById("ct-in").addEventListener("keydown",function(ev){if(ev.key==="Enter")runConnTest()});
/* Unrecognized or missing prefixes fall back to the Codex icon, matching
   providerIcon's own default. */
document.getElementById("ct-in").addEventListener("input",function(){
  var m=new RegExp("^\\s*("+ROUTE_PROVIDERS.join("|")+"):").exec(this.value);
  document.getElementById("ct-in-icon").innerHTML=providerIcon(m?m[1]:"codex",14);
});
window.addEventListener("resize",drawWires);
function fitView(){
  var nodes=layer.querySelectorAll(".node");
  if(!nodes.length){pan={x:0,y:0};zoom=1;applyView();drawWires();return}
  var minY=Infinity,maxY=-Infinity;
  nodes.forEach(function(n){
    minY=Math.min(minY,n.offsetTop);
    maxY=Math.max(maxY,n.offsetTop+n.offsetHeight);
  });
  var W=board.clientWidth,H=board.clientHeight;
  var padTop=18,padBottom=66;   // Leave room for the floating zoom toolbar at the bottom.
  var usable=H-padTop-padBottom;
  zoom=Math.max(.4,Math.min(1,usable/(maxY-minY)));
  pan.x=(W-W*zoom)/2;
  pan.y=padTop+(usable-(maxY-minY)*zoom)/2-minY*zoom;
  applyView();drawWires();
}
/* The board cannot be measured while its tab is hidden, so the first time
   the map tab becomes visible it gets its fit-to-view pass here. */
var mapNeedsFit=true;
function setTab(t){
  document.body.dataset.tab=t;
  document.querySelectorAll("nav.tabs a").forEach(function(a){a.classList.toggle("on",a.dataset.t===t)});
  if(t==="settings"){
    // Settings keeps its category in the hash so the deep link round-trips.
    var cat=document.getElementById("tab-settings").dataset.cat||"general";
    if(location.hash!=="#settings/"+cat)history.replaceState(null,"","#settings/"+cat);
    if(cat==="accounts")fetchAccounts();
  }else if(location.hash!=="#"+t){history.replaceState(null,"","#"+t)}
  if(t==="map"){drawWires();if(mapNeedsFit){mapNeedsFit=false;fitView()}}
  if(t==="status")fetchUsage();
  if(t==="log")fetchLogs();
  syncLogTimer();
}
var TAB_NAMES=["settings","status","map","log"];
/* --- subscription usage (merged into the Status tab's provider cards) ----- */
function fmtPct(v){return (Math.round(v*10)/10)+"%"}
function fmtReset(epochSec){
  if(!epochSec)return"리셋 시각 정보 없음";
  var ms=epochSec*1000,diffMin=Math.max(0,Math.round((ms-Date.now())/60000));
  var d=new Date(ms);
  function pad(n){return(n<10?"0":"")+n}
  var days=["일","월","화","수","목","금","토"];
  var abs=(d.toDateString()===new Date().toDateString())
    ?pad(d.getHours())+":"+pad(d.getMinutes())
    :days[d.getDay()]+"요일 "+pad(d.getHours())+":"+pad(d.getMinutes());
  var rel=diffMin>=1440?Math.floor(diffMin/1440)+"일 "+Math.floor(diffMin%1440/60)+"시간 후"
    :diffMin>=60?Math.floor(diffMin/60)+"시간 "+(diffMin%60)+"분 후"
    :diffMin+"분 후";
  return abs+" 리셋 ("+rel+")";
}
/* colorPct (accounts screens only) mirrors the bar's state color onto the
   % readout; the Status cards keep their original neutral readout. meta is
   this window's pool/usage observation metadata ({age_seconds, source,
   state}, balanced mode only) — absent everywhere else, so the reset line
   renders exactly as before when it is undefined. */
function usageWindowHtml(label,win,colorPct,meta){
  var pct=Math.round(win.used_percent*10)/10;
  var cls=pct>=90?"err":pct>=70?"warn":"";
  var metaHtml=meta?' · <span class="wmeta '+esc(meta.state||"")+'">'+esc(fmtAge(meta.age_seconds))+" · "+esc(meta.source)+'</span>':"";
  return'<div class="uwin"><div class="uwin-h"><span>'+esc(label)+'</span><span class="upct'+(colorPct&&cls?" "+cls:"")+'">'+fmtPct(pct)+'</span></div>'+
    '<div class="ubar"><i class="'+cls+'" style="width:'+Math.min(100,Math.max(0,pct))+'%"></i></div>'+
    '<div class="ureset">'+fmtReset(win.resets_at)+metaHtml+'</div></div>';
}
/* Renders whenever the provider reports a credit count, so this line is the
   single place credits are stated. The button is part of it only when there
   is something to spend — a card with none cannot be misclicked at all — and
   carries the count so the dialog needs no second source of truth. */
function resetCreditHtml(provider,data){
  var count=data.reset_credits_available;
  if(provider!=="codex"||typeof count!=="number")return"";
  return'<div class="uact"><span class="lbl">리셋 크레딧</span><span class="num">'+count+"개</span>"+
    (count>0
      ?'<button id="codex-reset-go" data-n="'+count+'">1개 사용</button>'+
        '<span class="hint">한도 창을 즉시 리셋합니다</span>'
      :'<span class="hint">쓸 수 있는 크레딧이 없습니다</span>')+"</div>";
}
/* One bar per line, not one per value: the loading card reads as a few calm
   rows instead of a mosaic. Each bar still carries the widest text of the
   line it stands in for, so that line keeps its exact height — and where a
   non-text element sets the height (the reset-credit button, the plan pill)
   the bar is that element, so the swap stays exact there too.

   Same rows as a loaded body: every provider reports a session and a weekly
   window, Claude adds a third only on plans that expose a Fable quota. */
/* Latin only: Korean inside the mono pill falls back to a taller font and
   makes the placeholder a pixel taller than the plan it stands in for. */
var PLAN_SKELETON='<b class="sk">STANDARD</b>';
function usageSkeletonHtml(provider){
  var row='<div class="uwin"><div class="uwin-h"><span class="sk">5시간 윈도우</span></div>'+
    '<div class="ubar"></div>'+
    '<div class="ureset"><span class="sk">0요일 00:00 리셋 (0시간 0분 후)</span></div></div>';
  return'<div aria-hidden="true">'+row+row+
    '<div class="umeta"><span class="sk">마지막 갱신 00:00:00</span></div>'+
    (provider==="codex"
      ?'<div class="uact"><button class="sk" disabled>리셋 크레딧 0개 · 1개 사용</button></div>':"")+
    "</div>";
}
function renderUsageProvider(provider,data){
  // The plan sits above, in the card's status box, so it is written to its
  // own hook rather than into the body this function rebuilds.
  document.getElementById("usage-plan-"+provider).innerHTML=
    data?(data.plan_type?"<b>"+esc(data.plan_type)+"</b> 플랜":""):PLAN_SKELETON;
  var body=document.getElementById("usage-body-"+provider);
  if(!data){
    body.innerHTML=usageSkeletonHtml(provider);
    return;
  }
  var meta='<div class="umeta">마지막 갱신 '+fmtLogTs(data.updated_at||Date.now()/1000)+"</div>";
  if(data.status==="ok"){
    // Do not render rows for windows absent from the response, such as Codex's five-hour window.
    var rows=(data.session?usageWindowHtml("5시간 윈도우",data.session):"")+
      (data.weekly?usageWindowHtml("주간 윈도우",data.weekly):"")+
      (data.fable_weekly?usageWindowHtml("Fable 주간",data.fable_weekly):"")+
      (data.monthly?usageWindowHtml("월간 윈도우",data.monthly):"");
    body.innerHTML=(rows||'<div class="ureset">표시되는 윈도우가 없습니다</div>')+meta+
      resetCreditHtml(provider,data);
    return;
  }
  var detail=data.status==="unavailable"?"사용량을 조회할 수 없습니다":"조회에 실패했습니다";
  body.innerHTML='<div class="stat'+(data.status==="unavailable"?"":" err")+'">● <span class="detail">'+
    detail+"</span>"+(data.error?'<div class="codeblock">'+esc(data.error)+"</div>":"")+"</div>"+meta;
}
/* The refresh button sits in the static card header, outside the re-rendered
   body, so every settle path must hand back its label and enabled state. */
function restoreUsageButtons(providers){
  providers.forEach(function(p){
    var btn=document.querySelector('.urefresh[data-provider="'+p+'"]');
    if(btn){btn.disabled=false;btn.textContent="갱신"}
  });
}
/* Without a provider, refresh all visible cards on tab entry; with one,
   refresh only that card. This list stays built-in-only because custom
   providers have no usage API. */
function fetchUsage(provider){
  var targets=provider?[provider]:["claude","codex","kimi","grok"].filter(function(p){
    return PROVIDER_VISIBLE[p]!==false;
  });
  if(!provider)targets.forEach(function(p){renderUsageProvider(p,null)});
  /* The skeleton only means "probe in flight": every failure path must still
     settle each card into an error state, or the animation runs forever. */
  var failAll=function(message){
    targets.forEach(function(p){
      renderUsageProvider(p,{status:"error",error:message,updated_at:Date.now()/1000});
    });
  };
  jfetch("/admin/usage"+(provider?"?provider="+provider:"")).then(function(r){
    if(!r.ok){
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+r.status+
        '</span><br>GET /admin/usage<br><span class="dim">'+esc(errDetail(r.body))+"</span>",true);
      failAll(errDetail(r.body));
      return;
    }
    targets.forEach(function(p){
      var data=r.body[p];
      if(data)renderUsageProvider(p,data);
      else renderUsageProvider(p,{status:"error",error:"usage response did not include this provider",updated_at:Date.now()/1000});
    });
  }).catch(function(){
    showToast('<span class="chip chip-err">ERROR</span><br>GET /admin/usage'+
      '<br><span class="dim">gateway unreachable</span>',true);
    failAll("gateway unreachable");
  }).finally(function(){restoreUsageButtons(targets)});
}
document.getElementById("tab-status").addEventListener("click",function(ev){
  var btn=ev.target.closest?ev.target.closest(".urefresh"):null;
  if(!btn||btn.disabled)return;
  btn.disabled=true;btn.textContent="…";
  fetchUsage(btn.dataset.provider);
});
/* --- spending a Codex reset credit ---------------------------------------
   The credit is gone once the gateway forwards the request, so the card's
   button only arms the dialog: confirming there is the single path that
   reaches /admin/providers/codex/reset-credit. */
var RESET_OUTCOMES={
  reset:{spent:true,text:"한도 창을 리셋했습니다"},
  // The backend reports these so the card can say why nothing changed.
  nothing_to_reset:{spent:false,text:"리셋할 한도가 없어 크레딧을 쓰지 않았습니다"},
  no_credit:{spent:false,text:"사용 가능한 리셋 크레딧이 없습니다"},
  already_redeemed:{spent:false,text:"이미 처리된 요청입니다"}
};
function closeResetModal(){document.getElementById("reset-modal").classList.remove("open")}
document.getElementById("tab-status").addEventListener("click",function(ev){
  var btn=ev.target.closest?ev.target.closest("#codex-reset-go"):null;
  if(!btn)return;
  document.getElementById("reset-modal-body").innerHTML=
    "보유한 리셋 크레딧 <b>"+esc(btn.dataset.n)+"개</b> 중 <b>1개</b>를 사용해 "+
    "Codex 한도 창을 즉시 리셋합니다.";
  var go=document.getElementById("reset-confirm");
  go.disabled=false;go.textContent="사용";
  document.getElementById("reset-modal").classList.add("open");
  // Focus the safe choice so Enter or a stray space cannot spend the credit.
  document.getElementById("reset-cancel").focus();
});
document.getElementById("reset-cancel").addEventListener("click",closeResetModal);
document.getElementById("reset-modal").addEventListener("click",function(ev){
  if(ev.target===this)closeResetModal();
});
document.addEventListener("keydown",function(ev){if(ev.key==="Escape")closeResetModal()});
document.getElementById("reset-confirm").addEventListener("click",function(){
  if(this.disabled)return;
  this.disabled=true;this.textContent="…";
  jfetch("/admin/providers/codex/reset-credit",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:"{}"
  }).then(function(r){
    closeResetModal();
    var b=r.body||{};
    if(r.ok&&b.status==="ok"){
      var outcome=RESET_OUTCOMES[b.outcome]||{spent:false,text:String(b.outcome)};
      showToast('<span class="chip chip-'+(outcome.spent?"ok":"err")+'">'+
        (outcome.spent?"RESET":"NO-OP")+"</span><br>"+esc(outcome.text),!outcome.spent);
    }else{
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+r.status+
        '</span><br>리셋 크레딧 사용<br><span class="dim">'+esc(b.error||errDetail(b))+"</span>",true);
    }
    fetchUsage("codex");   // windows and the remaining count both moved
  }).catch(function(){
    closeResetModal();
    showToast('<span class="chip chip-err">ERROR</span><br>리셋 크레딧 사용'+
      '<br><span class="dim">gateway unreachable</span>',true);
  });
});
document.getElementById("log-refresh").addEventListener("click",fetchLogs);
document.getElementById("log-live").addEventListener("click",function(){
  logLive=!logLive;
  this.classList.toggle("on",logLive);
  if(logLive)fetchLogs();
  syncLogTimer();
});
document.querySelectorAll("nav.tabs a").forEach(function(a){
  a.addEventListener("click",function(ev){ev.preventDefault();setTab(this.dataset.t)});
});
window.addEventListener("hashchange",function(){
  // "#settings/accounts" style deep links carry a category segment; the tab
  // name is always the first segment.
  var parts=location.hash.slice(1).split("/");
  if(TAB_NAMES.indexOf(parts[0])>=0)setTab(parts[0]);
  if(parts[0]==="settings")setSettingsCat(parts[1]||"general");
});
function boot(){
  document.getElementById("ep").textContent=location.host;
  // Fill the usage bodies before the first request goes out: an empty body
  // would paint short and then grow, which is the jump the skeleton exists
  // to remove.
  ["claude","codex","kimi","grok"].forEach(function(p){renderUsageProvider(p,null)});
  document.getElementById("ct-in-icon").innerHTML=codexIcon(14);
  jfetch("/api/hello").then(function(hello){
    authRequired=!!(hello.body&&hello.body.local_auth_required);
    if(authRequired&&!localToken)promptLocalToken();
    return Promise.all([
      jfetch("/admin/settings/mapping"),
      jfetch("/health"),
      jfetch("/admin/providers/codex/models"),
      jfetch("/admin/providers/kimi/models"),
      jfetch("/admin/providers/grok/models"),
      jfetch("/admin/settings/log-level"),
      jfetch("/admin/settings/compaction"),
      jfetch("/admin/settings/codex"),
      jfetch("/admin/providers/claude/pool/routing")
    ]);
  }).then(function(results){
    var mapping=results[0],health=results[1];
    var codexCatalog=results[2],kimiCatalog=results[3],grokCatalog=results[4],loglevel=results[5];
    var compactionResp=results[6],codexResp=results[7],routingResp=results[8];
    if(loglevel.ok)renderLogLevel(loglevel.body);
    if(compactionResp.ok){
      renderCompactionState(compactionResp.body);
    }else{
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+compactionResp.status+
        '</span><br>GET /admin/settings/compaction<br><span class="dim">'+esc(errDetail(compactionResp.body))+"</span>",true);
    }
    if(codexResp.ok&&isCodexEnvelope(codexResp.body)){
      renderCodexState(codexResp.body);
    }else{
      renderCodex();
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+codexResp.status+
        '</span><br>GET /admin/settings/codex<br><span class="dim">'+esc(errDetail(codexResp.body))+"</span>",true);
    }
    if(routingResp.ok&&isRoutingEnvelope(routingResp.body)){
      renderRoutingState(routingResp.body);
    }else{
      renderRouting();
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+routingResp.status+
        '</span><br>GET /admin/providers/claude/pool/routing<br><span class="dim">'+esc(errDetail(routingResp.body))+"</span>",true);
    }
    if(mapping.ok){
      DIR.LIVE=mapping.body.model_map||{};
      DIR.locked=!!(mapping.body.env_locked||{}).model_map;
      configureCustomProviders(mapping.body.custom_providers);
      renderFacts(mapping.body);
    }else{
      showToast('<span class="chip chip-err">ERROR</span><span class="lat">'+mapping.status+
        '</span><br>GET /admin/settings/mapping<br><span class="dim">'+esc(errDetail(mapping.body))+"</span>",true);
    }
    // A catalog only feeds the add-node suggestions, and Kimi's routinely 401s
    // when the gateway has no Kimi login, so failures stay silent here.
    if(codexCatalog.ok)CATALOG.codex=catalogIds(codexCatalog.body);
    if(kimiCatalog.ok)CATALOG.kimi=catalogIds(kimiCatalog.body);
    if(grokCatalog.ok)CATALOG.grok=catalogIds(grokCatalog.body);
    DIR.mapping=Object.assign({},DIR.LIVE);
    if(health.body&&health.body.providers){
      renderHealth(health.body);
      setProviderVisibility(health.body);
      renderProviderCards(health.body);
    }
    renderChrome();
    var bootParts=location.hash.slice(1).split("/");
    var bootTab=TAB_NAMES.indexOf(bootParts[0])>=0?bootParts[0]:"settings";
    setTab(bootTab);
    if(bootTab==="settings"&&bootParts[1])setSettingsCat(bootParts[1]);
    render();
    if(document.body.dataset.tab==="map"){mapNeedsFit=false;fitView()}
  });
}
boot();
