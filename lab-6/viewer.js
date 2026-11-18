const NS = 'http://www.w3.org/2000/svg';
const pretty = value => JSON.stringify(value, (_key, v) => typeof v==='number'?Number(v.toPrecision(6)):v, 2);
const names = {client:'Employee’s client', server:'Prediction server', c1:'Company 1 · Client', c2:'Company 2 · Client', c3:'Company 3 · Client', instructor:'Instructor evaluation'};
// Static scenario figures precede the executable workflow in each part.
function scenarioFigure(parent, part, element) {
  const he=part==='he';
  const title=he?'Part B scenario: private salary prediction':'Part A scenario: collaborative salary-model training';
  const figure=element('figure',{class:'scenario'},parent);
  const diagram=document.createElementNS(NS,'svg');
  diagram.setAttribute('viewBox',he?'0 0 900 395':'0 0 900 465');
  diagram.setAttribute('role','img');diagram.setAttribute('aria-label',title);
  figure.append(diagram);
  const add=(tag,attrs,text)=>{const n=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs)) n.setAttribute(k,v);if(text)n.textContent=text;diagram.append(n);return n;};
  const id='scenario-'+Math.random().toString(36).slice(2);
  const defs=add('defs',{});
  for(const [key,color] of [['out','#087d79'],['back','#b36a20']]) {
    const marker=document.createElementNS(NS,'marker');
    for(const [k,v] of Object.entries({id:id+key,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto'}))marker.setAttribute(k,v);
    const path=document.createElementNS(NS,'path');path.setAttribute('d','M 0 0 L 10 5 L 0 10 Z');path.setAttribute('fill',color);marker.append(path);defs.append(marker);
  }
  const box=(x,y,w,h,curious=false)=>add('rect',{x,y,width:w,height:h,rx:14,fill:curious?'#fff8ee':'#eaf7f3',stroke:curious?'#b36a20':'#087d79','stroke-width':2,'stroke-dasharray':curious?'8 5':'none'});
  const text=(x,y,value,size=17,bold=false,anchor='start')=>add('text',{x,y,'font-size':size,'font-weight':bold?700:400,'text-anchor':anchor,fill:'#173047'},value);
  const arrow=(x1,y1,x2,y2,back=false)=>add('line',{x1,y1,x2,y2,stroke:back?'#b36a20':'#087d79','stroke-width':2.5,'marker-end':`url(#${id+(back?'back':'out')})`});
  const label=(x,y,value)=>add('text',{x,y,'text-anchor':'middle','font-size':16,fill:'#173047',stroke:'#f8fbfc','stroke-width':6,'paint-order':'stroke'},value);
  if(he) {
    box(15,25,260,300);box(605,25,280,300,true);
    text(35,60,'Employee’s client',21,true);
    text(35,89,'Trusted device',16,true);
    text(35,130,'Experience + education');
    text(35,168,'Create keys; encrypt input');
    text(35,208,'Secret key stays here',17,true);
    text(35,276,'Decrypt salary prediction');
    text(625,60,'PayWise server',21,true);
    text(625,89,'Honest but curious',16,true);
    text(625,130,'Holds plaintext model w, b');
    text(625,182,'Receives public context');
    text(625,216,'Computes on ciphertext');
    text(625,276,'Cannot decrypt input/reply',16,true);
    arrow(275,115,605,115);label(440,102,'1. Public + evaluation keys');
    arrow(275,200,605,200);label(440,187,'2. Encrypted features');
    arrow(605,285,275,285,true);label(440,272,'3. Encrypted prediction');
    text(450,365,'Protected from PayWise: employee features and prediction',18,true,'middle');
  } else {
    box(225,15,450,130,true);
    text(450,49,'PayWise aggregation server',22,true,'middle');
    text(450,79,'Honest but curious',17,true,'middle');
    text(450,112,'Sees each client’s gradients and sample count',17,false,'middle');
    for(const [i,x,n] of [[1,15,20],[2,320,60],[3,625,120]]) {
      box(x,285,260,160);
      const mid=x+130,serverX=450+(i-2)*110;
      arrow(serverX-14,145,mid-14,285);
      arrow(mid+14,285,serverX+14,145,true);
      label(mid,195,'↓ Global model parameters');
      label(mid,234,'↑ Gradient + count');
      text(x+20,318,`Company ${i} · Client`,20,true);
      text(x+20,351,`${n} private employee records`,17);
      text(x+20,384,'Compute one local gradient',17);
      text(x+20,420,'Raw records stay here',17,true);
    }
  }
  element('figcaption',{text:he?'Solid green boundary: trusted client. Dashed amber boundary: server that may inspect everything it receives. The secret key and plaintext features never cross to PayWise.':'Each company keeps its own training data. PayWise broadcasts a common model; clients return gradient vectors and sample counts. No raw training records are uploaded.'},figure);
  const threat=element('div',{class:'threat-model'},parent);
  element('h3',{text:'Threat model'},threat);
  element('p',{text:he?'Assume PayWise follows the prediction protocol but tries to learn the employee’s private input from received messages. The client keeps its secret key safe. PayWise sees public/evaluation keys, ciphertexts, and its own model weights; it does not receive plaintext features or a decrypted prediction. This exercise does not verify that a malicious server computed the correct answer.':'Assume PayWise follows aggregation but may inspect every company’s uploaded update. Companies follow the local-training protocol. Keeping records local avoids pooling employee data, but PayWise still sees individual gradients and sample counts. Updates can leak information: this exercise does not add secure aggregation or differential privacy.'},threat);
  const task=element('div',{class:'part-task'},parent);
  element('h3',{text:'Your task'},task);
  if(!he) element('p',{class:'training-goal',text:'Goal: achieve held-out MSE ≤ 3.0 $k². Adjust the rounds and learning rate, rerun training, and record the settings that reach the goal.'},task);
  element('p',{text:he?'Implement client_encrypt using TenSEAL and serialized ciphertext bytes, the server’s encrypted_predict function for y = wᵀx + b, and client_decrypt to recover the returned prediction on the client. The starting model is salary ($k) = 12 × experience + 8 × education + 35, with both features scaled to [0, 1]. Execute the workflow, follow the messages, and compare the three decrypted predictions with the instructor’s reference.':'Implement client_gradient on each company client, aggregate_gradients on the server, and server_rounds. Set the training rounds and learning rate in the controls below (starting values: 1 round and 0.1). Each client computes one MSE gradient; the server averages using sample counts 20, 60, 120 and subtracts learning_rate × mean_gradient. In server_rounds, yield theta.copy() after each update so the viewer can record it. The communication adapter is provided.'},task);
  if(he) {
    const reference=element('p',{},task);
    element('a',{text:'TenSEAL documentation',href:'https://openmined.github.io/TenSEAL/',target:'_blank',rel:'noopener noreferrer'},reference);
    element('span',{text:' — reference for CKKS vectors, serialization, encrypted arithmetic, and decryption.'},reference);
  }
}

function renderPart({model, el}, part, shared) {
    const config = model.get('config');
    const root = document.createElement('div'); root.className = 'protocol-lab'; el.append(root);
    // Text received from student Python is always rendered with textContent.
    function element(tag, attrs={}, parent=root) {
      const node = document.createElement(tag);
      for (const [key,value] of Object.entries(attrs)) {
        if (key === 'text') node.textContent = value;
        else if (key === 'class') node.className = value;
        else node.setAttribute(key,value);
      }
      parent.append(node); return node;
    }
    const heading = element(part==='linked'?'h3':'h2');
    const intro = element('p');
    if(part!=='linked') scenarioFigure(root, part, element);
    const inputsLabel = element('label',{text:'Client inputs · experience / 20 and education / 4 (each between 0 and 1)'});
    const inputsEditor = element('textarea',{'aria-label':'Three client feature vectors',rows:'12'},inputsLabel);
    inputsEditor.value = pretty(config.inputs);
    const codeGrid = element('div',{class:'code-grid'});
    const studentPanel = element('section',{class:'panel'},codeGrid);
    element('div',{class:'badge',text:'Your tasks — Client and Server code'},studentPanel);
    const studentTitle = element('h3',{},studentPanel);
    const heEditor = element('textarea',{'aria-label':'Client encryption, server prediction, and client decryption code',rows:'20',spellcheck:'false'},studentPanel);
    heEditor.value=config.he_source;
    const flEditor = element('textarea',{'aria-label':'FedSGD client gradient, aggregation, and server rounds code',rows:'20',spellcheck:'false'},studentPanel);
    flEditor.value=config.fl_source;
    element('p',{class:'muted',text:'Editing clears this part’s results. Execute runs ordinary local Python; np is available in the aggregation task.'},studentPanel);
    codeGrid.style.gridTemplateColumns='1fr';
    const settings=element('div',{class:'toolbar'});
    settings.hidden=part!=='fl';
    const roundsLabel=element('label',{text:'Training rounds '},settings);
    const roundsInput=element('input',{type:'number',min:'1',max:'200',step:'1','aria-label':'Training rounds'},roundsLabel);
    roundsInput.value=String(shared.num_rounds);
    const rateLabel=element('label',{text:'Learning rate '},settings);
    const rateInput=element('input',{type:'number',min:'0.0001',step:'0.01','aria-label':'Learning rate'},rateLabel);
    rateInput.value=String(shared.learning_rate);
    const toolbar = element('div',{class:'toolbar'});
    const execute = element('button',{class:'primary',text:'Execute workflow'},toolbar);
    const twenty = element('button',{text:'Train model'},toolbar);
    const reset = element('button',{text:'Reset all results'},toolbar);
    const busyText = element('span',{class:'muted'},toolbar);
    const playback = element('div',{class:'toolbar'});
    const pause = element('button',{text:'Pause'},playback);
    const next = element('button',{text:'Next step'},playback);
    const replay = element('button',{text:'Replay'},playback);
    const speedLabel=element('label',{text:'Speed '},playback);
    const speed=element('select',{'aria-label':'Animation speed'},speedLabel);
    for (const value of [.5,1,2,4,10]) element('option',{value:String(value),text:`${value}×`},speed);
    speed.value='1';
    const progress=element('span',{class:'muted'},playback);
    const status=element('div',{class:'status',role:'status','aria-live':'polite'});
    const network=element('div',{class:'network'});
    const map=document.createElementNS(NS,'svg'); map.classList.add('network-map'); map.setAttribute('viewBox','0 0 900 210'); map.setAttribute('role','img'); map.setAttribute('aria-label','Animated messages between clients and server'); network.append(map);
    const transitCaption=element('p',{class:'muted'},network);
    const nodeGrid=element('div',{class:'node-grid'});
    const below=element('div',{class:'below'});
    const inspector=element('section',{class:'panel'},below);
    element('h3',{text:'Message inspector · last transmission'},inspector);
    element('p',{class:'muted',text:'Display numbers use six significant figures. Exported JSON preserves full precision.'},inspector);
    const messageText=element('pre',{},inspector);
    const evaluation=element('section',{class:'panel'},below);
    element('h3',{text:'Instructor evaluation · separate held-out data'},evaluation);
    const chart=document.createElementNS(NS,'svg'); chart.setAttribute('viewBox','0 0 430 200'); chart.classList.add('chart'); evaluation.append(chart);
    const evalText=element('pre',{},evaluation);
    const checks=element('div',{class:'checks'});
    const results=element('div',{class:'table-scroll'});
    const submission=element('section',{class:'panel',style:'margin-top:24px'});
    submission.hidden=part!=='he';
    element('h3',{text:'Submission'},submission);
    const exportBar=element('div',{class:'toolbar'},submission);
    const exportButton=element('button',{text:'Generate submission card',class:'primary'},exportBar);
    const exportStatus=element('p',{class:'muted'},submission);
    const cardPreview=element('img',{alt:'Submission card with encrypted prediction agreement, learning curve, and workflow checks',style:'display:block;width:100%;max-width:1000px;height:auto;margin:16px auto'},submission);
    cardPreview.hidden=true;
    const downloads=element('div',{},submission);
    element('p',{class:'muted',text:'Complete the three FedSGD functions and three HE functions, compare encrypted predictions, and finish your configured FL rounds. Submit the PNG card and JSON evidence on Canvas.'},submission);

    const active=part;
    let data={}, busy=false, completed=0, running=false, elapsed=0, lastTime=0;
    let states={}, lastMessage=null, shownMetrics=[], disposed=false, frame;
    let nodes={}, points={}, packet, route, wireLabel;
    const urls=[];
    const current=()=>data[active];
    const labelFor=key=>key==='server'&&active==='fl'?'Aggregation server':names[key]||key;
    function svg(tag,attrs,parent=map) { const n=document.createElementNS(NS,tag); for(const [k,v] of Object.entries(attrs)) n.setAttribute(k,v); parent.append(n); return n; }
    function drawNetwork() {
      map.replaceChildren(); nodeGrid.replaceChildren(); nodes={};
      const markerId='arrival-'+Math.random().toString(36).slice(2);
      const defs=svg('defs',{});
      const marker=svg('marker',{id:markerId,viewBox:'0 0 10 10',refX:10,refY:5,markerWidth:7,markerHeight:7,orient:'auto'},defs);
      svg('path',{d:'M 0 0 L 10 5 L 0 10 Z',fill:'#087d79'},marker);
      nodeGrid.className=`node-grid${active==='fl'?' fl':''}`;
      points=active==='fl'?{c1:[110,45],c2:[450,45],c3:[790,45],server:[450,165]}:{client:[150,100],server:[750,100]};
      for (const key of Object.keys(points)) {
        const [x,y]=points[key];
        if (key!=='server') svg('line',{x1:x,y1:y,x2:points.server[0],y2:points.server[1],stroke:'#b6c9d1','stroke-width':2});
      }
      for (const [key,[x,y]] of Object.entries(points)) {
        svg('rect',{x:x-92,y:y-22,width:184,height:44,rx:12,fill:key==='server'?'#173047':'#087d79'});
        const text=svg('text',{x,y:y+7,'text-anchor':'middle',fill:'white','font-size':22}); text.textContent=key==='server'?'Server':key==='client'?'Client':`Client ${key.slice(1)}`;
        const panel=element('section',{class:'panel node'},nodeGrid);
        element('h3',{text:labelFor(key)},panel); nodes[key]={panel,gradient:element('p',{class:'client-gradient'},panel),records:element('div',{class:'table-scroll'},panel),text:element('pre',{},panel)};
      }
      route=svg('line',{stroke:'#087d79','stroke-width':3,opacity:0,'marker-end':`url(#${markerId})`});
      packet=svg('circle',{r:9,fill:'#e99335',stroke:'#173047','stroke-width':2,opacity:0});
      wireLabel=svg('text',{x:450,y:205,'text-anchor':'middle','font-size':13,fill:'#173047'});
    }
    function initializePart() {
      const fl=part==='fl';
      heading.textContent=fl?'Part A · Train a shared salary model':part==='linked'?'Use the federated model for encrypted inference':'Part B · Predict without revealing the input';
      intro.textContent=fl?'PayWise wants to build a salary model using employee records held by three companies. The companies want a shared model, but their employees have not agreed to send individual records to PayWise. Each company runs a client that trains locally; PayWise runs the aggregation server.':'After training, PayWise deploys its salary prediction model. An employee wants to estimate their salary from experience and education, but does not want PayWise to learn those personal details. The employee’s client encrypts the features, and PayWise evaluates its model on the encrypted input.';
      if(part==='linked') intro.textContent='The actual model produced at the final configured round is automatically used for encrypted inference on all three inputs from Part B. Replay its client/server workflow here.';
      heEditor.hidden=fl; flEditor.hidden=!fl;
      inputsLabel.hidden=part!=='he'; execute.hidden=part!=='he'; twenty.hidden=!fl;
      studentPanel.hidden=part==='linked';
      studentTitle.textContent=fl?'Client gradient · weighted aggregation · server rounds':'Client encryption · server prediction · client decryption';
      drawNetwork(); loadTrace(false); updateButtons();
    }
    function loadTrace(autoplay) {
      completed=0; elapsed=0; states={}; lastMessage=null; shownMetrics=[];
      const result=current(); running=Boolean(autoplay&&result?.events?.length);
      if (matchMedia('(prefers-reduced-motion: reduce)').matches) running=false;
      drawState();
    }
    function roundLimit() { return data.fl?.num_rounds || shared.num_rounds; }
    function updateButtons() {
      busy=shared.busy;
      for(const b of [execute,twenty,reset,exportButton]) b.disabled=busy;
      const round=data.fl?.history?.at(-1)?.round||0;
      if(active==='fl') { twenty.disabled=busy||round>=roundLimit(); }
      exportButton.disabled=busy||!data.he||!data.linked;
      exportStatus.textContent=!data.fl?'Complete the client and server tasks and run the workflows to export.':!data.he?'Execute Part B with your finished encryption function.':!data.linked?'Finish all configured Part A rounds, then execute Part B to connect training and inference.':'Ready to export. Failed checks, if any, will be included honestly.';
      for(const editor of [heEditor,flEditor,inputsEditor,roundsInput,rateInput]) editor.disabled=busy;
      const has=Boolean(current()?.events?.length);
      pause.disabled=!has||completed>=current().events.length; next.disabled=!has||completed>=current().events.length; replay.disabled=!has;
    }
    function invalidate(changed) {
      const affected=changed==='all'?['fl','he','linked']:[changed,'linked'];
      for(const key of affected) delete data[key];
      downloads.replaceChildren();
      cardPreview.hidden=true; cardPreview.removeAttribute('src');
      if(affected.includes(part)) {running=false;loadTrace(false);busyText.textContent='Code or inputs changed. Execute to create a fresh run.';}
      if(part==='linked') root.hidden=true;
      updateButtons();
    }
    shared.invalidators.push(invalidate);
    shared.updaters.push(updateButtons);
    for(const editor of [heEditor,flEditor,inputsEditor]) editor.oninput=()=>{
      if(part==='he') {shared.he_source=heEditor.value;shared.inputs=inputsEditor.value;}
      if(part==='fl') shared.fl_source=flEditor.value;
      shared.revisions[part]++;
      for(const clear of shared.invalidators) clear(part);
    };
    for(const input of [roundsInput,rateInput]) input.oninput=()=>{
      shared.num_rounds=Number(roundsInput.value);
      shared.learning_rate=Number(rateInput.value);
      shared.revisions.fl++;
      for(const clear of shared.invalidators) clear('fl');
    };
    function request(action) {
      if(!Number.isInteger(shared.num_rounds)||shared.num_rounds<1||shared.num_rounds>200||!Number.isFinite(shared.learning_rate)||shared.learning_rate<=0){busyText.textContent='Choose 1–200 rounds and a positive learning rate.';return;}
      let inputs;
      try { inputs=JSON.parse(shared.inputs); if(!Array.isArray(inputs)||inputs.length!==3||!inputs.every(row=>Array.isArray(row)&&row.length===2&&row.every(x=>typeof x==='number'&&Number.isFinite(x)&&x>=0&&x<=1))) throw Error('Enter exactly three [experience, education] pairs, with values from 0 to 1.'); }
      catch(e) { busyText.textContent=e.message; return; }
      shared.busy=true; running=false; for(const update of shared.updaters) update();
      busyText.textContent='Computing in Python… playback starts when the actual trace arrives.';
      model.set('request',{num_rounds:shared.num_rounds,learning_rate:shared.learning_rate,id:++shared.id,revisions:{...shared.revisions},action,he_source:shared.he_source,fl_source:shared.fl_source,inputs}); model.save_changes();
    }
    execute.onclick=()=>request('he'); twenty.onclick=()=>request('twenty');
    reset.onclick=()=>{for(const clear of shared.invalidators) clear('all');request('reset');};
    exportButton.onclick=()=>request('card');
    function download(label,blob,filename) {
      const url=URL.createObjectURL(blob); urls.push(url);
      element('a',{text:label,href:url,download:filename,style:'display:inline-block;margin:8px 20px 8px 0'},downloads);
    }
    function receive() {
      const response=model.get('response'); if(!response.id) return;
      shared.busy=false; data={...(response.results||{})}; busyText.textContent=response.error||'Computation finished. Replay never runs Python again.';
      downloads.replaceChildren();
      cardPreview.hidden=true; cardPreview.removeAttribute('src');
      if(response.card && part==='he') {
        cardPreview.src='data:image/png;base64,'+response.card;
        cardPreview.hidden=false;
        const bytes=Uint8Array.from(atob(response.card),c=>c.charCodeAt(0));
        download('Download submission card (PNG)',new Blob([bytes],{type:'image/png'}),'lab-6-submission.png');
        download('Download evidence (JSON)',new Blob([JSON.stringify({results:data,he_source:shared.he_source,fl_source:shared.fl_source,inputs:JSON.parse(shared.inputs)},null,2)],{type:'application/json'}),'lab-6-evidence.json');
      }
      const affected=response.action==='he'?['he','linked']:['one','twenty'].includes(response.action)?['fl','linked']:[];
      if(response.action==='reset') loadTrace(false);
      else if(affected.includes(part)) loadTrace(part!=='linked');
      // Each part owns its playback; completing B does not restart A.
      if(part==='linked') root.hidden=!data.linked;

      updateButtons();
    }
    model.on('change:response',receive);
    function finishStep() {
      const event=current()?.events?.[completed]; if(!event) return;
      // Put this step's results first so students need not scroll past old inputs.
      for(const [key,update] of Object.entries(event.updates||{})) states[key]={...update,...Object.fromEntries(Object.entries(states[key]||{}).filter(([name])=>!(name in update)))};
      if(event.message) lastMessage=event.message;
      if(event.metric) shownMetrics.push(event.metric);
      completed++; elapsed=0;
      if(completed>=current().events.length) running=false;
      drawState();
    }
    next.onclick=()=>{running=false;finishStep();};
    pause.onclick=()=>{if(completed>=current().events.length) return; running=!running; drawState();};
    replay.onclick=()=>loadTrace(true);
    function drawChart() {
      chart.replaceChildren(); chart.hidden=active!=='fl'; if(active!=='fl') return;
      const history=[...(current()?.history||[]).filter(r=>r.round===0),...shownMetrics];
      const refs=current()?.baselines||{};
      const values=[...history.map(r=>r.mse),...Object.values(refs)].filter(x=>Number.isFinite(x)&&x>0);
      if(!values.length) return;
      const lo=Math.log10(Math.min(...values)*.8), hi=Math.log10(Math.max(...values)*1.2);
      const y=v=>160-(Math.log10(Math.max(v,1e-12))-lo)/(hi-lo)*135;
      svg('line',{x1:40,y1:160,x2:410,y2:160,stroke:'#738c9a'},chart);
      let index=0;
      for(const [label,value] of Object.entries(refs)) {
        const color=['#6b7787','#aa7340','#4d7894','#9d6696'][index++];
        svg('line',{x1:40,y1:y(value),x2:410,y2:y(value),stroke:color,'stroke-dasharray':'4 4'},chart);
        const text=svg('text',{x:45,y:12+index*12,'font-size':10,fill:color},chart); text.textContent=`${label}: ${value.toFixed(2)}`;
      }
      svg('polyline',{points:history.map(r=>`${40+r.round/roundLimit()*370},${y(r.mse)}`).join(' '),fill:'none',stroke:'#087d79','stroke-width':3},chart);
      for(const r of history) svg('circle',{cx:40+r.round/roundLimit()*370,cy:y(r.mse),r:3,fill:'#087d79'},chart);
      const label=svg('text',{x:40,y:190,'font-size':11,fill:'#173047'},chart);label.textContent=`0                 Communication round                 ${roundLimit()} · log MSE ($k²)`;
    }
    function drawResults() {
      results.replaceChildren(); checks.replaceChildren();
      const result=current(); if(!result||completed<result.events.length) return;
      for(const [label,ok] of Object.entries(result.checks||{})) element('span',{class:`check${ok?'':' fail'}`,text:`${ok?'PASS':'FAIL'} · ${label}`},checks);
      if(result.error) element('p',{class:'error',text:result.error},results);
      element('p',{class:'muted',text:`Measured Python workflow: ${result.elapsed_ms.toFixed(1)} ms. Animation speed is illustrative; first-run key setup is included for HE.`},results);
      const table=element('table',{},results); const head=element('tr',{},table);
      if(active!=='fl') {
        for(const label of ['Input','Plaintext $k','Decrypted $k','Absolute error $k','Encrypt ms','Server ms','Decrypt ms','Plaintext µs']) element('th',{text:label},head);
        for(const r of result.rows||[]) {const tr=element('tr',{},table); for(const value of [r.features.join(', '),r.reference.toFixed(5),r.prediction.toFixed(5),r.error.toExponential(2),r.encrypt_ms.toFixed(2),r.server_ms.toFixed(2),r.decrypt_ms.toFixed(2),r.plaintext_us.toFixed(2)]) element('td',{text:value},tr);}
      } else {
        for(const label of ['Round','Held-out MSE ($k²)']) element('th',{text:label},head);
        for(const r of result.history.filter(r=>[0,1,5,20].includes(r.round)||r.round===result.history.at(-1).round)) {const tr=element('tr',{},table); element('td',{text:r.round},tr);element('td',{text:r.mse.toFixed(5)},tr);}
        element('p',{text:'Baselines use the configured learning rate and number of rounds. FedSGD uses one full-batch gradient per client per round. Local-only models need not be worse.',class:'muted'},results);

      }
    }
    function drawState() {
      const result=current(), e=result?.events?.[completed];
      for(const [key,node] of Object.entries(nodes)) {
        const state=states[key]||{status:'Waiting for an operation or message arrival'};
        node.gradient.textContent=key.startsWith('c')&&active==='fl'?'Client gradient [w₁, w₂, b]: '+(state.gradient?state.gradient.map(v=>Number(v).toPrecision(5)).join(', '):'not computed yet'):'';
        node.records.replaceChildren();
        if(state.record_preview){
          element('p',{text:`${state.samples} distinct employee records · showing first ${state.record_preview.length}`},node.records);
          const table=element('table',{},node.records);const head=element('tr',{},table);
          for(const label of ['Experience (years)','Education (years)','Salary ($k)'])element('th',{text:label},head);
          for(const row of state.record_preview){const tr=element('tr',{},table);for(const value of [row.experience_years,row.education_years,row.salary_$k])element('td',{text:String(value)},tr);}
        }
        const {record_preview,gradient,...other}=state;
        node.text.textContent=pretty(other);node.panel.classList.toggle('active',e?.actor===key);
      }
      studentPanel.classList.toggle('active-code',['he_server','fl_server','train','encrypt','decrypt'].includes(e?.code));
      progress.textContent=`${completed} / ${result?.events?.length||0} steps complete`;
      status.textContent=e?`${labelFor(e.actor)} · ${e.text}`:result?result.error?'Workflow stopped: '+result.error:'Playback complete. Inspect results below.':'Complete the server function, then execute to follow the workflow.';
      status.classList.toggle('error',Boolean(result?.error&&completed===result.events.length));
      messageText.textContent=lastMessage?pretty({...lastMessage,sender:labelFor(lastMessage.sender),recipient:labelFor(lastMessage.recipient)}):'No message has arrived. Local records are not transmissions.';
      evalText.textContent=pretty(states.instructor||{status:'Waiting for instructor evaluation'});
      pause.textContent=running?'Pause':'Resume';
      drawPacket(e);
      drawChart(); drawResults(); updateButtons();
    }
    function drawPacket(e) {
      packet.setAttribute('opacity',0);route.setAttribute('opacity',0);wireLabel.textContent='';
      transitCaption.textContent='Local computations stay inside their labeled role. Animated arrows represent message transmissions.';
      if(!e?.message) return;
      const source=points[e.message.sender], destination=points[e.message.recipient];
      if(!source||!destination) return;
      const dx=destination[0]-source[0], dy=destination[1]-source[1];
      const inset=Math.min(dx?92/Math.abs(dx):Infinity,dy?22/Math.abs(dy):Infinity);
      const a=[source[0]+dx*inset,source[1]+dy*inset], b=[destination[0]-dx*inset,destination[1]-dy*inset];
      const fraction=Math.min(1,elapsed/1800);
      route.setAttribute('x1',a[0]);route.setAttribute('y1',a[1]);route.setAttribute('x2',b[0]);route.setAttribute('y2',b[1]);route.setAttribute('opacity',1);
      packet.setAttribute('cx',a[0]+(b[0]-a[0])*fraction);packet.setAttribute('cy',a[1]+(b[1]-a[1])*fraction);packet.setAttribute('opacity',1);
      wireLabel.textContent=e.message.kind;
      transitCaption.textContent=`In transit: ${labelFor(e.message.sender)} → ${labelFor(e.message.recipient)} · ${e.message.kind}. Recipient state updates on arrival.`;
    }
    function tick(now) {
      if(disposed) return;
      const dt=lastTime?Math.min(now-lastTime,100):0; lastTime=now;
      const e=current()?.events?.[completed];
      if(running&&e) {
        elapsed+=dt*Number(speed.value);
        const duration=e.message?1800:1250, fraction=Math.min(1,elapsed/duration);
        drawPacket(e);
        if(fraction>=1) finishStep();
      }
      frame=requestAnimationFrame(tick);
    }
    initializePart(); if(part==='linked') root.hidden=true; frame=requestAnimationFrame(tick);
    return ()=>{disposed=true;cancelAnimationFrame(frame);model.off('change:response',receive);for(const url of urls) URL.revokeObjectURL(url);root.remove();};
}

export default {
  render({model, el}) {
    const config=model.get('config');
    const shared={he_source:config.he_source,fl_source:config.fl_source,num_rounds:1,learning_rate:0.1,inputs:pretty(config.inputs),revisions:{he:0,fl:0},id:0,busy:false,invalidators:[],updaters:[]};
    const partA=document.createElement('section'); partA.setAttribute('aria-label','Part A · Federated learning');el.append(partA);
    const partB=document.createElement('section'); partB.setAttribute('aria-label','Part B · Homomorphic encryption');el.append(partB);
    partB.style.marginTop='32px';
    const cleanupA=renderPart({model,el:partA},'fl',shared);
    const cleanupB=renderPart({model,el:partB},'he',shared);
    return ()=>{cleanupA();cleanupB();partA.remove();partB.remove();};
  }
};
