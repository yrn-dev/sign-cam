/* Шапка портала и переход между режимами.
   Подключается синхронно сразу после <body>: шапка должна быть на месте
   до первой отрисовки, иначе страница дёрнется. */
(function(){
  const MODES = [
    {href:'/',            name:'Эфир',     icon:'<circle cx="12" cy="12" r="3.2"/><path d="M4.9 4.9a10 10 0 0 0 0 14.2M19.1 4.9a10 10 0 0 1 0 14.2M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4"/>'},
    {href:'/upload.html', name:'Файл',     icon:'<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M12 17v-6M9.5 13.5 12 11l2.5 2.5"/>'},
    {href:'/learn.html',  name:'Оқу', icon:'<path d="M3 9.5 12 5l9 4.5-9 4.5z"/><path d="M7 11.6V16c0 1.4 2.2 3 5 3s5-1.6 5-3v-4.4M21 9.5V15"/>'},
  ];
  const path = location.pathname.replace(/\/index\.html$/,'/');
  const cur = Math.max(0, MODES.findIndex(m => m.href === path));
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const root = document.documentElement;

  let from = null, vx = null, vy = null;
  try{
    from = sessionStorage.getItem('pt.from');
    vx = sessionStorage.getItem('pt.vx'); vy = sessionStorage.getItem('pt.vy');
    sessionStorage.removeItem('pt.from');
  }catch(e){}
  if(!reduce) root.classList.add('pt-enter');

  const svg = d => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;
  const bar = document.createElement('div');
  bar.className = 'pt-bar';
  bar.innerHTML = `
    <a class="pt-brand" href="/" data-i="0">
      <span class="pt-mark">${svg('<path d="M8 13V6.5a1.5 1.5 0 0 1 3 0V12M11 11V4.5a1.5 1.5 0 0 1 3 0V11M14 10.5V6a1.5 1.5 0 0 1 3 0v7a7 7 0 0 1-7 7h-.3a6 6 0 0 1-4.6-2.2L2.8 15a1.6 1.6 0 0 1 2.4-2.1L8 15" stroke="#409cff"/>')}</span>
      <span class="pt-name">SignCam<small>ымдау тілінен аудармашы</small></span>
    </a>
    <nav class="pt-modes" aria-label="Бөлімдер">
      <span class="pt-glide"></span>
      ${MODES.map((m,i)=>`<a class="pt-mode${i===cur?' on':''}" href="${m.href}" data-i="${i}"
          ${i===cur?'aria-current="page"':''}>${svg(m.icon)}<span>${m.name}</span><span class="pt-key">${i+1}</span></a>`).join('')}
    </nav>
    <div class="pt-right" id="ptRight"></div>`;
  const atmos = document.createElement('div');
  atmos.className = 'pt-atmos';
  atmos.innerHTML = '<div class="pt-aur a"></div><div class="pt-aur b"></div><div class="pt-aur c"></div>'
    + '<div class="pt-grid"></div><div class="pt-glow"></div>';
  document.currentScript.after(atmos, bar);

  // слабое устройство: камера и кодирование кадров важнее красоты фона
  if((navigator.deviceMemory && navigator.deviceMemory <= 2) ||
     (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2)) root.classList.add('pt-lite');
  document.addEventListener('visibilitychange', ()=> root.classList.toggle('pt-idle', document.hidden));

  /* свет догоняет курсор. Цикл кадров крутится, только пока свет движется,
     и сам останавливается, когда догнал */
  const glow = atmos.querySelector('.pt-glow');
  if(!reduce && matchMedia('(pointer: fine)').matches){
    let tx = innerWidth/2, ty = innerHeight/3, gx = tx, gy = ty, raf = 0;
    const step = ()=>{
      gx += (tx-gx)*.08; gy += (ty-gy)*.08;
      glow.style.transform = `translate3d(${gx}px,${gy}px,0)`;
      raf = Math.abs(tx-gx) + Math.abs(ty-gy) > .5 ? requestAnimationFrame(step) : 0;
    };
    addEventListener('pointermove', e=>{
      tx = e.clientX; ty = e.clientY;
      glow.classList.add('on');
      if(!raf) raf = requestAnimationFrame(step);
    }, {passive:true});
    document.addEventListener('pointerleave', ()=> glow.classList.remove('on'));
  }

  /* API для страниц: волна по фону в точке (x, y) и общая «энергия» фона 0…1 */
  let lastWave = 0;
  window.ptPulse = (x, y, strength = 1)=>{
    if(reduce || document.hidden) return;
    const t = performance.now();
    if(t - lastWave < 350) return;          // не больше ~3 волн в секунду
    lastWave = t;
    ['','soft'].forEach(k=>{
      const w = document.createElement('div');
      w.className = 'pt-wave ' + k;
      w.style.cssText = `--wx:${x}px;--wy:${y}px;--ws:${3 + 4*strength}`;
      atmos.appendChild(w);
      w.addEventListener('animationend', ()=> w.remove());
    });
  };
  window.ptEnergy = v => atmos.style.setProperty('--pt-energy', Math.max(0, Math.min(1, v)).toFixed(2));

  const glide = bar.querySelector('.pt-glide');
  const links = [...bar.querySelectorAll('.pt-mode')];
  function place(i){
    const a = links[i];
    glide.style.setProperty('--x', a.offsetLeft - 4 + 'px');
    glide.style.setProperty('--w', a.offsetWidth + 'px');
    glide.style.left = '4px';
  }

  // бегунок стартует там, откуда пришли, и доезжает до текущего режима
  const start = from !== null && +from !== cur && !reduce ? +from : cur;
  glide.classList.add('snap');
  place(start);
  requestAnimationFrame(()=>{
    glide.classList.remove('snap');
    glide.classList.add('ready');
    requestAnimationFrame(()=> place(cur));
  });
  const settle = ()=>{ glide.classList.add('snap'); place(cur); glide.offsetWidth; glide.classList.remove('snap'); };
  addEventListener('resize', settle);
  if(document.fonts) document.fonts.ready.then(settle);

  // занавес: при уходе раскрывается из кнопки, на новой странице растворяется
  if(from !== null && !reduce){
    const v = document.createElement('div');
    v.className = 'pt-veil full';
    if(vx) v.style.setProperty('--vx', vx);
    if(vy) v.style.setProperty('--vy', vy);
    document.body.appendChild(v);
    requestAnimationFrame(()=> requestAnimationFrame(()=>{
      v.classList.add('fade');
      setTimeout(()=> v.remove(), 800);
    }));
  }
  const reveal = ()=> requestAnimationFrame(()=> requestAnimationFrame(()=> root.classList.remove('pt-enter')));
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', reveal); else reveal();

  let leaving = false;
  function go(i, x, y){
    if(i === cur || leaving) return;
    const href = MODES[i].href;
    if(reduce){ location.href = href; return; }
    leaving = true;
    links.forEach((a,k)=> a.classList.toggle('on', k===i));
    place(i);
    const px = (x / innerWidth * 100).toFixed(1) + '%', py = (y / innerHeight * 100).toFixed(1) + '%';
    try{
      sessionStorage.setItem('pt.from', cur);
      sessionStorage.setItem('pt.vx', px); sessionStorage.setItem('pt.vy', py);
    }catch(e){}
    root.classList.add('pt-leave');
    const v = document.createElement('div');
    v.className = 'pt-veil';
    v.style.setProperty('--vx', px); v.style.setProperty('--vy', py);
    document.body.appendChild(v);
    v.offsetWidth;
    setTimeout(()=> v.classList.add('grow'), 120);   // сначала бегунок, потом занавес
    setTimeout(()=> location.href = href, 640);
  }

  bar.addEventListener('click', e=>{
    const a = e.target.closest('[data-i]');
    if(!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button) return;
    e.preventDefault();
    const r = a.getBoundingClientRect();
    go(+a.dataset.i, e.clientX || r.left + r.width/2, e.clientY || r.top + r.height/2);
  });

  addEventListener('keydown', e=>{
    if(e.altKey || e.ctrlKey || e.metaKey || /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    const i = +e.key - 1;
    if(i >= 0 && i < MODES.length){
      const r = links[i].getBoundingClientRect();
      go(i, r.left + r.width/2, r.top + r.height/2);
    }
  });

  // режимы лёгкие — подгружаем заранее, чтобы переход был мгновенным
  MODES.forEach((m,i)=>{
    if(i === cur) return;
    const l = document.createElement('link');
    l.rel = 'prefetch'; l.href = m.href;
    document.head.appendChild(l);
  });

  // возврат кнопкой «назад» из кэша: убрать занавес и вернуть бегунок
  addEventListener('pageshow', e=>{
    if(!e.persisted) return;
    leaving = false;
    root.classList.remove('pt-leave','pt-enter');
    document.querySelectorAll('.pt-veil').forEach(v=> v.remove());
    links.forEach((a,k)=> a.classList.toggle('on', k===cur));
    settle();
  });
})();
