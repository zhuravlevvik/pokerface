(() => {
  const byId = (id) => document.getElementById(id);
  const status = byId('status');
  let socket; let queue = []; let playing = false;
  const card = (value) => `<span class="card${value ? '' : ' hidden'}">${value || '??'}</span>`;
  const pct = (value) => `${(100 * value).toFixed(1)}%`;

  function renderBars(target, probabilities) {
    target.replaceChildren(...Object.entries(probabilities || {}).map(([name, value]) => {
      const row = document.createElement('div'); row.className = 'bar';
      row.innerHTML = `<span>${name}</span><div><i style="width:${Math.max(0, Math.min(100, value * 100))}%"></i></div><b>${pct(value)}</b>`;
      return row;
    }));
  }
  function renderTable(table) {
    byId('pot').textContent = table.pot;
    byId('street').textContent = table.street;
    byId('board').innerHTML = table.board.map(card).join('');
    const seats = byId('seats'); seats.replaceChildren();
    table.players.forEach((player) => {
      const el = byId('seat-template').content.firstElementChild.cloneNode(true);
      el.classList.toggle('actor', table.current_actor === player.seat);
      el.querySelector('.position').textContent = `${player.position} · #${player.seat}${player.is_hero ? ' (герой)' : ''}`;
      el.querySelector('.chips').textContent = `${player.stack} фишек${player.committed_street ? ` · в ставке ${player.committed_street}` : ''}`;
      el.querySelector('.cards').innerHTML = (player.hole_cards || [null, null]).map(card).join('');
      el.querySelector('.state').textContent = player.folded ? 'fold' : (player.all_in ? 'all-in' : (player.payout ? `payout +${player.payout}` : 'в игре'));
      seats.append(el);
    });
    const history = byId('history'); history.replaceChildren(...table.action_history.map((record) => {
      const li = document.createElement('li'); li.textContent = `${record.street}: ${table.players.find(p => p.seat === record.seat)?.position || '#' + record.seat} — ${record.action}${record.amount ? ` (${record.amount})` : ''}`; return li;
    }));
    renderGraph(table.equity_points || []);
  }
  function renderGraph(points) {
    const svg = byId('equity-graph'); svg.replaceChildren();
    const ns = 'http://www.w3.org/2000/svg';
    const make = (name, attrs) => { const el = document.createElementNS(ns, name); Object.entries(attrs).forEach(([k,v]) => el.setAttribute(k,v)); return el; };
    svg.append(make('line', {x1: 18,y1:112,x2:310,y2:112,class:'axis'})); svg.append(make('line',{x1:18,y1:12,x2:18,y2:112,class:'axis'}));
    if (points.length) { const coords = points.map((point, index) => [18 + index * (292 / Math.max(1, points.length - 1)), 112 - point.equity * 100]); svg.append(make('polyline', {points: coords.map(p => p.join(',')).join(' '), class:'line'})); coords.forEach(([x,y]) => svg.append(make('circle',{cx:x,cy:y,r:4,class:'dot'}))); }
    byId('equity-labels').textContent = points.map(p => `${p.street}: ${pct(p.equity)}`).join(' → ') || 'Нет решений героя';
  }
  function renderEvent(event) {
    if (event.table) renderTable(event.table);
    if (event.analysis) { const a = event.analysis; byId('selected-action').textContent = `Выбрано: ${a.action}`; byId('value').textContent = `${a.value_bb.toFixed(2)} BB`; byId('equity').textContent = `${pct(a.equity.total)} (win ${pct(a.equity.win)}, tie ${pct(a.equity.tie)})`; renderBars(byId('action-probs'), a.action_probabilities); renderBars(byId('size-probs'), a.bet_size_probabilities); }
    if (event.type === 'hand_complete') status.textContent = 'Раздача завершена.';
  }
  function consume() { if (!queue.length) { playing = false; return; } playing = true; renderEvent(queue.shift()); window.setTimeout(consume, 420); }
  function enqueue(events) { queue.push(...events); if (!playing) consume(); }
  function send(command) { if (!socket || socket.readyState !== WebSocket.OPEN) { status.textContent = 'Нет подключения к серверу.'; return; } queue = []; socket.send(JSON.stringify(command)); }
  function connect() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws'; socket = new WebSocket(`${scheme}://${location.host}/ws/table`);
    socket.onopen = () => { status.textContent = 'Подключено. Выберите новую раздачу.'; };
    socket.onmessage = ({data}) => { const message = JSON.parse(data); if (message.type === 'error') status.textContent = `Ошибка: ${message.detail}`; else enqueue([message]); };
    socket.onclose = () => { status.textContent = 'Соединение закрыто; повторное подключение…'; window.setTimeout(connect, 1000); };
  }
  byId('start').onclick = () => { const raw = byId('seed').value; send({type:'start_hand', mode:byId('mode').value, hero_seat:Number(byId('hero').value), seed:raw === '' ? null : Number(raw)}); };
  byId('replay-file').onchange = async (event) => { const file = event.target.files[0]; if (!file) return; try { send({type:'replay', mode:byId('mode').value, hero_seat:Number(byId('hero').value), replay:JSON.parse(await file.text())}); } catch (_) { status.textContent = 'Не удалось прочитать JSON replay.'; } };
  connect();
})();
