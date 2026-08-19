(() => {
  const byId = (id) => document.getElementById(id);
  const status = byId('status');
  let socket; let queue = []; let playing = false; let paused = false; let latestReplay = null;
  let policyCatalog = []; let defaultSeatPolicies = {};
  const card = (value) => `<span class="card${value ? '' : ' hidden'}">${value || '??'}</span>`;
  const pct = (value) => `${(100 * value).toFixed(1)}%`;
  const playerCount = () => Number(byId('player-count').value);
  const playbackDelay = () => Math.max(40, 420 / Number(byId('speed').value));
  function metricLabel(metric, protocol) {
    if (metric === 'expected_showdown_share' && protocol === 'active_hands_expected_showdown_share_v1') return 'Ожидаемая доля при showdown среди активных рук';
    if (metric === 'heuristic_hand_strength' && protocol === 'heuristic_hand_strength_v1') return 'Эвристическая сила руки (не equity модели)';
    if (metric === 'heads_up_showdown_share' && protocol === 'legacy_win_plus_half_tie_heads_up_v1') return 'HU equity (старый replay)';
    return metric || 'Скалярная оценка';
  }

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
    const seats = byId('seats'); seats.replaceChildren(); seats.className = `player-count-${table.players.length}`;
    table.players.forEach((player) => {
      const el = byId('seat-template').content.firstElementChild.cloneNode(true);
      el.classList.toggle('actor', table.current_actor === player.seat);
      const policy = player.policy ? ` · ${player.policy.name}` : '';
      el.querySelector('.position').textContent = `${player.position} · #${player.seat}${player.is_hero ? ' (герой)' : ''}${policy}`;
      el.querySelector('.chips').textContent = `${player.stack} фишек${player.committed_street ? ` · в ставке ${player.committed_street}` : ''}`;
      el.querySelector('.cards').innerHTML = (player.hole_cards || [null, null]).map(card).join('');
      const pnl = player.pnl ? ` · PnL ${player.pnl >= 0 ? '+' : ''}${player.pnl}` : '';
      el.querySelector('.state').textContent = player.folded ? `fold${pnl}` : (player.all_in ? `all-in${pnl}` : (player.payout ? `payout +${player.payout}${pnl}` : `в игре${pnl}`));
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
    if (points.length) {
      const coords = points.map((point, index) => [18 + index * (292 / Math.max(1, points.length - 1)), 112 - point.value * 100]);
      svg.append(make('polyline', {points: coords.map(p => p.join(',')).join(' '), class:'line'}));
      coords.forEach(([x,y]) => svg.append(make('circle',{cx:x,cy:y,r:4,class:'dot'})));
    }
    const first = points[0];
    byId('scalar-graph-title').textContent = first ? metricLabel(first.metric, first.protocol) : 'Скалярная оценка героя';
    svg.setAttribute('aria-label', first ? metricLabel(first.metric, first.protocol) : 'График скалярной оценки');
    byId('equity-labels').textContent = points.map(p => `${p.street}: ${pct(p.value)}`).join(' → ') || 'Нет решений героя';
  }
  function renderSummary(summary) {
    const policyNames = Object.values(summary.policies || {}).map((policy) => policy.name).join(' vs ');
    const pnl = Object.entries(summary.pnl || {}).map(([seat, value]) => `#${seat}: ${value >= 0 ? '+' : ''}${value}`).join(' · ');
    byId('series-summary').textContent = `${summary.hands || 0} раздач · ${policyNames || '—'} · PnL ${pnl || '—'}`;
  }
  function renderEvent(event) {
    if (event.table) renderTable(event.table);
    if (event.analysis) {
      const a = event.analysis; const name = event.policy ? ` (${event.policy.name})` : '';
      byId('selected-action').textContent = `Выбрано${name}: ${a.action}`;
      byId('value').textContent = `${a.value_bb.toFixed(2)} BB`;
      const metric = a.scalar_metric;
      const outcomes = `win ${pct(a.equity.win)}, tie ${pct(a.equity.tie)}, loss ${pct(a.equity.loss)}`;
      byId('equity').textContent = metric ? `${metricLabel(metric.name, metric.protocol)}: ${pct(metric.value)} (${outcomes})` : outcomes;
      renderBars(byId('action-probs'), a.action_probabilities); renderBars(byId('size-probs'), a.bet_size_probabilities);
    }
    if (event.type === 'series_started') renderSummary({hands: event.series.hands, policies: event.series.policies, pnl: {}});
    if (event.type === 'hand_complete') { latestReplay = event.replay; byId('download').disabled = false; status.textContent = event.series_pnl ? `Раздача завершена. Текущий PnL: ${Object.values(event.series_pnl).join(' / ')}` : 'Раздача завершена.'; }
    if (event.type === 'series_complete') { renderSummary(event.summary); status.textContent = 'Серия завершена.'; }
  }
  function setPlaybackControls() {
    const hasEvents = queue.length > 0 || playing;
    byId('pause').disabled = !hasEvents;
    byId('step').disabled = !queue.length;
    byId('next-hand').disabled = !queue.length;
    byId('pause').textContent = paused ? 'Продолжить' : 'Пауза';
  }
  function consume() {
    if (paused || !queue.length) { playing = false; setPlaybackControls(); return; }
    playing = true; renderEvent(queue.shift()); setPlaybackControls();
    window.setTimeout(consume, playbackDelay());
  }
  function enqueue(events) { queue.push(...events); setPlaybackControls(); if (!playing && !paused) consume(); }
  function send(command) {
    if (!socket || socket.readyState !== WebSocket.OPEN) { status.textContent = 'Нет подключения к серверу.'; return; }
    queue = []; latestReplay = null; byId('download').disabled = true; paused = false; setPlaybackControls(); socket.send(JSON.stringify(command));
  }
  function policySelect(seat) {
    const select = document.createElement('select'); select.id = `seat-policy-${seat}`;
    policyCatalog.forEach((policy) => { const option = document.createElement('option'); option.value = policy.id; option.textContent = policy.name; select.append(option); });
    select.value = defaultSeatPolicies[seat] || 'bot:rule'; return select;
  }
  function rebuildSeatControls() {
    const count = playerCount(); const controls = byId('seat-controls'); controls.replaceChildren();
    const hero = byId('hero'); const oldHero = hero.value; hero.replaceChildren();
    for (let seat = 0; seat < count; seat += 1) {
      const heroOption = document.createElement('option'); heroOption.value = seat; heroOption.textContent = `#${seat}`; hero.append(heroOption);
      const label = document.createElement('label'); label.textContent = `Место ${seat}`; label.append(policySelect(seat)); controls.append(label);
    }
    hero.value = Number(oldHero) < count ? oldHero : '0';
  }
  function selectedPolicies() {
    const result = {}; for (let seat = 0; seat < playerCount(); seat += 1) result[seat] = byId(`seat-policy-${seat}`).value; return result;
  }
  async function loadPolicies() {
    try {
      const response = await fetch('/api/policies'); const payload = await response.json();
      policyCatalog = Array.isArray(payload.policies) ? payload.policies : [];
      if (!policyCatalog.length) throw new Error('empty policy catalog');
      defaultSeatPolicies = payload.seat_policies || {};
      const defaults = payload.defaults || {};
      if (defaults.player_count && [2, 3, 5].includes(Number(defaults.player_count))) byId('player-count').value = defaults.player_count;
      if (defaults.hands) byId('hands').value = defaults.hands;
      if (Number.isInteger(defaults.seed_start)) byId('seed').value = defaults.seed_start;
    } catch (_) {
      policyCatalog = [{id:'bot:rule', name:'Rule bot', kind:'bot'}];
      status.textContent = 'Каталог политик недоступен: оставлен Rule bot.';
    }
    rebuildSeatControls();
  }
  function connect() {
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws'; socket = new WebSocket(`${scheme}://${location.host}/ws/table`);
    socket.onopen = () => { status.textContent = 'Подключено. Настройте стол и запустите просмотр.'; };
    socket.onmessage = ({data}) => { const message = JSON.parse(data); if (message.type === 'error') status.textContent = `Ошибка: ${message.detail}`; else enqueue([message]); };
    socket.onclose = () => { status.textContent = 'Соединение закрыто; повторное подключение…'; window.setTimeout(connect, 1000); };
  }
  byId('player-count').onchange = rebuildSeatControls;
  byId('start').onclick = () => {
    const raw = byId('seed').value;
    send({type:'start_hand', mode:byId('mode').value, hero_seat:Number(byId('hero').value), player_count:playerCount(), hands:Number(byId('hands').value), seed_start:raw === '' ? null : Number(raw), seat_policies:selectedPolicies()});
  };
  byId('pause').onclick = () => { paused = !paused; setPlaybackControls(); if (!paused && !playing) consume(); };
  byId('step').onclick = () => { if (queue.length) renderEvent(queue.shift()); setPlaybackControls(); };
  byId('next-hand').onclick = () => { while (queue.length && queue[0].type !== 'hand_started') queue.shift(); if (queue.length) renderEvent(queue.shift()); setPlaybackControls(); };
  byId('speed').onchange = () => { if (!paused && !playing && queue.length) consume(); };
  byId('download').onclick = () => {
    if (!latestReplay) return;
    const link = document.createElement('a'); link.href = URL.createObjectURL(new Blob([JSON.stringify(latestReplay, null, 2)], {type:'application/json'})); link.download = `pokerface-replay-${latestReplay.seed ?? 'random'}.json`; link.click(); URL.revokeObjectURL(link.href);
  };
  byId('replay-file').onchange = async (event) => { const file = event.target.files[0]; if (!file) return; try { send({type:'replay', mode:byId('mode').value, hero_seat:Number(byId('hero').value), replay:JSON.parse(await file.text())}); } catch (_) { status.textContent = 'Не удалось прочитать JSON replay.'; } };
  setPlaybackControls(); loadPolicies(); connect();
})();
