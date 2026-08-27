let MOVIES = {}, RECS = {}, SIMILAR = {}, META = {}, METRICS = [];

async function loadJSON(name) {
  const r = await fetch(`data/${name}.json`);
  return r.json();
}

function stars(avg) {
  const n = Math.round(avg || 0);
  return '<span class="star">' + '★'.repeat(n) + '</span>';
}

function card(movieId, extra) {
  const m = MOVIES[movieId];
  if (!m) return '';
  const poster = m.p
    ? `<img class="poster" src="${m.p}" alt="" loading="lazy">`
    : `<div class="poster ph">🎬</div>`;
  return `<div class="card">
    ${poster}
    <div class="body">
      <div class="title">${m.t}</div>
      <div class="genres">${(m.g || '').replaceAll('|', ' · ')}</div>
      <div class="meta">${stars(m.a)} ${m.a ?? ''} · ${m.n} ratings</div>
      ${extra || ''}
    </div>
  </div>`;
}

function renderRecs() {
  const uid = document.getElementById('userSel').value;
  const model = document.getElementById('modelSel').value;
  const list = (RECS[uid] && RECS[uid][model]) || [];
  document.getElementById('recsGrid').innerHTML = list.map(r =>
    card(r.m, r.w ? `<div class="why">${r.w}</div>` : '')
  ).join('') || '<p class="note">No recommendations.</p>';
}

function renderSimilar(movieId) {
  const m = MOVIES[movieId];
  document.getElementById('similarTitle').textContent =
    m ? `Movies similar to ${m.t}` : '';
  const list = SIMILAR[movieId] || [];
  document.getElementById('similarGrid').innerHTML = list.map(s =>
    card(s.m, `<div class="why">similarity ${s.s}</div>`)
  ).join('') || '<p class="note">No similar movies.</p>';
}

function fillMovieOptions(query) {
  const sel = document.getElementById('movieSel');
  const q = (query || '').toLowerCase().trim();
  const ids = Object.keys(MOVIES)
    .filter(id => !q || MOVIES[id].t.toLowerCase().includes(q))
    .sort((a, b) => MOVIES[b].n - MOVIES[a].n)
    .slice(0, 100);
  sel.innerHTML = ids.map(id => `<option value="${id}">${MOVIES[id].t}</option>`).join('');
  if (ids.length) renderSimilar(sel.value);
  else document.getElementById('similarGrid').innerHTML = '<p class="note">No match.</p>';
}

function renderCompare() {
  if (!METRICS.length) return;
  const k = 'Precision@10';
  const maxRmse = Math.max(...METRICS.map(m => m.RMSE));
  const maxP = Math.max(...METRICS.map(m => m[k]));
  const rows = METRICS.map(m => `
    <tr>
      <td>${m.model}</td>
      <td class="num">${m.RMSE.toFixed(3)}</td>
      <td class="num">${m.MAE.toFixed(3)}</td>
      <td class="num">${m[k].toFixed(3)}</td>
      <td style="width:180px"><div class="bar" style="width:${(m[k] / maxP * 100).toFixed(0)}%"></div></td>
    </tr>`).join('');
  document.getElementById('metricsWrap').innerHTML = `
    <table class="metrics">
      <thead><tr><th>Model</th><th>RMSE ↓</th><th>MAE ↓</th><th>Precision@10 ↑</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function setupTabs() {
  document.querySelectorAll('.tab').forEach(t => {
    t.onclick = () => {
      document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      document.getElementById(t.dataset.tab).classList.add('active');
    };
  });
}

async function main() {
  [META, MOVIES, RECS, SIMILAR, METRICS] = await Promise.all(
    ['meta', 'movies', 'recs', 'similar', 'metrics'].map(loadJSON)
  );

  const userSel = document.getElementById('userSel');
  userSel.innerHTML = META.users.map(u => `<option value="${u}">User ${u}</option>`).join('');
  const modelSel = document.getElementById('modelSel');
  modelSel.innerHTML = META.models.map(m => `<option value="${m}">${m}</option>`).join('');
  modelSel.value = META.default_model;

  userSel.onchange = renderRecs;
  modelSel.onchange = renderRecs;
  renderRecs();

  const search = document.getElementById('movieSearch');
  search.oninput = () => fillMovieOptions(search.value);
  document.getElementById('movieSel').onchange = e => renderSimilar(e.target.value);
  fillMovieOptions('Matrix');

  renderCompare();
  setupTabs();
}

main();