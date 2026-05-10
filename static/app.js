/* ═══════════════════════════════════════════════════════════════
   KingsGuard Dashboard — app.js
   ═══════════════════════════════════════════════════════════════ */

// ─── Chart defaults ───────────────────────────────────────────
Chart.defaults.color = '#8a8a6a';
Chart.defaults.font.family = "'Share Tech Mono', monospace";
Chart.defaults.borderColor = '#313E17';

// ─── Matrix Rain ──────────────────────────────────────────────
(function initMatrixRain() {
  const canvas = document.getElementById('matrix-rain');
  const ctx    = canvas.getContext('2d');
  let cols, drops;

  const chars = 'KINGSGUARD01アイウエオカキクケコサシスセソタチツテト∑∂∆ΦΨΩΛΣ';

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    cols  = Math.floor(canvas.width / 16);
    drops = Array(cols).fill(1);
  }

  function draw() {
    ctx.fillStyle = 'rgba(14,5,5,0.05)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#4C5C2D';
    ctx.font = '14px Share Tech Mono';
    for (let i = 0; i < drops.length; i++) {
      const ch = chars[Math.floor(Math.random() * chars.length)];
      ctx.fillText(ch, i * 16, drops[i] * 16);
      if (drops[i] * 16 > canvas.height && Math.random() > 0.975)
        drops[i] = 0;
      drops[i]++;
    }
  }

  resize();
  window.addEventListener('resize', resize);
  setInterval(draw, 50);
})();

// ─── Tabs ─────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => { p.classList.remove('active'); p.classList.add('hidden'); });
    btn.classList.add('active');
    const panel = document.getElementById('tab-' + btn.dataset.tab);
    panel.classList.remove('hidden');
    panel.classList.add('active');
  });
});

// ─── Model preload status polling ─────────────────────────────
async function fetchPreloadStatus() {
  try {
    const res  = await fetch('/api/preload/status');
    const data = await res.json();
    updatePill('l1',  data.l1);
    updatePill('ppl', data.ppl);
    updatePill('l2',  data.l2);
    // Keep polling while any model is loading
    const anyLoading = Object.values(data).some(m => m.status === 'loading' || m.status === 'pending');
    if (anyLoading) setTimeout(fetchPreloadStatus, 1500);
  } catch (e) { /* server not ready */ }
}

function updatePill(key, info) {
  const pill = document.getElementById('pill-' + key);
  if (!pill) return;
  const dot  = pill.querySelector('.dot');
  dot.className = 'dot dot--' + info.status;
  if (info.status === 'ready') {
    pill.title = `Loaded in ${info.time_ms} ms`;
  } else if (info.status === 'error') {
    pill.title = info.error;
    dot.className = 'dot dot--error';
  }
}

document.getElementById('btn-preload').addEventListener('click', async () => {
  await fetch('/api/preload/trigger', { method: 'POST' });
  fetchPreloadStatus();
});

// Kickoff status polling
fetchPreloadStatus();

// ─── Pipeline Layers Config ───────────────────────────────────
const LAYER_CONFIG = {
  'L4_pre':  { label: 'L4',  name: 'WATCHMAN PRE-CHECK',  role: 'Behavioral Monitor' },
  'L1':      { label: 'L1',  name: 'INTENT SCREENER',     role: 'Semantic Classifier' },
  'L1_ppl':  { label: 'L1+', name: 'PERPLEXITY FILTER',   role: 'Adversarial Camouflage Detector' },
  'L2':      { label: 'L2',  name: 'VAE PROFILER',        role: 'Zero-Day Anomaly Detector' },
  'L3':      { label: 'L3',  name: 'COUNCIL OF RIVALS',   role: 'Causal Logic Gate' },
  'L4_post': { label: 'L4',  name: 'TRUST SCORE UPDATE',  role: 'BOCPD + CUSUM' },
};

// ─── Query submit ─────────────────────────────────────────────
document.getElementById('btn-analyze').addEventListener('click', submitQuery);
document.getElementById('prompt-input').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') submitQuery();
});

document.querySelectorAll('.qt-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('prompt-input').value = btn.dataset.prompt;
  });
});

let currentEvtSrc = null;

async function submitQuery() {
  const prompt  = document.getElementById('prompt-input').value.trim();
  const agentId = document.getElementById('agent-id-input').value.trim() || 'default_user';
  if (!prompt) return;

  if (currentEvtSrc) { currentEvtSrc.close(); }

  // Reset UI
  resetPipeline();
  document.getElementById('verdict-idle').classList.remove('hidden');
  document.getElementById('verdict-result').classList.add('hidden');

  // Submit
  const res    = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, agent_id: agentId }),
  });
  const { job_id } = await res.json();

  // Subscribe to SSE
  currentEvtSrc = new EventSource(`/api/stream/${job_id}`);
  currentEvtSrc.onmessage = e => handleEvent(JSON.parse(e.data));
  currentEvtSrc.onerror = () => currentEvtSrc.close();
}

// ─── Pipeline rendering ───────────────────────────────────────
const layerCards = {};

function resetPipeline() {
  const track = document.getElementById('pipeline-track');
  track.innerHTML = '<div class="pipeline-idle-msg" id="pipeline-idle-msg">Initializing…</div>';
  Object.keys(layerCards).forEach(k => delete layerCards[k]);
}

function makeLayerCard(layerId) {
  const cfg = LAYER_CONFIG[layerId] || { label: layerId, name: layerId, role: '' };
  const div  = document.createElement('div');
  div.className = 'layer-card';
  div.id = 'lc-' + layerId;
  div.innerHTML = `
    <div class="lc-header">
      <span class="lc-id">${cfg.label}</span>
      <span class="lc-status pending" id="lcs-${layerId}">PENDING</span>
    </div>
    <div class="lc-name">${cfg.name}</div>
    <div class="lc-role">${cfg.role}</div>
    <div class="lc-metrics" id="lcm-${layerId}"></div>
    <div class="lc-output hidden" id="lco-${layerId}"></div>
  `;
  layerCards[layerId] = div;
  return div;
}

function getOrCreateCard(layerId) {
  if (layerCards[layerId]) return layerCards[layerId];
  const idle = document.getElementById('pipeline-idle-msg');
  if (idle) idle.remove();
  const card = makeLayerCard(layerId);
  document.getElementById('pipeline-track').appendChild(card);
  return card;
}

function setLayerStatus(layerId, status, text) {
  const card  = getOrCreateCard(layerId);
  const badge = document.getElementById('lcs-' + layerId);
  if (badge) {
    badge.textContent  = text || status;
    badge.className    = 'lc-status ' + status.toLowerCase().replace(/[^a-z]/g, '');
  }
  card.className = 'layer-card ' + (status === 'RUNNING' ? 'active' : status.toLowerCase().startsWith('threat') || status.toLowerCase().includes('error') || status.toLowerCase().includes('revoked') ? 'threat' : 'ok');
}

function addMetric(layerId, val, lbl) {
  const el = document.getElementById('lcm-' + layerId);
  if (!el) return;
  const div = document.createElement('div');
  div.className = 'lc-metric';
  div.innerHTML = `<div class="lc-metric-val">${val}</div><div class="lc-metric-lbl">${lbl}</div>`;
  el.appendChild(div);
}

function showOutput(layerId, text) {
  const el = document.getElementById('lco-' + layerId);
  if (!el) return;
  el.textContent = text;
  el.classList.remove('hidden');
}

function handleEvent(msg) {
  const { type, data } = msg;
  if (!data) return;

  switch (type) {
    case 'pipeline_start':
      break;

    case 'layer_start':
      getOrCreateCard(data.layer);
      setLayerStatus(data.layer, 'RUNNING', 'RUNNING');
      break;

    case 'layer_complete': {
      const st = data.status === 'ok' || !data.status ? 'OK' : data.status;
      setLayerStatus(data.layer, 'OK', st);
      addMetric(data.layer, `${data.ms} ms`, 'TIME');
      // Show key metric
      if (data.result) {
        const sm = data.result.security_metadata;
        if (sm?.confidence)  addMetric(data.layer, sm.confidence.toFixed(3), 'CONF');
        if (sm?.perplexity)  addMetric(data.layer, sm.perplexity.toFixed(1), 'PPL');
        if (data.result.score !== undefined) addMetric(data.layer, data.result.score.toFixed(4), 'MSE');
        if (data.result.avg_risk !== undefined) addMetric(data.layer, data.result.avg_risk.toFixed(3), 'RISK');
        const snippet = JSON.stringify(data.result).slice(0, 140);
        showOutput(data.layer, snippet + '…');
      }
      break;
    }

    case 'layer_error': {
      setLayerStatus(data.layer, 'THREAT', data.status || 'ERROR');
      addMetric(data.layer, `${data.ms} ms`, 'TIME');
      showOutput(data.layer, data.error || 'Error');
      break;
    }

    case 'pipeline_complete': {
      renderVerdict(data);
      loadHistory(); // Auto-refresh history
      break;
    }

    case 'pipeline_error':
      console.error('Pipeline error:', data.error);
      break;
  }
}

function renderVerdict(data) {
  document.getElementById('verdict-idle').classList.add('hidden');
  const vr = document.getElementById('verdict-result');
  vr.classList.remove('hidden');

  const badge  = document.getElementById('verdict-badge');
  badge.textContent = data.verdict || '—';
  badge.className   = 'verdict-badge';
  if (data.verdict === 'APPROVED')   badge.classList.add('approved');
  else if (data.verdict === 'BLOCKED') badge.classList.add('blocked');
  else                               badge.classList.add('quarantine');

  document.getElementById('vm-risk').textContent = data.avg_risk !== undefined ? data.avg_risk.toFixed(4) : '—';
  document.getElementById('vm-time').textContent = data.total_ms ? data.total_ms + ' ms' : '—';
}

// ═══════════════════════════════════════════════════════════════
// DATASET TAB
// ═══════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════
// DATASET TAB
// ═══════════════════════════════════════════════════════════════
let allDatasets = [];
let datasetCharts = {};

async function loadDatasets() {
  const res = await fetch('/api/datasets/info');
  allDatasets = await res.json();
  
  // Populate dataset list in Repository
  const listEl = document.getElementById('dataset-list');
  listEl.innerHTML = '';
  
  // Also populate Benchmark dropdown
  const benchSelect = document.getElementById('bench-dataset');
  if (benchSelect) benchSelect.innerHTML = '';

  allDatasets.forEach((ds, idx) => {
    // Repository list
    const item = document.createElement('div');
    item.className = 'dataset-item';
    item.innerHTML = `
      <div class="dataset-item-name">${ds.name}</div>
      <div class="dataset-item-meta">${ds.type} · ${ds.records} records · ${ds.size_kb} KB</div>
    `;
    item.onclick = () => selectDataset(idx, item);
    listEl.appendChild(item);

    // Benchmark dropdown
    if (benchSelect) {
      const opt = document.createElement('option');
      opt.value = ds.name;
      opt.textContent = ds.name;
      benchSelect.appendChild(opt);
    }
  });
}

function selectDataset(idx, el) {
  document.querySelectorAll('.dataset-item').forEach(i => i.classList.remove('active'));
  el.classList.add('active');
  
  const ds = allDatasets[idx];
  document.getElementById('dataset-detail-empty').classList.add('hidden');
  document.getElementById('dataset-detail-content').classList.remove('hidden');
  
  document.getElementById('ds-name').textContent = ds.name;
  document.getElementById('ds-type').textContent = ds.type;
  document.getElementById('ds-size').textContent = ds.size_kb + ' KB';
  document.getElementById('ds-records').textContent = ds.records + ' RECORDS';
  
  renderPreview(ds.samples);
}

function renderPreview(samples) {
  const thead = document.getElementById('ds-preview-thead');
  const tbody = document.getElementById('ds-preview-tbody');
  thead.innerHTML = '';
  tbody.innerHTML = '';
  
  if (!samples || !samples.length) return;
  
  const keys = Object.keys(samples[0]);
  const trHead = document.createElement('tr');
  keys.forEach(k => {
    const th = document.createElement('th');
    th.textContent = k;
    trHead.appendChild(th);
  });
  thead.appendChild(trHead);
  
  samples.forEach(s => {
    const tr = document.createElement('tr');
    keys.forEach(k => {
      const td = document.createElement('td');
      let val = s[k];
      if (typeof val === 'object') val = JSON.stringify(val);
      td.textContent = String(val).slice(0, 100) + (String(val).length > 100 ? '...' : '');
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

// Initial load
loadDatasets();

// ═══════════════════════════════════════════════════════════════
// BENCHMARK TAB
// ═══════════════════════════════════════════════════════════════
let globalCompareChart = null;

document.getElementById('btn-benchmark-all').addEventListener('click', runBenchmarkAll);

async function runBenchmarkAll() {
  const res = await fetch('/api/benchmark/all', { method: 'POST' });
  const data = await res.json();
  if (data.error) return alert(data.error);
  
  document.getElementById('bench-loading').classList.remove('hidden');
  pollBenchmarkStatus();
}

async function pollBenchmarkStatus() {
  const res = await fetch('/api/benchmark/status');
  const data = await res.json();
  
  const fill = document.getElementById('bench-progress-fill');
  const text = document.getElementById('bench-loading-text');
  
  fill.style.width = data.progress + '%';
  text.textContent = `Processing: ${data.current_dataset} (${data.progress}%)`;
  
  if (data.status === 'running') {
    setTimeout(pollBenchmarkStatus, 2000);
  } else {
    document.getElementById('bench-loading').classList.add('hidden');
    loadBenchmarkResults();
  }
}

async function loadBenchmarkResults() {
  try {
    const res = await fetch('/api/benchmark/results');
    const data = await res.json();
    if (data.error) return;
    
    document.getElementById('bench-all-results').classList.remove('hidden');
    renderGlobalCompare(data.datasets);
  } catch (e) {}
}

function renderGlobalCompare(dsData) {
  const ctx = document.getElementById('chart-global-compare');
  const tbody = document.getElementById('bench-metrics-tbody');
  if (!ctx || !tbody) return;
  
  if (globalCompareChart) globalCompareChart.destroy();
  
  const labels = Object.keys(dsData);
  const accs = labels.map(l => dsData[l].accuracy * 100);
  const times = labels.map(l => dsData[l].avg_ms);
  
  // Update Table
  tbody.innerHTML = '';
  labels.forEach(l => {
    const d = dsData[l];
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${l}</td>
      <td style="color:var(--yellow); font-weight:bold;">${(d.accuracy * 100).toFixed(1)}%</td>
      <td>${d.avg_ms} ms</td>
      <td style="color:#6fcf6f">${d.tp}</td>
      <td style="color:#ff6688">${d.fp}</td>
      <td style="color:#a8c868">${d.tn}</td>
      <td style="color:#ffaa44">${d.fn}</td>
    `;
    tbody.appendChild(tr);
  });

  globalCompareChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Accuracy %',
          data: accs,
          backgroundColor: 'rgba(255, 222, 66, 0.6)',
          borderColor: '#FFDE42',
          borderWidth: 1,
          yAxisID: 'y'
        },
        {
          label: 'Avg Latency (ms)',
          data: times,
          type: 'line',
          borderColor: '#6fcf6f',
          pointBackgroundColor: '#6fcf6f',
          borderWidth: 2,
          yAxisID: 'y1'
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { color: '#8a8a6a' }, title: { display: true, text: 'Accuracy %', color: '#8a8a6a' } },
        y1: { position: 'right', beginAtZero: true, grid: { display: false }, ticks: { color: '#8a8a6a' }, title: { display: true, text: 'Latency (ms)', color: '#8a8a6a' } },
        x: { ticks: { color: '#d4d4b0' } }
      },
      plugins: {
        legend: { labels: { color: '#8a8a6a' } }
      }
    }
  });
}

// Check for existing results on load
loadBenchmarkResults();

// ═══════════════════════════════════════════════════════════════
// BENCHMARK TAB
// ═══════════════════════════════════════════════════════════════
let benchCharts = {};

document.getElementById('btn-benchmark').addEventListener('click', runBenchmark);

async function loadDatasetList() {
  const res = await fetch('/api/datasets/list');
  const files = await res.json();
  const select = document.getElementById('bench-dataset');
  select.innerHTML = '';
  files.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f;
    opt.textContent = f;
    select.appendChild(opt);
  });
}

loadDatasetList();

async function runBenchmark() {
  const n = parseInt(document.getElementById('bench-n').value) || 20;
  const dataset = document.getElementById('bench-dataset').value;
  document.getElementById('bench-loading').classList.remove('hidden');
  document.getElementById('bench-results-tbody').innerHTML = '';

  try {
    const res  = await fetch('/api/dataset/benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ samples: n, dataset: dataset }),
    });
    const data = await res.json();

    document.getElementById('bench-loading').classList.add('hidden');

    if (data.error) {
      alert('Benchmark error: ' + data.error);
      return;
    }

    const m = data.metrics;
    document.getElementById('bm-accuracy').textContent  = pct(m.accuracy);
    document.getElementById('bm-precision').textContent = pct(m.precision);
    document.getElementById('bm-recall').textContent    = pct(m.recall);
    document.getElementById('bm-f1').textContent        = pct(m.f1);
    document.getElementById('bm-avg-ms').textContent    = m.avg_ms + ' ms';

    // Confusion matrix
    const cm = data.confusion_matrix;
    document.getElementById('cm-tp').querySelector('.cm-cell-val').textContent = cm.tp;
    document.getElementById('cm-fp').querySelector('.cm-cell-val').textContent = cm.fp;
    document.getElementById('cm-tn').querySelector('.cm-cell-val').textContent = cm.tn;
    document.getElementById('cm-fn').querySelector('.cm-cell-val').textContent = cm.fn;

    // Layer failure rates
    const layerStats = { L1: 0, PPL: 0, L2: 0, L3: 0 };
    const totalRows = data.rows.length;
    data.rows.forEach(r => {
      if (r.layers.L1 && r.layers.L1.status === 'FAIL') layerStats.L1++;
      if (r.layers.PPL && r.layers.PPL.status === 'FAIL') layerStats.PPL++;
      if (r.layers.L2 && r.layers.L2.status === 'ANOMALY_DETECTED') layerStats.L2++;
      if (r.layers.L3 && r.layers.L3.status === 'QUARANTINE') layerStats.L3++;
    });

    renderLayerChart('chart-layers-perf', 
      ['L1 Screener', 'Perplexity', 'VAE Profiler', 'Arbitrator'],
      [layerStats.L1, layerStats.PPL, layerStats.L2, layerStats.L3],
      totalRows
    );

    // Results table
    const tbody = document.getElementById('bench-results-tbody');
    tbody.innerHTML = '';
    for (const row of (data.rows || [])) {
      const tr = document.createElement('tr');
      const l1 = row.layers.L1 ? `<span class="tag tag-${row.layers.L1.status.toLowerCase()}">${row.layers.L1.score.toFixed(3)}</span>` : '—';
      const ppl = row.layers.PPL ? `<span class="tag tag-${row.layers.PPL.status.toLowerCase()}">${row.layers.PPL.score.toFixed(1)}</span>` : '—';
      const l2 = row.layers.L2 ? `<span class="tag tag-${row.layers.L2.status === 'BEHAVIOR_NORMAL' ? 'pass' : 'fail'}">${row.layers.L2.score.toFixed(4)}</span>` : '—';
      const l3 = row.layers.L3 ? `<span class="tag tag-${row.layers.L3.status === 'APPROVED' ? 'pass' : 'fail'}">${row.layers.L3.score.toFixed(3)}</span>` : '—';

      tr.innerHTML = `
        <td style="max-width:260px; font-size:0.75rem;">${escHtml(row.text)}</td>
        <td><span class="badge badge-${row.true_label}">${row.true_label === 'malicious' ? 'MAL' : 'BEN'}</span></td>
        <td><span class="badge badge-${row.pred_label}">${row.pred_label === 'malicious' ? 'MAL' : 'BEN'}</span></td>
        <td>${l1}</td>
        <td>${ppl}</td>
        <td>${l2}</td>
        <td>${l3}</td>
        <td style="font-size:0.75rem;">${row.ms}ms</td>
        <td><span class="${row.correct ? 'badge-correct' : 'badge-wrong'}">${row.correct ? '✓' : '✗'}</span></td>`;
      tbody.appendChild(tr);
    }
  } catch (e) {
    document.getElementById('bench-loading').classList.add('hidden');
    alert('Error: ' + e.message);
  }
}

function renderLayerChart(id, labels, values, total) {
  if (benchCharts[id]) { benchCharts[id].destroy(); }
  const ctx = document.getElementById(id);
  if (!ctx) return;
  benchCharts[id] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Threats Blocked / Flagged',
        data: values,
        backgroundColor: 'rgba(255, 222, 66, 0.4)',
        borderColor: '#FFDE42',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, max: total, ticks: { color: '#8a8a6a' }, grid: { color: '#313E17' } },
        x: { ticks: { color: '#d4d4b0' }, grid: { display: false } }
      }
    }
  });
}

function pct(v) { return (v * 100).toFixed(1) + '%'; }
function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ═══════════════════════════════════════════════════════════════
// HISTORY TAB
// ═══════════════════════════════════════════════════════════════
document.getElementById('btn-refresh-history').addEventListener('click', loadHistory);

async function loadHistory() {
  const res  = await fetch('/api/history');
  const rows = await res.json();
  const tbody = document.getElementById('history-tbody');
  const empty = document.getElementById('history-empty');
  tbody.innerHTML = '';
  if (!rows.length) { empty.classList.remove('hidden'); return; }
  empty.classList.add('hidden');
  for (const r of rows) {
    const tr = document.createElement('tr');
    const ts = r.timestamp ? new Date(r.timestamp).toLocaleTimeString() : '—';
    const vCls = r.verdict === 'APPROVED' ? 'approved' : r.verdict === 'BLOCKED' ? 'blocked' : 'quarantine';
    tr.innerHTML = `
      <td>${ts}</td>
      <td style="max-width:240px;word-break:break-word;">${escHtml(r.prompt || '—')}</td>
      <td style="font-family:var(--font-mono);font-size:.7rem;">${escHtml(r.agent_id || '—')}</td>
      <td><span class="badge badge-${vCls}">${r.verdict || '—'}</span></td>
      <td style="font-family:var(--font-mono);">${r.avg_risk !== undefined ? r.avg_risk.toFixed(4) : '—'}</td>
      <td style="font-family:var(--font-mono);">${r.total_ms || '—'} ms</td>`;
    tbody.appendChild(tr);
  }
}
