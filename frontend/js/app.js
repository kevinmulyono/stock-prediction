/**
 * IDX AI Stock Trader Dashboard (Track C - Ali)
 * Frontend Application Logic & ApexCharts Integration
 */

// Global State
const state = {
  selectedModel: 'ensemble',
  selectedTicker: 'BBCA.JK',
  selectedTimeframe: '1Y',
  showSignalsOverlay: true,
  stocks: [],
  latestPredictions: {},
  activePrediction: null,
  candlestickChart: null,
  comparisonChart: null,
  modelRegistry: null
};

// DOM Elements
const elements = {
  apiStatusBadge: document.getElementById('apiStatusBadge'),
  apiStatusText: document.getElementById('apiStatusText'),
  modelSelect: document.getElementById('modelSelect'),
  btnTriggerDaily: document.getElementById('btnTriggerDaily'),
  btnOpenModelInfo: document.getElementById('btnOpenModelInfo'),
  btnCloseModal: document.getElementById('btnCloseModal'),
  modelInfoModal: document.getElementById('modelInfoModal'),
  indicatorsTagCloud: document.getElementById('indicatorsTagCloud'),
  targetDateBadge: document.getElementById('targetDateBadge'),
  tickerStripContainer: document.getElementById('tickerStripContainer'),
  chartTickerSymbol: document.getElementById('chartTickerSymbol'),
  chartCompanyName: document.getElementById('chartCompanyName'),
  chartLatestPrice: document.getElementById('chartLatestPrice'),
  timeframeGroup: document.getElementById('timeframeGroup'),
  toggleSignalsOverlay: document.getElementById('toggleSignalsOverlay'),
  candlestickChartContainer: document.getElementById('candlestickChartContainer'),
  widgetModelBadge: document.getElementById('widgetModelBadge'),
  signalBanner: document.getElementById('signalBanner'),
  signalBannerIcon: document.getElementById('signalBannerIcon'),
  signalBannerAction: document.getElementById('signalBannerAction'),
  signalBannerTargetDate: document.getElementById('signalBannerTargetDate'),
  sidebarProbUp: document.getElementById('sidebarProbUp'),
  sidebarProbBar: document.getElementById('sidebarProbBar'),
  sidebarRecommendationText: document.getElementById('sidebarRecommendationText'),
  leaderboardTableBody: document.getElementById('leaderboardTableBody'),
  mlMetricsTableBody: document.getElementById('mlMetricsTableBody'),
  strategyComparisonChart: document.getElementById('strategyComparisonChart'),
  toastContainer: document.getElementById('toastContainer')
};

// Utilities
function formatIDR(value) {
  if (value === null || value === undefined) return 'Rp -';
  return 'Rp ' + Number(value).toLocaleString('id-ID', { maximumFractionDigits: 0 });
}

function showToast(message, type = 'success') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span>${type === 'success' ? '✅' : '⚠️'}</span>
    <span>${message}</span>
  `;
  elements.toastContainer.appendChild(toast);
  setTimeout(() => toast.classList.add('show'), 10);
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 400);
  }, 4000);
}

// API Service
const api = {
  async getHealth() {
    const res = await fetch('/api/health');
    return await res.json();
  },
  async getStocks() {
    const res = await fetch('/api/stocks');
    return await res.json();
  },
  async getStockHistory(ticker, timeframe) {
    const res = await fetch(`/api/stocks/${ticker}/history?timeframe=${timeframe}`);
    return await res.json();
  },
  async getLatestPredictions(model) {
    const res = await fetch(`/api/predictions/latest?model=${model}`);
    return await res.json();
  },
  async getSignalsOverlay(ticker, model, timeframe) {
    const res = await fetch(`/api/predictions/signals-overlay?ticker=${ticker}&model=${model}&timeframe=${timeframe}`);
    return await res.json();
  },
  async getBacktestSummary() {
    const res = await fetch('/api/backtest/summary');
    return await res.json();
  },
  async getModelComparison() {
    const res = await fetch('/api/models/comparison');
    return await res.json();
  },
  async getModelRegistry() {
    const res = await fetch('/api/models/registry');
    return await res.json();
  },
  async triggerDailyPrediction(model) {
    const res = await fetch('/api/predictions/trigger-daily', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model })
    });
    return await res.json();
  }
};

// UI Renderers
function renderTickerStrip() {
  elements.tickerStripContainer.innerHTML = '';

  state.stocks.forEach(stock => {
    const pred = state.latestPredictions[stock.ticker];
    const isSelected = stock.ticker === state.selectedTicker;

    const card = document.createElement('div');
    card.className = `ticker-card ${isSelected ? 'active' : ''}`;
    card.setAttribute('data-ticker', stock.ticker);

    const signal = pred ? pred.signal : 'HOLD';
    const probUp = pred ? pred.probability_up : 50.0;
    const signalClass = signal.toLowerCase();

    const changeClass = (stock.change_pct || 0) >= 0 ? 'up' : 'down';
    const changeIcon = (stock.change_pct || 0) >= 0 ? '▲' : '▼';
    const changeSign = (stock.change_pct || 0) >= 0 ? '+' : '';

    let probMeterClass = 'neutral';
    if (signal === 'BUY') probMeterClass = 'bull';
    else if (signal === 'SELL') probMeterClass = 'bear';

    card.innerHTML = `
      <div class="ticker-card-top">
        <div>
          <div class="ticker-symbol">${stock.ticker.replace('.JK', '')}</div>
          <div class="ticker-name" title="${stock.company_name}">${stock.company_name}</div>
        </div>
        <div class="signal-badge ${signalClass}">${signal}</div>
      </div>
      <div class="ticker-price">${formatIDR(stock.latest_close)}</div>
      <div class="ticker-change ${changeClass}">
        <span>${changeIcon}</span>
        <span>${changeSign}${(stock.change_pct || 0).toFixed(2)}%</span>
      </div>
      <div class="prob-meter-container">
        <div class="prob-meter-label">
          <span>P(Up)</span>
          <strong>${probUp.toFixed(1)}%</strong>
        </div>
        <div class="prob-meter-bar">
          <div class="prob-meter-fill ${probMeterClass}" style="width: ${Math.min(100, Math.max(0, probUp))}%"></div>
        </div>
      </div>
    `;

    card.addEventListener('click', () => {
      selectTicker(stock.ticker);
    });

    elements.tickerStripContainer.appendChild(card);
  });
}

function updateActiveSidebar() {
  const stock = state.stocks.find(s => s.ticker === state.selectedTicker);
  const pred = state.latestPredictions[state.selectedTicker];

  if (stock) {
    elements.chartTickerSymbol.textContent = stock.ticker;
    elements.chartCompanyName.textContent = stock.company_name;
    elements.chartLatestPrice.textContent = formatIDR(stock.latest_close);
  }

  elements.widgetModelBadge.textContent = state.selectedModel.toUpperCase();

  if (pred) {
    const signal = pred.signal;
    const signalClass = signal.toLowerCase();

    elements.signalBanner.className = `signal-banner ${signalClass}`;
    elements.signalBannerIcon.textContent = signal === 'BUY' ? '🟢' : (signal === 'SELL' ? '🔴' : '🟡');
    elements.signalBannerAction.textContent = `${signal} SIGNAL`;
    elements.signalBannerTargetDate.textContent = `Target: ${pred.target_date} (T+1)`;

    elements.sidebarProbUp.textContent = `${pred.probability_up.toFixed(1)}%`;
    elements.sidebarProbBar.style.width = `${pred.probability_up}%`;

    let meterClass = 'neutral';
    if (signal === 'BUY') meterClass = 'bull';
    else if (signal === 'SELL') meterClass = 'bear';
    elements.sidebarProbBar.className = `prob-meter-fill ${meterClass}`;

    elements.sidebarRecommendationText.textContent = pred.recommendation || 'Model memantau volatilitas pasar.';
  }
}

async function renderCandlestickChart() {
  try {
    const [historyRes, signalsRes] = await Promise.all([
      api.getStockHistory(state.selectedTicker, state.selectedTimeframe),
      state.showSignalsOverlay
        ? api.getSignalsOverlay(state.selectedTicker, state.selectedModel, state.selectedTimeframe)
        : Promise.resolve({ data: [] })
    ]);

    const candleSeriesData = historyRes.data.map(item => ({
      x: new Date(item.date).getTime(),
      y: [item.open, item.high, item.low, item.close]
    }));

    // Generate BUY/SELL annotations
    const pointAnnotations = [];
    if (state.showSignalsOverlay && signalsRes && signalsRes.data) {
      signalsRes.data.forEach(s => {
        const isBuy = s.signal === 'BUY';
        pointAnnotations.push({
          x: new Date(s.date).getTime(),
          y: s.price,
          marker: {
            size: 5,
            fillColor: isBuy ? '#10b981' : '#f43f5e',
            strokeColor: '#ffffff',
            strokeWidth: 1.5,
            shape: 'circle'
          },
          label: {
            borderColor: isBuy ? '#10b981' : '#f43f5e',
            borderWidth: 1,
            borderRadius: 4,
            textAnchor: 'middle',
            offsetY: isBuy ? -18 : 22,
            style: {
              background: isBuy ? 'rgba(16, 185, 129, 0.9)' : 'rgba(244, 63, 94, 0.9)',
              color: '#ffffff',
              fontSize: '9px',
              fontFamily: 'Inter',
              fontWeight: 700,
              padding: { left: 4, right: 4, top: 2, bottom: 2 }
            },
            text: `${s.signal}`
          }
        });
      });
    }

    const options = {
      series: [
        {
          name: `${state.selectedTicker} Price`,
          type: 'candlestick',
          data: candleSeriesData
        }
      ],
      chart: {
        type: 'candlestick',
        height: 380,
        background: 'transparent',
        toolbar: {
          show: true,
          tools: {
            download: true,
            selection: true,
            zoom: true,
            zoomin: true,
            zoomout: true,
            pan: true,
            reset: true
          }
        },
        animations: {
          enabled: true,
          easing: 'easeinout',
          speed: 400
        }
      },
      theme: {
        mode: 'dark'
      },
      plotOptions: {
        candlestick: {
          colors: {
            upward: '#10b981',
            downward: '#f43f5e'
          },
          wick: {
            useFillColor: true
          }
        }
      },
      xaxis: {
        type: 'datetime',
        labels: {
          style: {
            colors: '#94a3b8',
            fontFamily: 'Inter',
            fontSize: '11px'
          },
          datetimeFormatter: {
            year: 'yyyy',
            month: "MMM 'yy",
            day: 'dd MMM'
          }
        },
        axisBorder: {
          color: 'rgba(255, 255, 255, 0.1)'
        },
        axisTicks: {
          color: 'rgba(255, 255, 255, 0.1)'
        }
      },
      yaxis: {
        tooltip: {
          enabled: true
        },
        labels: {
          style: {
            colors: '#94a3b8',
            fontFamily: 'JetBrains Mono',
            fontSize: '11px'
          },
          formatter: (val) => Number(val).toLocaleString('id-ID')
        }
      },
      grid: {
        borderColor: 'rgba(255, 255, 255, 0.05)',
        strokeDashArray: 3
      },
      tooltip: {
        theme: 'dark',
        shared: true,
        x: {
          format: 'dd MMM yyyy'
        }
      },
      annotations: {
        points: pointAnnotations
      }
    };

    if (state.candlestickChart) {
      state.candlestickChart.updateOptions(options);
    } else {
      elements.candlestickChartContainer.innerHTML = '';
      state.candlestickChart = new ApexCharts(elements.candlestickChartContainer, options);
      await state.candlestickChart.render();
    }
  } catch (err) {
    console.error('Error rendering chart:', err);
    elements.candlestickChartContainer.innerHTML = `
      <div style="display:flex; justify-content:center; align-items:center; height:380px; color:var(--bear-red);">
        Gagal memuat grafik harga: ${err.message}
      </div>
    `;
  }
}

function renderLeaderboardTable(summaryData) {
  elements.leaderboardTableBody.innerHTML = '';

  summaryData.data.forEach((row, idx) => {
    const tr = document.createElement('tr');
    const isWinner = row.model.includes('Ensemble');
    const isBenchmark = row.model.includes('Benchmark') || row.model.includes('Buy & Hold');

    if (isWinner) tr.className = 'winner-row';
    else if (isBenchmark) tr.className = 'benchmark-row';

    let rankBadge = '';
    if (idx === 0) rankBadge = '<span class="rank-pill gold">🥇</span>';
    else if (idx === 1) rankBadge = '<span class="rank-pill silver">🥈</span>';
    else if (idx === 2) rankBadge = '<span class="rank-pill bronze">🥉</span>';

    const returnClass = row.total_return_pct >= 0 ? 'return-positive' : 'return-negative';
    const excessClass = row.excess_return_pct >= 0 ? 'return-positive' : 'return-negative';

    tr.innerHTML = `
      <td>${rankBadge} ${row.model}</td>
      <td class="${returnClass}">${row.total_return_pct >= 0 ? '+' : ''}${row.total_return_pct.toFixed(2)}%</td>
      <td class="${excessClass}">${row.excess_return_pct >= 0 ? '+' : ''}${row.excess_return_pct.toFixed(2)}%</td>
      <td>${row.sharpe_ratio.toFixed(2)}</td>
      <td class="return-negative">${row.max_drawdown_pct.toFixed(2)}%</td>
      <td>${row.win_rate_pct.toFixed(1)}%</td>
      <td>${row.trades}</td>
    `;
    elements.leaderboardTableBody.appendChild(tr);
  });
}

function renderMLMetricsTable(metricsData) {
  elements.mlMetricsTableBody.innerHTML = '';

  metricsData.data.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${row.model}</strong></td>
      <td style="color: var(--accent-cyan); font-weight:700;">${row.roc_auc.toFixed(4)}</td>
      <td>${row.accuracy.toFixed(1)}%</td>
      <td>${row.brier_score.toFixed(4)}</td>
      <td>${row.precision.toFixed(1)}%</td>
    `;
    elements.mlMetricsTableBody.appendChild(tr);
  });
}

function renderStrategyComparisonChart(summaryData) {
  const models = summaryData.data.map(d => d.model.replace('(Track B)', '').trim());
  const returns = summaryData.data.map(d => d.total_return_pct);

  const colors = returns.map(val => val >= 0 ? '#10b981' : '#f43f5e');

  const options = {
    series: [
      {
        name: 'Total Return (%)',
        data: returns
      }
    ],
    chart: {
      type: 'bar',
      height: 240,
      background: 'transparent',
      toolbar: { show: false }
    },
    plotOptions: {
      bar: {
        horizontal: true,
        distributed: true,
        borderRadius: 4,
        dataLabels: { position: 'top' }
      }
    },
    colors: colors,
    dataLabels: {
      enabled: true,
      formatter: (val) => `${val >= 0 ? '+' : ''}${val.toFixed(1)}%`,
      offsetX: 20,
      style: {
        fontSize: '11px',
        fontFamily: 'Inter',
        colors: ['#fff']
      }
    },
    xaxis: {
      labels: {
        style: { colors: '#94a3b8', fontFamily: 'Inter' },
        formatter: (val) => `${val}%`
      }
    },
    yaxis: {
      labels: {
        style: { colors: '#f8fafc', fontFamily: 'Inter', fontSize: '11px' }
      }
    },
    theme: { mode: 'dark' },
    grid: {
      borderColor: 'rgba(255, 255, 255, 0.05)'
    },
    legend: { show: false }
  };

  if (state.comparisonChart) {
    state.comparisonChart.updateOptions(options);
  } else {
    elements.strategyComparisonChart.innerHTML = '';
    state.comparisonChart = new ApexCharts(elements.strategyComparisonChart, options);
    state.comparisonChart.render();
  }
}

function renderModelRegistryModal(registryData) {
  state.modelRegistry = registryData;
  elements.indicatorsTagCloud.innerHTML = '';

  if (registryData && registryData.features && registryData.features.feature_columns) {
    registryData.features.feature_columns.forEach(feat => {
      const span = document.createElement('span');
      span.className = 'indicator-tag';
      span.textContent = feat;
      elements.indicatorsTagCloud.appendChild(span);
    });
  }
}

// Controller Actions
async function selectTicker(ticker) {
  state.selectedTicker = ticker;
  document.querySelectorAll('.ticker-card').forEach(c => {
    if (c.getAttribute('data-ticker') === ticker) c.classList.add('active');
    else c.classList.remove('active');
  });
  updateActiveSidebar();
  await renderCandlestickChart();
}

async function handleModelChange(newModel) {
  state.selectedModel = newModel;
  try {
    const predRes = await api.getLatestPredictions(newModel);
    state.latestPredictions = {};
    predRes.data.forEach(p => {
      state.latestPredictions[p.ticker] = p;
    });

    if (predRes.latest_target_date) {
      elements.targetDateBadge.textContent = `Target: ${predRes.latest_target_date} (T+1)`;
    }

    renderTickerStrip();
    updateActiveSidebar();
    await renderCandlestickChart();
    showToast(`Model berganti ke: ${newModel.toUpperCase()}`, 'success');
  } catch (err) {
    console.error('Failed to change model:', err);
    showToast(`Gagal memuat prediksi model: ${err.message}`, 'error');
  }
}

async function handleTriggerDaily() {
  const btn = elements.btnTriggerDaily;
  btn.classList.add('loading');
  btn.disabled = true;

  try {
    showToast('Memulai pipeline inferensi live harian T+1...', 'success');
    const res = await api.triggerDailyPrediction(state.selectedModel);

    showToast(res.message, 'success');
    await refreshAllData();
  } catch (err) {
    console.error('Daily prediction error:', err);
    showToast(`Gagal eksekusi scoring live: ${err.message}`, 'error');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

async function refreshAllData() {
  try {
    const [stocksRes, predsRes, backtestRes, mlRes, regRes] = await Promise.all([
      api.getStocks(),
      api.getLatestPredictions(state.selectedModel),
      api.getBacktestSummary(),
      api.getModelComparison(),
      api.getModelRegistry()
    ]);

    state.stocks = stocksRes.data;
    state.latestPredictions = {};
    predsRes.data.forEach(p => {
      state.latestPredictions[p.ticker] = p;
    });

    if (predsRes.latest_target_date) {
      elements.targetDateBadge.textContent = `Target: ${predsRes.latest_target_date} (T+1)`;
    }

    renderTickerStrip();
    updateActiveSidebar();
    await renderCandlestickChart();
    renderLeaderboardTable(backtestRes);
    renderStrategyComparisonChart(backtestRes);
    renderMLMetricsTable(mlRes);
    renderModelRegistryModal(regRes);
  } catch (err) {
    console.error('Initialization error:', err);
    showToast('Koneksi API gagal: ' + err.message, 'error');
    elements.apiStatusBadge.style.borderColor = 'var(--bear-red)';
    elements.apiStatusText.textContent = 'API Offline';
  }
}

// Event Listeners Initialization
function initEventListeners() {
  elements.modelSelect.addEventListener('change', (e) => {
    handleModelChange(e.target.value);
  });

  elements.btnTriggerDaily.addEventListener('click', () => {
    handleTriggerDaily();
  });

  elements.timeframeGroup.querySelectorAll('.tf-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      elements.timeframeGroup.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.selectedTimeframe = btn.getAttribute('data-tf');
      await renderCandlestickChart();
    });
  });

  elements.toggleSignalsOverlay.addEventListener('change', async (e) => {
    state.showSignalsOverlay = e.target.checked;
    await renderCandlestickChart();
  });

  elements.btnOpenModelInfo.addEventListener('click', () => {
    elements.modelInfoModal.classList.add('open');
  });

  elements.btnCloseModal.addEventListener('click', () => {
    elements.modelInfoModal.classList.remove('open');
  });

  elements.modelInfoModal.addEventListener('click', (e) => {
    if (e.target === elements.modelInfoModal) {
      elements.modelInfoModal.classList.remove('open');
    }
  });
}

// Startup
document.addEventListener('DOMContentLoaded', async () => {
  initEventListeners();
  await refreshAllData();
});
