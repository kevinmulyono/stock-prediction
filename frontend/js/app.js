/**
 * IDX AI Stock Trader Dashboard (Track C - Ali)
 * Frontend Application Logic with TradingView Lightweight Charts
 * Stockbit / Investing.com Experience with Smart Signal Filtering & Fullscreen
 */

// Global State
const state = {
  selectedModel: 'ensemble',
  selectedTicker: 'BBCA.JK',
  selectedTimeframe: '6M', // Default to 6M for wide, readable candles like Stockbit
  signalOverlayMode: 'entries', // 'entries' | 'all' | 'off'
  isFullscreen: false,
  stocks: [],
  latestPredictions: {},
  activePrediction: null,
  rawHistoryData: [],
  rawSignalsData: [],
  tvChart: null,
  candleSeries: null,
  volumeSeries: null,
  comparisonChart: null, // ApexCharts for strategy return bar comparison
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
  chartPanel: document.getElementById('chartPanel'),
  chartTickerSymbol: document.getElementById('chartTickerSymbol'),
  chartCompanyName: document.getElementById('chartCompanyName'),
  chartLatestPrice: document.getElementById('chartLatestPrice'),
  timeframeGroup: document.getElementById('timeframeGroup'),
  signalOverlayMode: document.getElementById('signalOverlayMode'),
  btnFullscreen: document.getElementById('btnFullscreen'),
  btnResetZoom: document.getElementById('btnResetZoom'),
  chartLegendBar: document.getElementById('chartLegendBar'),
  legendStockTitle: document.getElementById('legendStockTitle'),
  legendOpen: document.getElementById('legendOpen'),
  legendHigh: document.getElementById('legendHigh'),
  legendLow: document.getElementById('legendLow'),
  legendClose: document.getElementById('legendClose'),
  legendChange: document.getElementById('legendChange'),
  legendVolume: document.getElementById('legendVolume'),
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

function formatVolume(val) {
  if (!val || val === 0) return '0';
  const num = Number(val);
  if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
  if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
  return num.toLocaleString('id-ID');
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

// ===================================================================
// TRADINGVIEW LIGHTWEIGHT CHARTS ENGINE (STOCKBIT / INVESTING.COM UX)
// ===================================================================

function initTradingViewChart() {
  if (typeof LightweightCharts === 'undefined') {
    console.error('TradingView LightweightCharts library not loaded.');
    return;
  }

  const container = elements.candlestickChartContainer;
  container.innerHTML = '';

  const chartOptions = {
    layout: {
      textColor: '#94a3b8',
      background: { type: 'solid', color: '#0a0e17' },
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
      fontSize: 11
    },
    grid: {
      vertLines: { color: 'rgba(255, 255, 255, 0.04)' },
      horzLines: { color: 'rgba(255, 255, 255, 0.04)' }
    },
    crosshair: {
      mode: LightweightCharts.CrosshairMode.Normal,
      vertLine: {
        width: 1,
        color: 'rgba(99, 102, 241, 0.5)',
        style: LightweightCharts.LineStyle.Dashed,
        labelBackgroundColor: '#6366f1'
      },
      horzLine: {
        width: 1,
        color: 'rgba(99, 102, 241, 0.5)',
        style: LightweightCharts.LineStyle.Dashed,
        labelBackgroundColor: '#6366f1'
      }
    },
    rightPriceScale: {
      borderColor: 'rgba(255, 255, 255, 0.08)',
      scaleMargins: {
        top: 0.1,
        bottom: 0.22 // Leaves 22% space at bottom for volume bars
      },
      autoScale: true
    },
    timeScale: {
      borderColor: 'rgba(255, 255, 255, 0.08)',
      timeVisible: true,
      secondsVisible: false,
      rightOffset: 6,
      barSpacing: 8,
      minBarSpacing: 3
    },
    handleScroll: {
      mouseWheel: true,
      pressedMouseMove: true,
      horzTouchDrag: true,
      vertTouchDrag: true
    },
    handleScale: {
      axisPressedMouseMove: true,
      mouseWheel: true,
      pinch: true
    }
  };

  state.tvChart = LightweightCharts.createChart(container, chartOptions);

  // Candlestick Series (Bullish Green / Bearish Red)
  state.candleSeries = state.tvChart.addCandlestickSeries({
    upColor: '#10b981',
    downColor: '#f43f5e',
    borderVisible: false,
    wickUpColor: '#10b981',
    wickDownColor: '#f43f5e',
    priceFormat: {
      type: 'price',
      precision: 0,
      minMove: 1
    }
  });

  // Volume Histogram Series (Stockbit style overlay)
  state.volumeSeries = state.tvChart.addHistogramSeries({
    priceFormat: { type: 'volume' },
    priceScaleId: '', // Overlay inside same chart area
    scaleMargins: {
      top: 0.8,
      bottom: 0
    }
  });

  // Subscribe Crosshair for Dynamic Stockbit/TradingView OHLCV Legend
  state.tvChart.subscribeCrosshairMove(param => {
    updateDynamicLegend(param);
  });

  // Auto-resize on window resize
  window.addEventListener('resize', handleChartResize);
}

function handleChartResize() {
  if (!state.tvChart || !elements.candlestickChartContainer) return;

  if (state.isFullscreen) {
    const width = Math.max(400, window.innerWidth - 64);
    const height = Math.max(300, window.innerHeight - 175);
    state.tvChart.applyOptions({ width, height });
  } else {
    const panel = elements.chartPanel;
    // Calculate width strictly bounded by panel client width minus padding
    const width = Math.max(300, Math.floor(panel.clientWidth - 48));
    const height = 440;
    state.tvChart.applyOptions({ width, height });
  }
}

function updateDynamicLegend(param) {
  elements.legendStockTitle.textContent = `${state.selectedTicker} · 1D · IDX`;

  let candle = null;
  let volume = null;

  if (param && param.time && state.candleSeries) {
    candle = param.seriesData.get(state.candleSeries);
    if (state.volumeSeries) {
      volume = param.seriesData.get(state.volumeSeries);
    }
  }

  // Fallback to latest available historical candle if cursor is outside chart
  if (!candle && state.rawHistoryData && state.rawHistoryData.length > 0) {
    const last = state.rawHistoryData[state.rawHistoryData.length - 1];
    candle = {
      open: last.open,
      high: last.high,
      low: last.low,
      close: last.close
    };
    volume = { value: last.volume };
  }

  if (candle) {
    elements.legendOpen.textContent = Number(candle.open).toLocaleString('id-ID');
    elements.legendHigh.textContent = Number(candle.high).toLocaleString('id-ID');
    elements.legendLow.textContent = Number(candle.low).toLocaleString('id-ID');
    elements.legendClose.textContent = Number(candle.close).toLocaleString('id-ID');

    const chgVal = candle.close - candle.open;
    const chgPct = candle.open > 0 ? (chgVal / candle.open) * 100 : 0;
    const sign = chgPct >= 0 ? '+' : '';
    const chgText = `${sign}${chgPct.toFixed(2)}% (${sign}${Number(chgVal).toLocaleString('id-ID')})`;

    elements.legendChange.textContent = chgText;
    elements.legendChange.className = chgPct >= 0 ? 'chg-up' : 'chg-down';

    if (volume && volume.value) {
      elements.legendVolume.textContent = formatVolume(volume.value);
    } else {
      elements.legendVolume.textContent = '--';
    }
  }
}

/**
 * Smart Signal Filtering:
 * Solves the issue where 100+ daily signals overlap into a solid block of red/green.
 * - 'entries': Filters consecutive duplicate signals so only genuine Trade Entries (BUY)
 *   and Position Closures (SELL) are marked with crisp native arrows!
 * - 'all': Shows small discrete markers on all signal days without overlapping text.
 * - 'off': Clears all markers for a clean pure candlestick view.
 */
function applySignalsMarkers(signalsList) {
  if (!state.candleSeries) return;

  if (state.signalOverlayMode === 'off' || !signalsList || signalsList.length === 0) {
    state.candleSeries.setMarkers([]);
    return;
  }

  const sorted = [...signalsList].sort((a, b) => new Date(a.date) - new Date(b.date));
  const markers = [];

  if (state.signalOverlayMode === 'entries') {
    let activePosition = null; // 'LONG' or null/'CASH'

    sorted.forEach(s => {
      if (s.signal === 'BUY' && activePosition !== 'LONG') {
        // ENTRY BUY SIGNAL
        markers.push({
          time: s.date,
          position: 'belowBar',
          color: '#10b981',
          shape: 'arrowUp',
          text: `BUY (${s.probability_up.toFixed(0)}%)`
        });
        activePosition = 'LONG';
      } else if (s.signal === 'SELL' && activePosition === 'LONG') {
        // EXIT / SELL SIGNAL
        markers.push({
          time: s.date,
          position: 'aboveBar',
          color: '#f43f5e',
          shape: 'arrowDown',
          text: `SELL (${s.probability_up.toFixed(0)}%)`
        });
        activePosition = 'CASH';
      }
    });
  } else if (state.signalOverlayMode === 'all') {
    // Show discrete clean arrows for all days
    sorted.forEach(s => {
      const isBuy = s.signal === 'BUY';
      markers.push({
        time: s.date,
        position: isBuy ? 'belowBar' : 'aboveBar',
        color: isBuy ? '#10b981' : '#f43f5e',
        shape: isBuy ? 'arrowUp' : 'arrowDown',
        text: `${s.signal}`
      });
    });
  }

  state.candleSeries.setMarkers(markers);
}

async function renderTradingViewChart() {
  if (!state.tvChart) {
    initTradingViewChart();
  }

  try {
    const [historyRes, signalsRes] = await Promise.all([
      api.getStockHistory(state.selectedTicker, state.selectedTimeframe),
      api.getSignalsOverlay(state.selectedTicker, state.selectedModel, state.selectedTimeframe)
    ]);

    state.rawHistoryData = historyRes.data || [];
    state.rawSignalsData = signalsRes.data || [];

    if (state.rawHistoryData.length === 0) {
      return;
    }

    // Prepare Candlestick data for Lightweight Charts (format: { time: 'YYYY-MM-DD', open, high, low, close })
    const candleData = state.rawHistoryData.map(item => ({
      time: item.date,
      open: item.open,
      high: item.high,
      low: item.low,
      close: item.close
    }));

    // Prepare Volume data (format: { time: 'YYYY-MM-DD', value, color })
    const volumeData = state.rawHistoryData.map(item => ({
      time: item.date,
      value: item.volume,
      color: item.close >= item.open ? 'rgba(16, 185, 129, 0.35)' : 'rgba(244, 63, 94, 0.35)'
    }));

    state.candleSeries.setData(candleData);
    state.volumeSeries.setData(volumeData);

    // Apply Smart Clean Markers
    applySignalsMarkers(state.rawSignalsData);

    // Adjust chart fit
    state.tvChart.timeScale().fitContent();

    // Update legend with latest candle
    updateDynamicLegend(null);
  } catch (err) {
    console.error('Error rendering Lightweight Chart:', err);
    showToast(`Gagal memuat grafik: ${err.message}`, 'error');
  }
}

function toggleFullscreen() {
  const panel = elements.chartPanel;
  state.isFullscreen = !state.isFullscreen;
  panel.classList.toggle('fullscreen-active', state.isFullscreen);

  if (state.isFullscreen) {
    elements.btnFullscreen.textContent = '✕ Tutup Layar';
    elements.btnFullscreen.classList.add('active');
  } else {
    elements.btnFullscreen.textContent = '⛶ Fullscreen';
    elements.btnFullscreen.classList.remove('active');
  }

  // Force chart to resize cleanly to new bounds
  handleChartResize();
  requestAnimationFrame(() => {
    handleChartResize();
    if (state.tvChart) {
      state.tvChart.timeScale().fitContent();
    }
  });
}

// Global Escape Key to exit fullscreen cleanly
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && state.isFullscreen) {
    toggleFullscreen();
  }
});

// ===================================================================
// OTHER DASHBOARD COMPONENTS (TICKERS, SIDEBAR, LEADERBOARD)
// ===================================================================

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

    elements.sidebarRecommendationText.textContent = pred.recommendation || 'Model memantau pergerakan harga pasar.';
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
  await renderTradingViewChart();
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
    await renderTradingViewChart();
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
    await renderTradingViewChart();
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

  // Timeframe buttons (1M, 3M, 6M, 1Y, ALL)
  elements.timeframeGroup.querySelectorAll('.tf-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      elements.timeframeGroup.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.selectedTimeframe = btn.getAttribute('data-tf');
      await renderTradingViewChart();
    });
  });

  // Signal Overlay Mode Dropdown (Entries / All / Off)
  elements.signalOverlayMode.addEventListener('change', (e) => {
    state.signalOverlayMode = e.target.value;
    applySignalsMarkers(state.rawSignalsData);
  });

  // Fullscreen button
  elements.btnFullscreen.addEventListener('click', () => {
    toggleFullscreen();
  });

  // Reset Zoom button
  elements.btnResetZoom.addEventListener('click', () => {
    if (state.tvChart) {
      state.tvChart.timeScale().fitContent();
    }
  });

  // Modal open / close
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
