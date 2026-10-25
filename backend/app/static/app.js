/**
 * AlphaQuant - A股全栈量化策略研究与回测看板交互逻辑
 * 支持：A股真实数据回测、月度收益热力矩阵、参数网格寻优与2D敏感性热力图
 */

// 全局状态管理
const state = {
  strategies: [],
  currentStrategyId: null,
  backtestResult: null,
  optimizationResult: null,
  activeChartTab: 'equity',
  chartInstance: null,
  cachedKline: null
};

// DOM 元素引用
const els = {
  strategySelect: document.getElementById('strategy-select'),
  strategyDesc: document.getElementById('strategy-desc'),
  paramsContainer: document.getElementById('dynamic-params-container'),
  inputSymbol: document.getElementById('input-symbol'),
  inputStartDate: document.getElementById('input-start-date'),
  inputEndDate: document.getElementById('input-end-date'),
  inputCash: document.getElementById('input-cash'),
  inputCommission: document.getElementById('input-commission'),
  inputStampTax: document.getElementById('input-stamp-tax'),
  inputMinCommission: document.getElementById('input-min-commission'),
  inputSlippage: document.getElementById('input-slippage'),
  checkT1: document.getElementById('check-t1'),
  checkLimits: document.getElementById('check-limits'),
  runBtn: document.getElementById('run-btn'),
  optimizeBtn: document.getElementById('optimize-btn'),
  selectTargetMetric: document.getElementById('select-target-metric'),
  backtestForm: document.getElementById('backtest-form'),
  toggleCostBtn: document.getElementById('toggle-cost-btn'),
  costPanel: document.getElementById('cost-panel'),

  mainChart: document.getElementById('main-chart'),
  monthlyMatrixContainer: document.getElementById('monthly-matrix-container'),
  matrixTbody: document.getElementById('matrix-tbody'),
  chartLegendBox: document.getElementById('chart-legend-box'),
  tabOptHeatmap: document.getElementById('tab-opt-heatmap'),
  tabPaneOpt: document.getElementById('tab-pane-opt'),

  tradesCount: document.getElementById('trades-count'),
  tradesTableBody: document.getElementById('trades-table-body'),
  optTableBody: document.getElementById('opt-table-body'),
  logsContainer: document.getElementById('logs-container'),

  // 展示与指标
  displayTarget: document.getElementById('display-target'),
  displayStrategy: document.getElementById('display-strategy'),
  displayRange: document.getElementById('display-range'),
  dispInitialCash: document.getElementById('disp-initial-cash'),
  dispFinalAsset: document.getElementById('disp-final-asset'),

  mTotalReturn: document.getElementById('m-total-return'),
  mExcessBadge: document.getElementById('m-excess-badge'),
  mBmkReturn: document.getElementById('m-bmk-return'),
  mAnnualReturn: document.getElementById('m-annual-return'),
  mBmkAnnual: document.getElementById('m-bmk-annual'),
  mSharpe: document.getElementById('m-sharpe'),
  mSortino: document.getElementById('m-sortino'),
  mMaxDd: document.getElementById('m-max-dd'),
  mCalmar: document.getElementById('m-calmar'),
  mWinRate: document.getElementById('m-win-rate'),
  mProfitLoss: document.getElementById('m-profit-loss'),
  mTotalTrades: document.getElementById('m-total-trades'),
  mWinLossCount: document.getElementById('m-win-loss-count')
};

// 页面初始化
document.addEventListener('DOMContentLoaded', async () => {
  initChart();
  bindEvents();
  initViewSwitcher();
  initSandbox();
  initFactorResearch();
  await loadStrategies();
  // 首次进入自动跑一次默认回测展示给用户
  triggerBacktest();
});

// 初始化 ECharts
function initChart() {
  if (els.mainChart) {
    state.chartInstance = echarts.init(els.mainChart);
    window.addEventListener('resize', () => {
      state.chartInstance && state.chartInstance.resize();
    });
  }
}

// 绑定交互事件
function bindEvents() {
  // 费率面板折叠展开
  els.toggleCostBtn.addEventListener('click', () => {
    els.costPanel.classList.toggle('open');
    const icon = els.toggleCostBtn.querySelector('.toggle-icon');
    icon.textContent = els.costPanel.classList.contains('open') ? '▲' : '▼';
  });

  // 标的速选按键
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (btn.dataset.strategy) {
        els.strategySelect.value = btn.dataset.strategy;
        selectStrategy(btn.dataset.strategy);
      }
      els.inputSymbol.value = btn.dataset.symbol;
      triggerBacktest();
    });
  });

  // 日期快捷区间按钮 (近1年 / 近2年 / 近3年)
  document.querySelectorAll('.chip-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const days = parseInt(btn.dataset.days, 10);
      const end = new Date();
      const start = new Date();
      start.setDate(end.getDate() - days);

      els.inputEndDate.value = end.toISOString().split('T')[0];
      els.inputStartDate.value = start.toISOString().split('T')[0];
    });
  });

  // 策略切换
  els.strategySelect.addEventListener('change', (e) => {
    selectStrategy(e.target.value);
  });

  // 回测表单提交
  els.backtestForm.addEventListener('submit', (e) => {
    e.preventDefault();
    triggerBacktest();
  });

  // 启动参数网格寻优
  if (els.optimizeBtn) {
    els.optimizeBtn.addEventListener('click', () => {
      triggerOptimization();
    });
  }

  // 图表 Tab 切换
  document.querySelectorAll('.chart-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.activeChartTab = tab.dataset.tab;
      renderActiveChart();
    });
  });

  // 明细面板 Tab 切换
  document.querySelectorAll('.details-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.details-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const paneId = tab.dataset.pane;
      document.querySelectorAll('.details-pane').forEach(p => p.classList.remove('active'));
      document.getElementById(`pane-${paneId}`).classList.add('active');
    });
  });
}

// 加载可用策略列表
async function loadStrategies() {
  try {
    const res = await fetch('/api/strategies');
    const data = await res.json();
    state.strategies = data.strategies || [];

    els.strategySelect.innerHTML = '';
    state.strategies.forEach(s => {
      const opt = document.createElement('option');
      opt.value = s.id;
      opt.textContent = `${s.name} (${s.id})`;
      els.strategySelect.appendChild(opt);
    });

    if (state.strategies.length > 0) {
      selectStrategy(state.strategies[0].id);
    }
  } catch (err) {
    console.error('获取策略列表失败:', err);
  }
}

// 选中某个策略，动态渲染其参数表单
function selectStrategy(strategyId) {
  state.currentStrategyId = strategyId;
  const current = state.strategies.find(s => s.id === strategyId);
  if (!current) return;

  // 根据策略是否为多标的组合，切换相关 UI 与默认标的池
  const tabAlloc = document.getElementById('tab-allocation');
  if (current.is_portfolio) {
    if (tabAlloc) tabAlloc.style.display = 'inline-block';
    if (current.default_pool && (!els.inputSymbol.value || !els.inputSymbol.value.includes(','))) {
      els.inputSymbol.value = current.default_pool.join(', ');
    }
  } else {
    if (tabAlloc) tabAlloc.style.display = 'none';
    if (state.activeChartTab === 'allocation') {
      state.activeChartTab = 'equity';
      document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
      document.querySelector('.chart-tab[data-tab="equity"]').classList.add('active');
    }
  }

  els.strategyDesc.textContent = current.description;
  els.paramsContainer.innerHTML = '';

  current.param_schema.forEach(field => {
    const item = document.createElement('div');
    item.className = 'param-item';

    const labelRow = document.createElement('div');
    labelRow.className = 'param-label-row';
    labelRow.innerHTML = `<span>${field.label}</span><span id="val-${field.key}">${field.default}</span>`;

    const input = document.createElement('input');
    input.type = 'number';
    input.className = 'form-control dynamic-param-input';
    input.dataset.key = field.key;
    input.value = field.default;
    if (field.min !== undefined) input.min = field.min;
    if (field.max !== undefined) input.max = field.max;
    if (field.type === 'float') input.step = '0.05';

    input.addEventListener('input', () => {
      document.getElementById(`val-${field.key}`).textContent = input.value;
    });

    item.appendChild(labelRow);
    item.appendChild(input);
    els.paramsContainer.appendChild(item);
  });
}

// 收集策略动态参数
function getDynamicParams() {
  const params = {};
  const inputs = els.paramsContainer.querySelectorAll('.dynamic-param-input');
  inputs.forEach(input => {
    const key = input.dataset.key;
    params[key] = parseFloat(input.value);
  });
  return params;
}

// 触发运行回测
async function triggerBacktest() {
  const symbol = els.inputSymbol.value.trim();
  if (!symbol) {
    alert('请输入股票或ETF代码');
    return;
  }

  els.runBtn.classList.add('loading');
  els.runBtn.disabled = true;

  const payload = {
    symbol: symbol,
    start_date: els.inputStartDate.value,
    end_date: els.inputEndDate.value,
    strategy_name: state.currentStrategyId || 'DualMAStrategy',
    strategy_params: getDynamicParams(),
    initial_cash: parseFloat(els.inputCash.value) || 100000,
    commission_rate: parseFloat(els.inputCommission.value) || 0.00025,
    min_commission: parseFloat(els.inputMinCommission.value) || 5.0,
    stamp_tax_rate: parseFloat(els.inputStampTax.value) || 0.0005,
    transfer_fee_rate: 0.00001,
    slippage_rate: parseFloat(els.inputSlippage.value) || 0.0005,
    enable_t_plus_one: els.checkT1.checked,
    enforce_price_limits: els.checkLimits.checked,
    benchmark_symbol: '000300'
  };

  try {
    const res = await fetch('/api/backtest/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const resData = await res.json();
    if (res.ok && resData.code === 200) {
      state.backtestResult = resData.data;
      updateUIWithResults(resData.data);
    } else {
      alert(`回测失败: ${resData.detail || '未知错误'}`);
    }
  } catch (err) {
    console.error('回测请求异常:', err);
    alert('网络或服务端异常，请检查后端运行状态');
  } finally {
    els.runBtn.classList.remove('loading');
    els.runBtn.disabled = false;
  }
}

// 刷新看板 UI
function updateUIWithResults(data) {
  // 标头信息更新
  const currentStrat = state.strategies.find(s => s.id === state.currentStrategyId);
  els.displayTarget.textContent = `${data.symbol} A股回测`;
  els.displayStrategy.textContent = currentStrat ? currentStrat.name : data.strategy_name;
  els.displayRange.textContent = `${els.inputStartDate.value} 至 ${els.inputEndDate.value}`;
  els.dispInitialCash.textContent = `¥${data.initial_cash.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
  els.dispFinalAsset.textContent = `¥${data.final_asset.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;

  // 指标卡片
  renderMetrics(data);

  // 渲染活动图表
  renderActiveChart();

  // 渲染交易明细
  renderTradesTable(data.trades || []);

  // 渲染引擎日志
  renderLogs(data.logs || []);
}

// 渲染6大指标卡片
function renderMetrics(data) {
  // 累计收益与超额
  const totalRet = data.total_return;
  const bmkRet = data.benchmark_return;
  const excess = totalRet - bmkRet;

  els.mTotalReturn.textContent = `${totalRet > 0 ? '+' : ''}${totalRet.toFixed(2)}%`;
  els.mTotalReturn.className = `metric-value ${totalRet >= 0 ? 'text-green' : 'text-red'}`;
  els.mExcessBadge.textContent = `超额 ${excess > 0 ? '+' : ''}${excess.toFixed(2)}%`;
  els.mExcessBadge.className = `metric-badge ${excess >= 0 ? 'text-green' : 'text-red'}`;
  els.mBmkReturn.textContent = `${bmkRet > 0 ? '+' : ''}${bmkRet.toFixed(2)}%`;

  // 年化收益
  els.mAnnualReturn.textContent = `${data.annualized_return > 0 ? '+' : ''}${data.annualized_return.toFixed(2)}%`;
  els.mAnnualReturn.className = `metric-value ${data.annualized_return >= 0 ? 'text-green' : 'text-red'}`;
  els.mBmkAnnual.textContent = `${data.benchmark_annual_return > 0 ? '+' : ''}${data.benchmark_annual_return.toFixed(2)}%`;

  // 夏普与索提诺
  els.mSharpe.textContent = data.sharpe_ratio.toFixed(2);
  els.mSortino.textContent = data.sortino_ratio.toFixed(2);

  // 最大回撤
  els.mMaxDd.textContent = `${data.max_drawdown.toFixed(2)}%`;
  els.mCalmar.textContent = data.calmar_ratio.toFixed(2);

  // 胜率与盈亏比
  els.mWinRate.textContent = `${data.win_rate.toFixed(1)}%`;
  els.mProfitLoss.textContent = data.profit_loss_ratio > 900 ? '极高' : data.profit_loss_ratio.toFixed(2);

  // 交易笔数
  els.mTotalTrades.textContent = data.total_trades;
  els.mWinLossCount.textContent = `${data.winning_trades} 胜 / ${data.losing_trades} 负`;
}

// 根据当前选中的 Tab 绘制图表或月度矩阵
async function renderActiveChart() {
  if (!state.chartInstance) return;

  // 1. 月度收益热力表 Tab
  if (state.activeChartTab === 'monthly') {
    els.mainChart.style.display = 'none';
    els.monthlyMatrixContainer.style.display = 'block';
    els.chartLegendBox.style.display = 'none';
    if (state.backtestResult && state.backtestResult.monthly_returns) {
      renderMonthlyMatrix(state.backtestResult.monthly_returns);
    }
    return;
  }

  // 2. 参数敏感性热力图 Tab
  if (state.activeChartTab === 'heatmap') {
    els.mainChart.style.display = 'block';
    els.monthlyMatrixContainer.style.display = 'none';
    els.chartLegendBox.style.display = 'none';
    state.chartInstance.resize();
    renderOptimizationHeatmap();
    return;
  }

  // 3. 资产配置变迁堆叠图 Tab
  if (state.activeChartTab === 'allocation') {
    els.mainChart.style.display = 'block';
    els.monthlyMatrixContainer.style.display = 'none';
    els.chartLegendBox.style.display = 'none';
    state.chartInstance.resize();
    renderAllocationStreamChart(state.backtestResult ? state.backtestResult.allocation_history || [] : []);
    return;
  }

  // 4. 常规时序图表 (净值曲线、K线蜡烛、水下回撤)
  els.mainChart.style.display = 'block';
  els.monthlyMatrixContainer.style.display = 'none';
  els.chartLegendBox.style.display = (state.activeChartTab === 'equity' ? 'flex' : 'none');
  state.chartInstance.resize();

  if (!state.backtestResult) return;
  const data = state.backtestResult;
  state.chartInstance.clear();

  if (state.activeChartTab === 'equity') {
    renderEquityChart(data.equity_curve || [], data.benchmark_curve || []);
  } else if (state.activeChartTab === 'kline') {
    await renderKlineChart(data.symbol, data.trades || []);
  } else if (state.activeChartTab === 'drawdown') {
    renderDrawdownChart(data.drawdown_curve || []);
  }
}

// 绘制资产配置时序堆叠流图 (展示多资产组合在不同历史阶段持有的 ETF/股票比例)
function renderAllocationStreamChart(allocationHistory) {
  if (!allocationHistory || allocationHistory.length === 0) {
    state.chartInstance.setOption({
      title: {
        text: '当前策略无资产配置时序数据，请在左侧选择【全天候ETF轮动策略】运行回测',
        left: 'center',
        top: 'middle',
        textStyle: { color: '#64748b', fontSize: 13 }
      }
    });
    return;
  }

  const dates = allocationHistory.map(a => a.date);

  const allSymbols = new Set();
  allocationHistory.forEach(a => {
    Object.keys(a.weights || {}).forEach(k => allSymbols.add(k));
  });
  const symbolList = Array.from(allSymbols);

  const nameMap = {
    'sh.510300': '沪深300ETF (510300)',
    'sh.510500': '中证500ETF (510500)',
    'sh.513100': '纳斯达克ETF (513100)',
    'sh.518880': '黄金ETF (518880)',
    'sh.511010': '国债ETF (511010 避险)',
    'sh.600519': '贵州茅台',
    'sz.000858': '五粮液',
    'sz.300750': '宁德时代',
    'sz.002594': '比亚迪',
    'sh.601318': '中国平安',
    'sh.600036': '招商银行',
    'sz.000333': '美的集团',
    'sh.600900': '长江电力',
    'sh.601899': '紫金矿业',
    'sz.300059': '东方财富',
    'CASH': '现金储备 (CASH)'
  };

  const series = symbolList.map(sym => {
    const data = allocationHistory.map(a => {
      const w = (a.weights && a.weights[sym] !== undefined) ? a.weights[sym] : 0.0;
      return +(w * 100).toFixed(1);
    });

    return {
      name: nameMap[sym] || sym,
      type: 'line',
      stack: 'Total',
      smooth: true,
      lineStyle: { width: 0.8 },
      showSymbol: false,
      areaStyle: { opacity: 0.82 },
      emphasis: { focus: 'series' },
      data: data
    };
  });

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross', label: { backgroundColor: '#1e293b' } },
      backgroundColor: 'rgba(17, 22, 34, 0.95)',
      borderColor: 'rgba(255, 255, 255, 0.1)',
      textStyle: { color: '#f8fafc', fontSize: 12 },
      formatter: function (params) {
        let tip = `<div style="font-weight:600;margin-bottom:6px;">${params[0].axisValue} 资产配置结构</div>`;
        params.forEach(p => {
          if (p.value > 0) {
            tip += `<div style="display:flex;justify-content:space-between;gap:12px;"><span>${p.marker} ${p.seriesName}</span><b>${p.value}%</b></div>`;
          }
        });
        return tip;
      }
    },
    legend: {
      data: series.map(s => s.name),
      textStyle: { color: '#94a3b8', fontSize: 11 },
      top: 5
    },
    grid: {
      left: '3%',
      right: '4%',
      top: '18%',
      bottom: '15%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: dates,
      axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.1)' } },
      axisLabel: { color: '#64748b', fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      max: 100,
      axisLabel: { color: '#64748b', fontSize: 11, formatter: '{value}%' },
      splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } }
    },
    dataZoom: [
      { type: 'slider', show: true, bottom: 5, height: 20, fillerColor: 'rgba(99, 102, 241, 0.15)' },
      { type: 'inside' }
    ],
    series: series
  };

  state.chartInstance.setOption(option);
}

// 1. 绘制净值与基准对比曲线
function renderEquityChart(equityCurve, benchmarkCurve) {
  const dates = equityCurve.map(item => item.date);
  const stratValues = equityCurve.map(item => ((item.value - 1.0) * 100).toFixed(2));
  const bmkValues = benchmarkCurve.map(item => ((item.value - 1.0) * 100).toFixed(2));

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(17, 22, 34, 0.95)',
      borderColor: 'rgba(255, 255, 255, 0.1)',
      textStyle: { color: '#f8fafc', fontSize: 12 },
      formatter: function (params) {
        let tip = `<div style="font-weight:600;margin-bottom:4px;">${params[0].axisValue}</div>`;
        params.forEach(p => {
          const colorDot = `<span style="display:inline-block;margin-right:5px;border-radius:10px;width:9px;height:9px;background-color:${p.color};"></span>`;
          tip += `<div>${colorDot} ${p.seriesName}: <b>${p.value}%</b></div>`;
        });
        return tip;
      }
    },
    grid: {
      left: '3%',
      right: '3%',
      top: '12%',
      bottom: '15%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.1)' } },
      axisLabel: { color: '#64748b', fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      axisLabel: {
        color: '#64748b',
        fontSize: 11,
        formatter: '{value}%'
      },
      splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } }
    },
    dataZoom: [
      {
        type: 'slider',
        show: true,
        bottom: 5,
        height: 20,
        borderColor: 'transparent',
        fillerColor: 'rgba(99, 102, 241, 0.15)',
        textStyle: { color: '#64748b' }
      },
      { type: 'inside' }
    ],
    series: [
      {
        name: '策略累计收益',
        type: 'line',
        data: stratValues,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#6366f1', width: 2.5 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(99, 102, 241, 0.35)' },
            { offset: 1, color: 'rgba(99, 102, 241, 0.0)' }
          ])
        }
      },
      {
        name: '沪深300基准',
        type: 'line',
        data: bmkValues,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#94a3b8', width: 1.5, type: 'dashed' }
      }
    ]
  };

  state.chartInstance.setOption(option);
}

// 2. 绘制标的 K 线与买卖点标记
async function renderKlineChart(symbol, trades) {
  let klineData = state.cachedKline;
  if (!klineData || klineData.symbol !== symbol) {
    try {
      const res = await fetch(`/api/market/kline?symbol=${symbol}&start_date=${els.inputStartDate.value}&end_date=${els.inputEndDate.value}`);
      const json = await res.json();
      klineData = json.data;
      state.cachedKline = { symbol, data: klineData };
    } catch (e) {
      console.error('获取K线数据失败', e);
      return;
    }
  }

  const dates = klineData.map(d => d.date);
  const oclh = klineData.map(d => [d.open, d.close, d.low, d.high]);

  const markData = [];
  trades.forEach(t => {
    const isBuy = t.action === 'BUY';
    markData.push({
      name: isBuy ? '买入' : '卖出',
      coord: [t.timestamp, t.price],
      value: isBuy ? `买 ¥${t.price}` : `卖 ¥${t.price}`,
      symbol: isBuy ? 'triangle' : 'pin',
      symbolRotate: isBuy ? 0 : 180,
      symbolSize: 22,
      itemStyle: {
        color: isBuy ? '#10b981' : '#f43f5e'
      }
    });
  });

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' },
      backgroundColor: 'rgba(17, 22, 34, 0.95)',
      borderColor: 'rgba(255, 255, 255, 0.1)',
      textStyle: { color: '#f8fafc', fontSize: 12 }
    },
    grid: { left: '3%', right: '3%', top: '10%', bottom: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.1)' } },
      axisLabel: { color: '#64748b', fontSize: 11 }
    },
    yAxis: {
      scale: true,
      splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } },
      axisLabel: { color: '#64748b', fontSize: 11 }
    },
    dataZoom: [
      { type: 'slider', show: true, bottom: 5, height: 20, fillerColor: 'rgba(99, 102, 241, 0.15)' },
      { type: 'inside' }
    ],
    series: [
      {
        name: '标的K线',
        type: 'candlestick',
        data: oclh,
        itemStyle: {
          color: '#ef4444',
          color0: '#10b981',
          borderColor: '#ef4444',
          borderColor0: '#10b981'
        },
        markPoint: {
          data: markData
        }
      }
    ]
  };

  state.chartInstance.setOption(option);
}

// 3. 绘制水下动态回撤曲线
function renderDrawdownChart(drawdownCurve) {
  const dates = drawdownCurve.map(item => item.date);
  const values = drawdownCurve.map(item => item.value);

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(17, 22, 34, 0.95)',
      borderColor: 'rgba(255, 255, 255, 0.1)',
      textStyle: { color: '#f8fafc', fontSize: 12 },
      formatter: '{b}<br/>动态回撤: <b>{c}%</b>'
    },
    grid: { left: '3%', right: '3%', top: '10%', bottom: '15%', containLabel: true },
    xAxis: {
      type: 'category',
      data: dates,
      axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.1)' } },
      axisLabel: { color: '#64748b', fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      max: 0,
      axisLabel: { color: '#64748b', fontSize: 11, formatter: '{value}%' },
      splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } }
    },
    dataZoom: [
      { type: 'slider', show: true, bottom: 5, height: 20, fillerColor: 'rgba(244, 63, 94, 0.15)' },
      { type: 'inside' }
    ],
    series: [
      {
        name: '动态回撤',
        type: 'line',
        data: values,
        smooth: true,
        showSymbol: false,
        lineStyle: { color: '#f43f5e', width: 2 },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(244, 63, 94, 0.0)' },
            { offset: 1, color: 'rgba(244, 63, 94, 0.4)' }
          ])
        }
      }
    ]
  };

  state.chartInstance.setOption(option);
}

// 4. 渲染月度收益热力矩阵
function renderMonthlyMatrix(monthlyData) {
  if (!monthlyData || !monthlyData.years || monthlyData.years.length === 0) {
    els.matrixTbody.innerHTML = `<tr><td colspan="14" class="empty-cell">暂无月度收益数据</td></tr>`;
    return;
  }

  els.matrixTbody.innerHTML = '';
  const sortedYears = [...monthlyData.years].reverse();

  sortedYears.forEach(year => {
    const yData = monthlyData.data[year] || {};
    const tr = document.createElement('tr');

    let html = `<td class="matrix-year-cell">${year}</td>`;

    // 1~12 月
    for (let m = 1; m <= 12; m++) {
      const val = yData[String(m)];
      if (val !== null && val !== undefined) {
        const isPos = val >= 0;
        const colorClass = isPos ? 'cell-pos' : 'cell-neg';
        // A股红涨绿跌，计算背景饱和度
        const opacity = Math.min(Math.abs(val) / 8.0 * 0.45 + 0.08, 0.55).toFixed(2);
        const bg = isPos ? `rgba(239, 68, 68, ${opacity})` : `rgba(16, 185, 129, ${opacity})`;
        const sign = val > 0 ? '+' : '';
        html += `<td class="${colorClass}" style="background-color: ${bg};" title="${year}年${m}月: ${sign}${val.toFixed(2)}%">${sign}${val.toFixed(1)}%</td>`;
      } else {
        html += `<td style="color: #475569;">-</td>`;
      }
    }

    // 全年累计
    const totalVal = yData['total'];
    if (totalVal !== null && totalVal !== undefined) {
      const isPos = totalVal >= 0;
      const colorClass = isPos ? 'cell-pos' : 'cell-neg';
      const opacity = Math.min(Math.abs(totalVal) / 20.0 * 0.5 + 0.1, 0.65).toFixed(2);
      const bg = isPos ? `rgba(239, 68, 68, ${opacity})` : `rgba(16, 185, 129, ${opacity})`;
      const sign = totalVal > 0 ? '+' : '';
      html += `<td class="matrix-total-cell ${colorClass}" style="background-color: ${bg};" title="${year}全年: ${sign}${totalVal.toFixed(2)}%">${sign}${totalVal.toFixed(2)}%</td>`;
    } else {
      html += `<td class="matrix-total-cell" style="color: #475569;">-</td>`;
    }

    tr.innerHTML = html;
    els.matrixTbody.appendChild(tr);
  });
}

// 5. 触发参数网格寻优
async function triggerOptimization() {
  const symbol = els.inputSymbol.value.trim();
  if (!symbol) {
    alert('请输入股票或ETF代码');
    return;
  }

  els.optimizeBtn.classList.add('loading');
  els.optimizeBtn.disabled = true;

  const targetMetric = els.selectTargetMetric.value;

  const payload = {
    symbol: symbol,
    start_date: els.inputStartDate.value,
    end_date: els.inputEndDate.value,
    strategy_name: state.currentStrategyId || 'DualMAStrategy',
    target_metric: targetMetric,
    initial_cash: parseFloat(els.inputCash.value) || 100000,
    commission_rate: parseFloat(els.inputCommission.value) || 0.00025,
    min_commission: parseFloat(els.inputMinCommission.value) || 5.0,
    stamp_tax_rate: parseFloat(els.inputStampTax.value) || 0.0005,
    transfer_fee_rate: 0.00001,
    slippage_rate: parseFloat(els.inputSlippage.value) || 0.0005,
    enable_t_plus_one: els.checkT1.checked,
    enforce_price_limits: els.checkLimits.checked,
    benchmark_symbol: '000300'
  };

  try {
    const res = await fetch('/api/backtest/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const resData = await res.json();
    if (res.ok && resData.code === 200) {
      state.optimizationResult = resData.data;

      // 1. 自动切换到“参数敏感性热力图” Tab
      document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
      els.tabOptHeatmap.classList.add('active');
      state.activeChartTab = 'heatmap';
      renderActiveChart();

      // 2. 自动切换到底部“参数寻优榜单” Tab
      document.querySelectorAll('.details-tab').forEach(t => t.classList.remove('active'));
      els.tabPaneOpt.classList.add('active');
      document.querySelectorAll('.details-pane').forEach(p => p.classList.remove('active'));
      document.getElementById('pane-opt').classList.add('active');

      // 3. 渲染寻优榜单表格
      renderOptimizationTable(resData.data.ranked_results || [], targetMetric);
    } else {
      alert(`网格寻优失败: ${resData.detail || '未知错误'}`);
    }
  } catch (err) {
    console.error('寻优请求异常:', err);
    alert('网格寻优服务异常');
  } finally {
    els.optimizeBtn.classList.remove('loading');
    els.optimizeBtn.disabled = false;
  }
}

// 6. 绘制参数敏感性 2D 热力图
function renderOptimizationHeatmap() {
  if (!state.optimizationResult || !state.optimizationResult.heatmap_2d) {
    state.chartInstance.setOption({
      title: {
        text: '当前参数寻优无2D热力图数据，请先点击【⚡ 启动参数网格寻优】',
        left: 'center',
        top: 'middle',
        textStyle: { color: '#64748b', fontSize: 13 }
      }
    });
    return;
  }

  const h = state.optimizationResult.heatmap_2d;
  const xLabels = h.x_labels;
  const yLabels = h.y_labels;
  const data = h.points; // [[x, y, val], ...]

  const validVals = data.map(p => p[2]).filter(v => v !== null && !isNaN(v));
  const minVal = validVals.length > 0 ? Math.min(...validVals) : 0;
  const maxVal = validVals.length > 0 ? Math.max(...validVals) : 1;

  const metricNameMap = {
    'sharpe_ratio': '夏普比率',
    'total_return': '累计收益率(%)',
    'calmar_ratio': '卡玛比率',
    'win_rate': '胜率(%)'
  };
  const metricLabel = metricNameMap[h.target_metric] || h.target_metric;

  const option = {
    backgroundColor: 'transparent',
    tooltip: {
      position: 'top',
      backgroundColor: 'rgba(17, 22, 34, 0.95)',
      borderColor: 'rgba(255, 255, 255, 0.1)',
      textStyle: { color: '#f8fafc', fontSize: 12 },
      formatter: function (p) {
        const xVal = xLabels[p.value[0]];
        const yVal = yLabels[p.value[1]];
        const score = p.value[2] !== null ? p.value[2] : '无效';
        return `<div>${h.param_x}: <b>${xVal}</b></div><div>${h.param_y}: <b>${yVal}</b></div><div>${metricLabel}: <b style="color:#38bdf8;">${score}</b></div>`;
      }
    },
    grid: {
      left: '8%',
      right: '12%',
      top: '12%',
      bottom: '15%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      data: xLabels,
      name: h.param_x,
      nameLocation: 'middle',
      nameGap: 28,
      nameTextStyle: { color: '#94a3b8', fontSize: 12 },
      splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.01)', 'rgba(255,255,255,0.03)'] } },
      axisLabel: { color: '#94a3b8' },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }
    },
    yAxis: {
      type: 'category',
      data: yLabels,
      name: h.param_y,
      nameLocation: 'middle',
      nameGap: 36,
      nameTextStyle: { color: '#94a3b8', fontSize: 12 },
      splitArea: { show: true, areaStyle: { color: ['rgba(255,255,255,0.01)', 'rgba(255,255,255,0.03)'] } },
      axisLabel: { color: '#94a3b8' },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }
    },
    visualMap: {
      min: minVal,
      max: maxVal,
      calculable: true,
      orient: 'vertical',
      right: '1%',
      top: 'middle',
      inRange: {
        color: ['#0f172a', '#1e3a8a', '#0284c7', '#10b981', '#f59e0b', '#ef4444']
      },
      textStyle: { color: '#94a3b8', fontSize: 11 },
      formatter: function (v) { return v.toFixed(1); }
    },
    series: [
      {
        name: metricLabel,
        type: 'heatmap',
        data: data,
        label: {
          show: true,
          color: '#ffffff',
          fontSize: 11,
          formatter: function (p) {
            return p.value[2] !== null ? p.value[2] : '-';
          }
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ]
  };

  state.chartInstance.setOption(option);
}

// 7. 渲染寻优榜单表格
function renderOptimizationTable(rankedList, targetMetric) {
  if (!rankedList || rankedList.length === 0) {
    els.optTableBody.innerHTML = `<tr><td colspan="10" class="empty-cell">未找到有效的寻优结果</td></tr>`;
    return;
  }

  els.optTableBody.innerHTML = '';
  rankedList.forEach((item, idx) => {
    const tr = document.createElement('tr');

    // 格式化参数文本
    const paramStr = Object.entries(item.params).map(([k, v]) => `${k}=${v}`).join(', ');
    const isTop1 = idx === 0;

    tr.innerHTML = `
      <td><span style="font-weight:${isTop1 ? '700' : '500'};color:${isTop1 ? '#fbbf24' : 'inherit'};">#${idx + 1}</span></td>
      <td style="font-family:var(--font-mono);color:#a5b4fc;">${paramStr}</td>
      <td style="font-weight:600;color:var(--cyan);">${item.target_metric_value.toFixed(2)}</td>
      <td class="${item.total_return >= 0 ? 'text-green' : 'text-red'}">${item.total_return > 0 ? '+' : ''}${item.total_return.toFixed(2)}%</td>
      <td class="${item.annualized_return >= 0 ? 'text-green' : 'text-red'}">${item.annualized_return > 0 ? '+' : ''}${item.annualized_return.toFixed(2)}%</td>
      <td>${item.sharpe_ratio.toFixed(2)}</td>
      <td class="text-red">${item.max_drawdown.toFixed(2)}%</td>
      <td>${item.win_rate.toFixed(1)}%</td>
      <td>${item.total_trades}</td>
      <td><button class="opt-apply-btn" data-params='${JSON.stringify(item.params)}'>应用</button></td>
    `;
    els.optTableBody.appendChild(tr);
  });

  // 绑定“应用此参数”事件
  els.optTableBody.querySelectorAll('.opt-apply-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const p = JSON.parse(btn.dataset.params);
      applyOptimizedParams(p);
    });
  });
}

// 8. 将寻优结果参数一键回填至左侧表单并触发回测
function applyOptimizedParams(params) {
  Object.entries(params).forEach(([key, val]) => {
    const input = els.paramsContainer.querySelector(`.dynamic-param-input[data-key="${key}"]`);
    if (input) {
      input.value = val;
      const labelVal = document.getElementById(`val-${key}`);
      if (labelVal) labelVal.textContent = val;
    }
  });

  // 切回净值曲线 Tab
  document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
  document.querySelector('.chart-tab[data-tab="equity"]').classList.add('active');
  state.activeChartTab = 'equity';

  // 立即触发一次回测
  triggerBacktest();
}

// 渲染交易明细表格
function renderTradesTable(trades) {
  els.tradesCount.textContent = trades.length;
  if (!trades || trades.length === 0) {
    els.tradesTableBody.innerHTML = `<tr><td colspan="10" class="empty-cell">该回测区间无成交触发</td></tr>`;
    return;
  }

  els.tradesTableBody.innerHTML = '';
  trades.forEach(t => {
    const tr = document.createElement('tr');
    const isBuy = t.action === 'BUY';
    const pnlClass = t.realized_pnl > 0 ? 'text-green' : (t.realized_pnl < 0 ? 'text-red' : '');

    tr.innerHTML = `
      <td>${t.timestamp}</td>
      <td>${t.symbol}</td>
      <td><span class="tag-action ${isBuy ? 'buy' : 'sell'}">${isBuy ? '买入' : '卖出'}</span></td>
      <td>¥${t.price.toFixed(2)}</td>
      <td>${t.shares}股</td>
      <td>¥${t.commission.toFixed(2)}</td>
      <td>¥${t.stamp_tax.toFixed(2)}</td>
      <td>¥${t.total_cost.toFixed(2)}</td>
      <td class="${pnlClass}">${isBuy ? '--' : `¥${t.realized_pnl > 0 ? '+' : ''}${t.realized_pnl.toFixed(2)}`}</td>
      <td>¥${t.cash_after.toLocaleString('en-US', { minimumFractionDigits: 2 })}</td>
    `;
    els.tradesTableBody.appendChild(tr);
  });
}

// 渲染回测引擎日志
function renderLogs(logs) {
  if (!logs || logs.length === 0) {
    els.logsContainer.innerHTML = `<div class="log-line">暂无回测日志</div>`;
    return;
  }

  els.logsContainer.innerHTML = '';
  logs.forEach(msg => {
    const div = document.createElement('div');
    div.className = 'log-line';
    if (msg.includes('买入')) div.classList.add('buy');
    else if (msg.includes('卖出')) div.classList.add('sell');
    else if (msg.includes('失败') || msg.includes('跳过') || msg.includes('受阻')) div.classList.add('warn');

    div.textContent = msg;
    els.logsContainer.appendChild(div);
  });

  els.logsContainer.scrollTop = els.logsContainer.scrollHeight;
}

/* ==========================================================================
   Phase 2: 顶部多模块主功能视图切换 (View Switcher)
   ========================================================================== */
function initViewSwitcher() {
  const modeBtns = document.querySelectorAll('.nav-mode-btn');
  modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      modeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const targetId = btn.dataset.target;
      document.querySelectorAll('.app-view').forEach(view => {
        view.style.display = 'none';
        view.classList.remove('active');
      });

      const targetView = document.getElementById(targetId);
      if (targetView) {
        targetView.style.display = 'block';
        targetView.classList.add('active');
      }

      // 调整图表大小以防初次隐藏导致宽高为0
      setTimeout(() => {
        if (targetId === 'view-backtest' && state.chartInstance) {
          state.chartInstance.resize();
        } else if (targetId === 'view-sandbox' && sandboxState.chart) {
          sandboxState.chart.resize();
        } else if (targetId === 'view-factors' && factorState.chart) {
          factorState.chart.resize();
        }
      }, 50);
    });
  });
}

/* ==========================================================================
   Phase 2: 策略代码在线沙箱模块 (Strategy Online Sandbox)
   ========================================================================== */
const sandboxState = {
  templates: {},
  currentResult: null,
  activeTab: 'equity',
  chart: null
};

function initSandbox() {
  const templateSelect = document.getElementById('sandbox-template-select');
  const loadTemplateBtn = document.getElementById('sandbox-load-template-btn');
  const runBtn = document.getElementById('sandbox-run-btn');
  const codeEditor = document.getElementById('sandbox-code');
  const lineNumbers = document.getElementById('sandbox-line-numbers');
  const chartEl = document.getElementById('sandbox-chart');

  if (chartEl) {
    sandboxState.chart = echarts.init(chartEl);
    window.addEventListener('resize', () => {
      sandboxState.chart && sandboxState.chart.resize();
    });
  }

  // 1. 行号与滚动同步
  function updateLineNumbers() {
    if (!codeEditor || !lineNumbers) return;
    const lines = codeEditor.value.split('\n').length;
    lineNumbers.innerHTML = Array.from({ length: lines }, (_, i) => i + 1).join('\n');
    lineNumbers.scrollTop = codeEditor.scrollTop;
  }

  if (codeEditor) {
    codeEditor.addEventListener('input', updateLineNumbers);
    codeEditor.addEventListener('scroll', () => {
      if (lineNumbers) lineNumbers.scrollTop = codeEditor.scrollTop;
    });

    // 支持 TAB 键缩进 4 空格
    codeEditor.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const start = codeEditor.selectionStart;
        const end = codeEditor.selectionEnd;
        codeEditor.value = codeEditor.value.substring(0, start) + '    ' + codeEditor.value.substring(end);
        codeEditor.selectionStart = codeEditor.selectionEnd = start + 4;
        updateLineNumbers();
      }
    });
  }

  // 2. 加载模板列表
  fetch('/api/sandbox/templates')
    .then(res => res.json())
    .then(data => {
      if (data && data.templates) {
        sandboxState.templates = data.templates;
        templateSelect.innerHTML = '';
        Object.keys(data.templates).forEach(key => {
          const t = data.templates[key];
          const opt = document.createElement('option');
          opt.value = key;
          opt.textContent = `${t.name} (${key})`;
          templateSelect.appendChild(opt);
        });

        // 默认载入第一个模板
        const firstKey = Object.keys(data.templates)[0];
        if (firstKey) {
          codeEditor.value = data.templates[firstKey].code;
          updateLineNumbers();
        }
      }
    })
    .catch(err => console.error('加载沙箱模板失败:', err));

  // 载入模板按钮
  loadTemplateBtn && loadTemplateBtn.addEventListener('click', () => {
    const key = templateSelect.value;
    if (sandboxState.templates[key]) {
      codeEditor.value = sandboxState.templates[key].code;
      updateLineNumbers();
      logSandboxConsole(`已载入模板: ${sandboxState.templates[key].name}`, 'info');
    }
  });

  // 3. 沙箱回测图表 Tab 切换
  document.querySelectorAll('[data-sbtab]').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('[data-sbtab]').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      sandboxState.activeTab = tab.dataset.sbtab;
      renderSandboxChart();
    });
  });

  // 4. 运行沙箱回测
  runBtn && runBtn.addEventListener('click', async () => {
    const code = codeEditor.value;
    const symbol = document.getElementById('sandbox-symbol').value.trim() || '510300';
    const startDate = document.getElementById('sandbox-start-date').value || '2023-01-01';
    const endDate = document.getElementById('sandbox-end-date').value || '2024-01-01';
    const cash = parseFloat(document.getElementById('sandbox-cash').value) || 100000;

    runBtn.classList.add('loading');
    updateSandboxStatus('ready', '正在编译撮合...');
    logSandboxConsole(`[${new Date().toLocaleTimeString()}] 开始编译用户代码并启动 A 股沙箱撮合 (${symbol}, ${startDate} 至 ${endDate})...`, 'info');

    try {
      const resp = await fetch('/api/sandbox/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code: code,
          symbol: symbol,
          start_date: startDate,
          end_date: endDate,
          initial_cash: cash
        })
      });

      const res = await resp.json();
      runBtn.classList.remove('loading');

      if (!res.success) {
        updateSandboxStatus('error', res.error_type || '执行异常');
        logSandboxConsole(`❌ 沙箱报错: ${res.message}\n${res.detail || ''}`, 'error');
        return;
      }

      // 运行成功
      updateSandboxStatus('success', '回测成功');
      const metricsData = res.data;
      sandboxState.currentResult = metricsData;

      logSandboxConsole(`✅ 策略编译通过并完成回测！期末资产: ¥${metricsData.final_asset?.toFixed(2)}, 累计收益: ${metricsData.metrics?.total_return?.toFixed(2)}%, 交易笔数: ${metricsData.trades?.length || 0}`, 'success');

      // 更新看板
      renderSandboxDashboard(metricsData, symbol, startDate, endDate);
    } catch (err) {
      runBtn.classList.remove('loading');
      updateSandboxStatus('error', '网络异常');
      logSandboxConsole(`网络或服务器异常: ${err.message}`, 'error');
    }
  });
}

function updateSandboxStatus(status, text) {
  const badge = document.getElementById('sandbox-status-badge');
  const timeEl = document.getElementById('sandbox-diag-time');
  if (badge) {
    badge.className = `badge-status ${status}`;
    badge.textContent = text;
  }
  if (timeEl) {
    timeEl.textContent = new Date().toLocaleTimeString();
  }
}

function logSandboxConsole(msg, type = 'info') {
  const logBox = document.getElementById('sandbox-diagnostics-log');
  if (!logBox) return;
  const line = document.createElement('div');
  line.className = `console-line ${type}`;
  line.textContent = msg;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

function renderSandboxDashboard(data, symbol, startDate, endDate) {
  const m = data;
  document.getElementById('sb-disp-symbol').textContent = `${symbol}`;
  document.getElementById('sb-disp-name').textContent = data.strategy_name || '自定义策略';
  document.getElementById('sb-disp-range').textContent = `${startDate} 至 ${endDate}`;
  document.getElementById('sb-disp-final-asset').textContent = `¥${(data.final_asset || 0).toLocaleString('en-US', { minimumFractionDigits: 2 })}`;

  // 指标卡片
  const tot = m.total_return || 0;
  const bmk = m.benchmark_return || 0;
  const excess = tot - bmk;

  const totEl = document.getElementById('sb-m-total-return');
  totEl.textContent = `${tot > 0 ? '+' : ''}${tot.toFixed(2)}%`;
  totEl.className = `metric-value ${tot >= 0 ? 'text-green' : 'text-red'}`;

  const excessBadge = document.getElementById('sb-excess-badge');
  excessBadge.textContent = `超额 ${excess > 0 ? '+' : ''}${excess.toFixed(2)}%`;
  excessBadge.className = `metric-badge ${excess >= 0 ? 'positive' : 'danger'}`;

  document.getElementById('sb-m-bmk-return').textContent = `${bmk.toFixed(2)}%`;
  document.getElementById('sb-m-annual-return').textContent = `${(m.annualized_return || 0).toFixed(2)}%`;
  document.getElementById('sb-m-bmk-annual').textContent = `${(m.benchmark_annual_return || 0).toFixed(2)}%`;
  document.getElementById('sb-m-sharpe').textContent = (m.sharpe_ratio || 0).toFixed(2);
  document.getElementById('sb-m-sortino').textContent = (m.sortino_ratio || 0).toFixed(2);
  document.getElementById('sb-m-max-dd').textContent = `${(m.max_drawdown || 0).toFixed(2)}%`;
  document.getElementById('sb-m-calmar').textContent = (m.calmar_ratio || 0).toFixed(2);
  document.getElementById('sb-m-win-rate').textContent = `${(m.win_rate || 0).toFixed(1)}%`;
  document.getElementById('sb-m-profit-loss').textContent = (m.profit_loss_ratio || 0).toFixed(2);
  document.getElementById('sb-m-total-trades').textContent = m.total_trades || 0;
  document.getElementById('sb-m-win-loss').textContent = `${m.winning_trades || 0} / ${m.losing_trades || 0}`;

  // 图表渲染
  renderSandboxChart();

  // 成交记录
  renderSandboxTrades(data.trades || []);
}

function renderSandboxChart() {
  if (!sandboxState.chart || !sandboxState.currentResult) return;
  const data = sandboxState.currentResult;
  const eqCurve = data.equity_curve || [];
  const bmkCurve = data.benchmark_curve || [];
  const ddCurve = data.drawdown_curve || [];
  if (eqCurve.length === 0) return;

  const dates = eqCurve.map(d => d.date);

  if (sandboxState.activeTab === 'equity') {
    const stratNav = eqCurve.map(d => d.value);
    const bmkNav = bmkCurve.map(d => d.value || 1.0);

    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(17, 22, 34, 0.95)',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        textStyle: { color: '#f8fafc', fontSize: 12 },
        formatter: (params) => {
          let s = `<div style="font-weight:600;margin-bottom:4px;">${params[0].axisValue}</div>`;
          params.forEach(p => {
            const ret = ((p.value - 1) * 100).toFixed(2);
            s += `<div>${p.marker} ${p.seriesName}: <b>${p.value.toFixed(4)}</b> (${ret > 0 ? '+' : ''}${ret}%)</div>`;
          });
          return s;
        }
      },
      legend: { show: false },
      grid: { left: '3%', right: '4%', top: '10%', bottom: '8%', containLabel: true },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.15)' } },
        axisLabel: { color: '#64748b', fontSize: 11 }
      },
      yAxis: {
        type: 'value',
        scale: true,
        splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } },
        axisLabel: { color: '#64748b', fontSize: 11, formatter: v => v.toFixed(2) }
      },
      series: [
        {
          name: '沙箱净值',
          type: 'line',
          data: stratNav,
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 2.5, color: '#38bdf8' },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(56, 189, 248, 0.25)' },
              { offset: 1, color: 'rgba(56, 189, 248, 0.00)' }
            ])
          }
        },
        {
          name: '沪深300',
          type: 'line',
          data: bmkNav,
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 1.5, color: '#818cf8', type: 'dashed' }
        }
      ]
    };
    sandboxState.chart.setOption(option, true);
  } else {
    // 回撤曲线
    const drawdowns = ddCurve.map(d => d.value);
    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        formatter: (params) => `${params[0].axisValue}<br/>动态回撤: <b style="color:#f43f5e">${params[0].value}%</b>`
      },
      grid: { left: '3%', right: '4%', top: '10%', bottom: '8%', containLabel: true },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.15)' } },
        axisLabel: { color: '#64748b', fontSize: 11 }
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: '#64748b', fontSize: 11, formatter: v => `${v}%` },
        splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } }
      },
      series: [
        {
          name: '水下回撤',
          type: 'line',
          data: drawdowns,
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 2, color: '#f43f5e' },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(244, 63, 94, 0.05)' },
              { offset: 1, color: 'rgba(244, 63, 94, 0.35)' }
            ])
          }
        }
      ]
    };
    sandboxState.chart.setOption(option, true);
  }
}

function renderSandboxTrades(trades) {
  const tbody = document.getElementById('sb-trades-table-body');
  const countEl = document.getElementById('sb-trades-count');
  if (!tbody) return;

  countEl && (countEl.textContent = trades.length);
  if (!trades || trades.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="empty-cell">该沙箱回测无成交触发</td></tr>`;
    return;
  }

  tbody.innerHTML = '';
  trades.forEach(t => {
    const tr = document.createElement('tr');
    const isBuy = t.action === 'BUY';
    const pnlClass = t.realized_pnl > 0 ? 'text-green' : (t.realized_pnl < 0 ? 'text-red' : '');

    tr.innerHTML = `
      <td>${t.timestamp}</td>
      <td>${t.symbol}</td>
      <td><span class="tag-action ${isBuy ? 'buy' : 'sell'}">${isBuy ? '买入' : '卖出'}</span></td>
      <td>¥${t.price.toFixed(2)}</td>
      <td>${t.shares}股</td>
      <td>¥${t.commission.toFixed(2)}</td>
      <td>¥${t.stamp_tax.toFixed(2)}</td>
      <td>¥${t.total_cost.toFixed(2)}</td>
      <td class="${pnlClass}">${isBuy ? '--' : `¥${t.realized_pnl > 0 ? '+' : ''}${t.realized_pnl.toFixed(2)}`}</td>
      <td>¥${t.cash_after.toLocaleString('en-US', { minimumFractionDigits: 2 })}</td>
    `;
    tbody.appendChild(tr);
  });
}

/* ==========================================================================
   Phase 2: 可视化因子投研分析中心 (Factor Research & Analysis Center)
   ========================================================================== */
const factorState = {
  factors: [],
  currentResult: null,
  activeTab: 'nav',
  chart: null
};

function initFactorResearch() {
  const factorSelect = document.getElementById('factor-select');
  const symbolsInput = document.getElementById('factor-symbols-input');
  const analyzeBtn = document.getElementById('factor-analyze-btn');
  const chartEl = document.getElementById('factor-main-chart');

  if (chartEl) {
    factorState.chart = echarts.init(chartEl);
    window.addEventListener('resize', () => {
      factorState.chart && factorState.chart.resize();
    });
  }

  // 1. 加载因子列表
  fetch('/api/factors/available')
    .then(res => res.json())
    .then(data => {
      if (data && data.factors) {
        factorState.factors = data.factors;
        factorSelect.innerHTML = '';
        data.factors.forEach(f => {
          const opt = document.createElement('option');
          opt.value = f.id;
          opt.textContent = `${f.name} [${f.category}]`;
          factorSelect.appendChild(opt);
        });

        // 更新首个因子的元数据
        if (data.factors[0]) {
          updateFactorMeta(data.factors[0]);
        }
      }
    })
    .catch(err => console.error('获取因子列表失败:', err));

  factorSelect && factorSelect.addEventListener('change', () => {
    const selected = factorState.factors.find(f => f.id === factorSelect.value);
    if (selected) {
      updateFactorMeta(selected);
    }
  });

  function updateFactorMeta(f) {
    const catEl = document.getElementById('factor-category-tag');
    const formulaEl = document.getElementById('factor-formula-text');
    const descEl = document.getElementById('factor-desc-text');
    if (catEl) catEl.textContent = f.category;
    if (formulaEl) formulaEl.textContent = f.formula;
    if (descEl) descEl.textContent = f.description;
  }

  // 2. 标的池快捷按键
  const poolShortcuts = {
    etf: '510300,510500,513100,518880,511010',
    bluechip: '600519,000858,300750,002594,601318,600036,000333,600900,601899,300059',
    growth: '300750,002475,688981,002230,300033,688012'
  };

  document.querySelectorAll('[data-fpool]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-fpool]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const poolKey = btn.dataset.fpool;
      if (poolShortcuts[poolKey]) {
        symbolsInput.value = poolShortcuts[poolKey];
      }
    });
  });

  // 3. 图表 Tab 切换
  document.querySelectorAll('[data-factab]').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('[data-factab]').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      factorState.activeTab = tab.dataset.factab;
      renderFactorChart();
    });
  });

  // 4. 运行因子分析
  analyzeBtn && analyzeBtn.addEventListener('click', async () => {
    const factorName = factorSelect.value || 'momentum_20';
    const rawSymbols = symbolsInput.value.replace(/;/g, ',').replace(/\s+/g, ',').split(',');
    const symbols = rawSymbols.map(s => s.trim()).filter(s => s.length > 0);
    const horizonRadio = document.querySelector('input[name="forward-horizon"]:checked');
    const forwardPeriod = horizonRadio ? parseInt(horizonRadio.value) : 5;

    if (symbols.length < 2) {
      alert('因子截面分析需要至少提供 2 只标的！');
      return;
    }

    analyzeBtn.classList.add('loading');

    try {
      const resp = await fetch('/api/factors/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          factor_name: factorName,
          symbols: symbols,
          start_date: '2023-01-01',
          end_date: '2024-01-01',
          forward_period: forwardPeriod,
          quantiles: 5
        })
      });

      const res = await resp.json();
      analyzeBtn.classList.remove('loading');

      if (res.code !== 200) {
        alert(`因子分析异常: ${res.detail || res.message}`);
        return;
      }

      factorState.currentResult = res.data;
      renderFactorDashboard(res.data);
    } catch (err) {
      analyzeBtn.classList.remove('loading');
      alert(`网络或接口异常: ${err.message}`);
    }
  });
}

function renderFactorDashboard(data) {
  const m = data.metrics || {};

  // Rank-IC Mean
  const meanIc = m.ic_mean || 0;
  const icEl = document.getElementById('factor-m-ic-mean');
  icEl.textContent = meanIc.toFixed(4);
  icEl.className = `metric-value ${meanIc > 0 ? 'highlight-cyan' : 'text-red'}`;

  const icBadge = document.getElementById('factor-ic-badge');
  if (Math.abs(meanIc) >= 0.05) {
    icBadge.textContent = '显著有效';
    icBadge.className = 'metric-badge positive';
  } else if (Math.abs(meanIc) >= 0.02) {
    icBadge.textContent = '弱有效';
    icBadge.className = 'metric-badge neutral';
  } else {
    icBadge.textContent = '无明显相关';
    icBadge.className = 'metric-badge danger';
  }

  document.getElementById('factor-m-ic-std').textContent = (m.ic_std || 0).toFixed(4);
  document.getElementById('factor-m-ic-ir').textContent = (m.ic_ir || 0).toFixed(2);
  document.getElementById('factor-m-ann-ir').textContent = (m.annualized_ir || 0).toFixed(2);
  document.getElementById('factor-m-pos-ratio').textContent = `${(m.positive_ic_ratio || 0).toFixed(1)}%`;
  document.getElementById('factor-m-tstat').textContent = (m.ic_t_stat || 0).toFixed(2);

  // 多空收益
  const lsRet = m.long_short_return || 0;
  const lsEl = document.getElementById('factor-m-ls-return');
  lsEl.textContent = `${lsRet > 0 ? '+' : ''}${lsRet.toFixed(2)}%`;
  lsEl.className = `metric-value ${lsRet >= 0 ? 'text-green' : 'text-red'}`;

  document.getElementById('factor-m-ls-annual').textContent = `${(m.long_short_annual || 0).toFixed(2)}%`;
  document.getElementById('factor-m-ls-maxdd').textContent = `${(m.long_short_max_dd || 0).toFixed(2)}%`;
  document.getElementById('factor-m-mono-score').textContent = (m.monotonicity_score || 0).toFixed(2);

  // 渲染图表
  renderFactorChart();

  // 渲染分层绩效表
  renderFactorGroupTable(data.group_stats || []);
}

function renderFactorChart() {
  if (!factorState.chart || !factorState.currentResult) return;
  const data = factorState.currentResult;

  if (factorState.activeTab === 'nav') {
    // 5 组分层净值曲线 + 多空对冲线
    const dates = data.nav_dates || [];
    const qNavs = data.quantile_navs || {};
    const lsNav = data.long_short_nav || [];

    const palette = ['#ef4444', '#38bdf8', '#818cf8', '#94a3b8', '#64748b'];

    const series = [];
    Object.keys(qNavs).forEach((gKey, idx) => {
      series.push({
        name: `${gKey} 分组`,
        type: 'line',
        data: qNavs[gKey],
        smooth: true,
        showSymbol: false,
        lineStyle: { width: idx === 0 ? 2.5 : 1.5, color: palette[idx % palette.length] }
      });
    });

    if (lsNav.length > 0) {
      series.push({
        name: '多空对冲 (Long-Short)',
        type: 'line',
        data: lsNav,
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 3, color: '#10b981' }
      });
    }

    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(17, 22, 34, 0.95)',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        textStyle: { color: '#f8fafc', fontSize: 12 },
        formatter: (params) => {
          let s = `<div style="font-weight:600;margin-bottom:4px;">${params[0].axisValue}</div>`;
          params.forEach(p => {
            const ret = ((p.value - 1) * 100).toFixed(2);
            s += `<div>${p.marker} ${p.seriesName}: <b>${p.value.toFixed(4)}</b> (${ret > 0 ? '+' : ''}${ret}%)</div>`;
          });
          return s;
        }
      },
      legend: {
        show: true,
        top: 0,
        textStyle: { color: '#94a3b8', fontSize: 11 }
      },
      grid: { left: '3%', right: '4%', top: '14%', bottom: '8%', containLabel: true },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.15)' } },
        axisLabel: { color: '#64748b', fontSize: 11 }
      },
      yAxis: {
        type: 'value',
        scale: true,
        splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } },
        axisLabel: { color: '#64748b', fontSize: 11, formatter: v => v.toFixed(2) }
      },
      series: series
    };
    factorState.chart.setOption(option, true);
  } else if (factorState.activeTab === 'ic') {
    // Rank-IC 每日时序柱状图 + 累计 IC 曲线 (双轴)
    const icSeries = data.ic_series || [];
    const dates = icSeries.map(d => d.date);
    const dailyIc = icSeries.map(d => d.ic);
    const cumIc = icSeries.map(d => d.cumulative_ic);

    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(17, 22, 34, 0.95)',
        textStyle: { color: '#f8fafc', fontSize: 12 }
      },
      legend: {
        show: true,
        top: 0,
        textStyle: { color: '#94a3b8', fontSize: 11 }
      },
      grid: { left: '3%', right: '4%', top: '14%', bottom: '8%', containLabel: true },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.15)' } },
        axisLabel: { color: '#64748b', fontSize: 11 }
      },
      yAxis: [
        {
          type: 'value',
          name: '每日 Rank-IC',
          nameTextStyle: { color: '#64748b', fontSize: 11 },
          splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } },
          axisLabel: { color: '#64748b', fontSize: 11, formatter: v => v.toFixed(2) }
        },
        {
          type: 'value',
          name: '累计 IC',
          nameTextStyle: { color: '#38bdf8', fontSize: 11 },
          splitLine: { show: false },
          axisLabel: { color: '#38bdf8', fontSize: 11, formatter: v => v.toFixed(2) }
        }
      ],
      series: [
        {
          name: 'Rank-IC',
          type: 'bar',
          yAxisIndex: 0,
          data: dailyIc,
          itemStyle: {
            color: (params) => params.value >= 0 ? '#ef4444' : '#10b981'
          }
        },
        {
          name: '累计 IC 趋势',
          type: 'line',
          yAxisIndex: 1,
          data: cumIc,
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 2.5, color: '#38bdf8' }
        }
      ]
    };
    factorState.chart.setOption(option, true);
  } else if (factorState.activeTab === 'group_bar') {
    // 各分组年化收益柱状图
    const stats = data.group_stats || [];
    const groups = stats.map(s => s.group);
    const annReturns = stats.map(s => s.annual_return);
    const sharpes = stats.map(s => s.sharpe_ratio);

    const option = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' }
      },
      legend: {
        show: true,
        top: 0,
        textStyle: { color: '#94a3b8', fontSize: 11 }
      },
      grid: { left: '3%', right: '4%', top: '14%', bottom: '8%', containLabel: true },
      xAxis: {
        type: 'category',
        data: groups,
        axisLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.15)' } },
        axisLabel: { color: '#94a3b8', fontSize: 12 }
      },
      yAxis: [
        {
          type: 'value',
          name: '年化收益率 (%)',
          nameTextStyle: { color: '#64748b', fontSize: 11 },
          splitLine: { lineStyle: { color: 'rgba(255, 255, 255, 0.05)' } },
          axisLabel: { color: '#64748b', fontSize: 11, formatter: v => `${v}%` }
        },
        {
          type: 'value',
          name: '夏普比率',
          nameTextStyle: { color: '#06b6d4', fontSize: 11 },
          splitLine: { show: false },
          axisLabel: { color: '#06b6d4', fontSize: 11 }
        }
      ],
      series: [
        {
          name: '年化收益率 (%)',
          type: 'bar',
          yAxisIndex: 0,
          data: annReturns,
          itemStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: '#6366f1' },
              { offset: 1, color: '#4338ca' }
            ]),
            borderRadius: [4, 4, 0, 0]
          }
        },
        {
          name: '夏普比率',
          type: 'line',
          yAxisIndex: 1,
          data: sharpes,
          lineStyle: { width: 3, color: '#06b6d4' },
          itemStyle: { color: '#06b6d4' }
        }
      ]
    };
    factorState.chart.setOption(option, true);
  }
}

function renderFactorGroupTable(stats) {
  const tbody = document.getElementById('factor-group-table-body');
  if (!tbody) return;

  if (!stats || stats.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-cell">无分层统计数据</td></tr>`;
    return;
  }

  tbody.innerHTML = '';
  stats.forEach((s, idx) => {
    const tr = document.createElement('tr');
    const isTop = idx === 0;
    const isBottom = idx === stats.length - 1;
    const label = isTop ? 'Top 20% 优势多头' : (isBottom ? 'Bottom 20% 劣势空头' : `中间分层梯队 ${idx+1}`);

    const totClass = s.total_return > 0 ? 'text-green' : (s.total_return < 0 ? 'text-red' : '');
    const annClass = s.annual_return > 0 ? 'text-green' : (s.annual_return < 0 ? 'text-red' : '');

    tr.innerHTML = `
      <td><b>${s.group}</b></td>
      <td><span class="meta-tag" style="background:${isTop ? 'rgba(239, 68, 68, 0.2)' : 'rgba(255,255,255,0.06)'};color:${isTop ? '#f87171' : '#94a3b8'}">${label}</span></td>
      <td class="${totClass}">${s.total_return > 0 ? '+' : ''}${s.total_return.toFixed(2)}%</td>
      <td class="${annClass}">${s.annual_return > 0 ? '+' : ''}${s.annual_return.toFixed(2)}%</td>
      <td class="text-red">${s.max_drawdown.toFixed(2)}%</td>
      <td class="highlight-cyan">${s.sharpe_ratio.toFixed(2)}</td>
      <td>${isTop ? '⭐ 建议超配' : (isBottom ? '⚠️ 建议规避/融券做空' : '持平中性')}</td>
    `;
    tbody.appendChild(tr);
  });
}

