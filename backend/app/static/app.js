/**
 * AlphaQuant - A股全栈量化策略研究与回测看板交互逻辑
 */

// 全局状态管理
const state = {
  strategies: [],
  currentStrategyId: null,
  backtestResult: null,
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
  backtestForm: document.getElementById('backtest-form'),
  toggleCostBtn: document.getElementById('toggle-cost-btn'),
  costPanel: document.getElementById('cost-panel'),
  mainChart: document.getElementById('main-chart'),
  tradesCount: document.getElementById('trades-count'),
  tradesTableBody: document.getElementById('trades-table-body'),
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

// 根据当前选中的 Tab 绘制图表
async function renderActiveChart() {
  if (!state.backtestResult || !state.chartInstance) return;

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
  // 获取 K 线行情数据
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
  // ECharts Candlestick 顺序: [open, close, low, high]
  const oclh = klineData.map(d => [d.open, d.close, d.low, d.high]);

  // 构造买卖点 markPoint
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
          color: '#ef4444',         // 阳线红
          color0: '#10b981',        // 阴线绿
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
