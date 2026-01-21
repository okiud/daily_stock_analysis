# -*- coding: utf-8 -*-
"""
===================================
加密货币智能分析模块
===================================

职责：
1. 获取加密货币（BTC-USD 等）历史数据
2. 调用 AI 进行加密货币专属分析
3. 生成加密货币分析报告

特点：
- 24/7 全天候交易，与股票市场不同
- 波动性更大，关注不同的技术指标
- 定制化的 AI 分析 prompt
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List

from config import get_config
from data_provider.crypto_fetcher import CryptoFetcher, CRYPTO_NAME_MAP

logger = logging.getLogger(__name__)


@dataclass
class CryptoAnalysisResult:
    """
    加密货币分析结果数据类
    """
    symbol: str  # 如 BTC-USD
    name: str    # 如 比特币
    
    # 核心指标
    sentiment_score: int  # 0-100
    trend_prediction: str  # 强烈看多/看多/震荡/看空/强烈看空
    operation_advice: str  # 买入/加仓/持有/减仓/卖出/观望
    confidence_level: str = "中"  # 高/中/低
    
    # 价格数据
    current_price: float = 0.0
    price_change_24h: float = 0.0  # 24小时涨跌幅
    price_change_7d: float = 0.0   # 7日涨跌幅
    
    # 技术分析
    trend_analysis: str = ""
    ma_analysis: str = ""
    volume_analysis: str = ""
    volatility_analysis: str = ""
    
    # 市场情绪
    market_sentiment: str = ""
    news_summary: str = ""
    
    # 综合分析
    analysis_summary: str = ""
    key_points: str = ""
    risk_warning: str = ""
    
    # 决策仪表盘
    dashboard: Optional[Dict[str, Any]] = None
    
    # 元数据
    raw_response: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def get_emoji(self) -> str:
        """根据操作建议返回对应 emoji"""
        emoji_map = {
            '买入': '🟢', '加仓': '🟢', '强烈买入': '💚',
            '持有': '🟡', '观望': '⚪',
            '减仓': '🟠', '卖出': '🔴', '强烈卖出': '❌',
        }
        return emoji_map.get(self.operation_advice, '🟡')
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'confidence_level': self.confidence_level,
            'current_price': self.current_price,
            'price_change_24h': self.price_change_24h,
            'price_change_7d': self.price_change_7d,
            'trend_analysis': self.trend_analysis,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'dashboard': self.dashboard,
            'success': self.success,
        }


class CryptoAnalyzer:
    """
    加密货币 AI 分析器
    
    职责：
    1. 获取加密货币数据
    2. 调用 AI 进行分析
    3. 生成分析报告
    """
    
    # 加密货币专用系统提示词
    CRYPTO_SYSTEM_PROMPT = """你是一位专业的加密货币量化分析师，负责生成【加密货币决策仪表盘】分析报告。

## 你将收到的量化数据

### 1. 动量指标
- **RSI (14日)**：0-100，<30 超卖，>70 超买
- **MACD**：快慢线差值，柱状图，金叉/死叉信号
- **趋势强度 (trend_strength)**：综合评分，正值看多，负值看空

### 2. 波动指标
- **布林带**：上轨、中轨、下轨，位置 0-100%
- **ATR (平均真实波幅)**：波动大小，用于止损设置
- **年化波动率**：风险评估

### 3. 资金流向
- **主动买入占比 (buy_ratio)**：>50% 买方主导，<50% 卖方主导
- **资金流向 (money_flow)**：正值净流入，负值净流出
- **连续流入/流出天数 (flow_streak)**：趋势持续性

### 4. 均线系统
- **MA5/10/20/60**：多周期均线
- **MA趋势 (ma_trend)**：1=多头排列，-1=空头排列，0=震荡
- **MA20偏离度**：价格与20日均线的偏离程度

### 5. 支撑阻力
- **20日阻力位/支撑位**：近期高低点
- **距离阻力/支撑百分比**：当前位置评估

### 6. 量能指标
- **成交量/成交额**：市场活跃度
- **量比**：与近5日平均的比值
- **成交笔数**：交易活跃程度

## 量化策略规则（必须遵循）

### 买入信号组合
✅ RSI < 40 且 MACD 金叉 且 布林带位置 < 30% 且 资金净流入 → **强烈买入**
✅ 均线多头排列 且 价格回踩 MA20 且 量能放大 → **买入**
✅ RSI 超卖反弹 且 支撑位有效 → **逢低买入**

### 卖出信号组合
❌ RSI > 70 且 MACD 死叉 且 布林带位置 > 80% → **强烈卖出**
❌ 均线空头排列 且 跌破支撑位 且 资金净流出 → **卖出**
❌ 价格大幅偏离 MA20 (>15%) 且 量能萎缩 → **减仓**

### 观望条件
⚠️ RSI 在 40-60 之间 且 均线缠绕 → **震荡观望**
⚠️ 布林带收窄 (ATR下降) → **等待突破方向**

## 输出格式：JSON

```json
{
    "sentiment_score": 0-100整数,
    "trend_prediction": "强烈看多/看多/震荡/看空/强烈看空",
    "operation_advice": "买入/加仓/持有/减仓/卖出/观望",
    "confidence_level": "高/中/低",
    
    "dashboard": {
        "core_conclusion": {
            "one_sentence": "一句话核心结论（结合量化信号）",
            "signal_type": "🟢买入/🟡观望/🔴卖出/⚠️警告",
            "quant_signals": ["信号1", "信号2", "信号3"],
            "position_advice": {
                "no_position": "空仓者建议",
                "has_position": "持仓者建议"
            }
        },
        
        "quant_analysis": {
            "rsi_status": "超卖/正常/超买",
            "macd_status": "金叉/震荡/死叉",
            "boll_status": "下轨/中轨/上轨",
            "money_flow": "净流入/均衡/净流出",
            "ma_trend": "多头/震荡/空头",
            "overall_score": 0-100
        },
        
        "trading_plan": {
            "ideal_buy": "理想买入点 (最激进的低位入场价，基于布林带下轨或强支撑位，是分批挂单的最低一档)",
            "secondary_buy": "次优买入点 (保守入场价，位于理想买入点和当前价之间，约为理想买入点上方1-3%，用于价格未触及理想点就反弹时的备选入场)",
            "stop_loss": "止损位 (低于理想买入点，结合ATR计算，通常为理想买入点下方1-1.5倍ATR)",
            "take_profit": "目标位 (结合阻力位和布林带上轨)",
            "position_size": "建议仓位 (结合波动率调整)",
            "risk_reward": "风险收益比"
        },
        
        "signal_checklist": [
            "✅/⚠️/❌ RSI: 具体描述",
            "✅/⚠️/❌ MACD: 具体描述",
            "✅/⚠️/❌ 布林带: 具体描述",
            "✅/⚠️/❌ 资金流向: 具体描述",
            "✅/⚠️/❌ 均线趋势: 具体描述",
            "✅/⚠️/❌ 支撑阻力: 具体描述"
        ]
    },
    
    "analysis_summary": "100字综合分析（必须引用具体量化数据）",
    "key_points": "3-5个核心看点（基于量化信号）",
    "risk_warning": "风险提示（基于波动率和资金流向）",
    
    "trend_analysis": "趋势分析（结合MA趋势和趋势强度）",
    "momentum_analysis": "动量分析（结合RSI和MACD）",
    "volume_analysis": "量能分析（结合成交额、买入占比）",
    "volatility_analysis": "波动分析（结合ATR和布林带）"
}
```

## 评分规则（严格执行）

| 条件 | 评分调整 |
|------|---------|
| RSI < 30 (超卖) | +15 |
| RSI > 70 (超买) | -15 |
| MACD 金叉 | +10 |
| MACD 死叉 | -10 |
| 均线多头排列 | +15 |
| 均线空头排列 | -15 |
| 资金净流入 > 5% | +10 |
| 资金净流出 > 5% | -10 |
| 布林带位置 < 20% | +10 |
| 布林带位置 > 80% | -10 |
| 价格在支撑位附近 (<3%) | +5 |
| 价格在阻力位附近 (<3%) | -5 |

基础分 50 分，根据以上规则调整，最终评分 0-100。"""

    def __init__(self, analyzer=None, search_service=None):
        """
        初始化加密货币分析器
        
        Args:
            analyzer: AI 分析器（GeminiAnalyzer），可选
            search_service: 搜索服务，可选
        """
        self.fetcher = CryptoFetcher()
        self.analyzer = analyzer
        self.search_service = search_service
        self.config = get_config()
    
    def _init_ai_analyzer(self):
        """延迟初始化 AI 分析器"""
        if self.analyzer is None:
            from analyzer import GeminiAnalyzer
            self.analyzer = GeminiAnalyzer()
    
    def _format_crypto_prompt(
        self, 
        symbol: str, 
        name: str,
        df, 
        news_context: Optional[str] = None
    ) -> str:
        """
        格式化加密货币分析提示词（包含完整量化数据）
        """
        # 获取最新数据
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # 计算关键指标
        current_price = latest['close']
        price_change = ((current_price - prev['close']) / prev['close']) * 100
        
        # 7日涨跌幅
        if len(df) >= 7:
            price_7d_ago = df.iloc[-7]['close']
            change_7d = ((current_price - price_7d_ago) / price_7d_ago) * 100
        else:
            change_7d = 0
        
        # 构建提示词
        prompt = f"""请分析以下加密货币的走势并生成决策仪表盘：

## 基本信息
- **交易对**: {symbol}
- **名称**: {name}
- **当前价格**: ${current_price:,.2f}
- **24h涨跌**: {price_change:+.2f}%
- **7日涨跌**: {change_7d:+.2f}%

---

## 📊 量化指标数据

### 动量指标
- **RSI (14日)**: {latest.get('rsi', 50):.1f} ({latest.get('rsi_signal', 'neutral')})
- **MACD**: {latest.get('macd', 0):,.2f}
- **MACD信号线**: {latest.get('macd_signal', 0):,.2f}
- **MACD柱状图**: {latest.get('macd_hist', 0):,.2f}
- **MACD金叉/死叉**: {latest.get('macd_cross', 0)} (1=金叉, -1=死叉, 0=无)
- **趋势强度**: {latest.get('trend_strength', 0):.2f}

### 布林带
- **上轨**: ${latest.get('boll_upper', 0):,.2f}
- **中轨**: ${latest.get('boll_mid', 0):,.2f}
- **下轨**: ${latest.get('boll_lower', 0):,.2f}
- **当前位置**: {latest.get('boll_position', 50):.1f}% (0=下轨, 100=上轨)

### 波动指标
- **ATR (14日)**: ${latest.get('atr', 0):,.2f}
- **ATR百分比**: {latest.get('atr_percent', 0):.2f}%
- **年化波动率**: {latest.get('volatility', 0):.1%}

### 资金流向
- **主动买入占比**: {latest.get('buy_ratio', 50):.2f}%
- **资金流向**: {latest.get('money_flow', 0):+.2f} (正=净流入, 负=净流出)
- **连续流入/流出天数**: {latest.get('flow_streak', 0)}
- **成交额**: ${latest.get('amount', 0):,.0f}
- **成交笔数**: {latest.get('trade_count', 0):,}

### 均线系统
- **MA5**: ${latest.get('ma5', 0):,.2f}
- **MA10**: ${latest.get('ma10', 0):,.2f}
- **MA20**: ${latest.get('ma20', 0):,.2f}
- **MA60**: ${latest.get('ma60', 0):,.2f}
- **均线趋势**: {latest.get('ma_trend', 0)} (1=多头排列, 0=震荡, -1=空头排列)
- **MA20偏离度**: {latest.get('ma20_deviation', 0):+.2f}%
- **量比**: {latest.get('volume_ratio', 1):.2f}

### 支撑阻力
- **20日阻力位**: ${latest.get('resistance_20d', 0):,.2f} (距离 {latest.get('to_resistance', 0):.2f}%)
- **20日支撑位**: ${latest.get('support_20d', 0):,.2f} (距离 {latest.get('to_support', 0):.2f}%)

### 综合评分
- **系统买入信号**: {latest.get('buy_signal', 50):.0f}/100

---

## 📈 近10日数据走势

| 日期 | 收盘价 | 涨跌% | RSI | MACD柱 | 资金流向 | 买入占比 |
|------|--------|-------|-----|--------|---------|---------|
"""
        # 添加最近10日数据表格
        recent_data = df.tail(10)
        for _, row in recent_data.iterrows():
            date_str = row['date'].strftime('%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:5]
            prompt += f"| {date_str} | ${row['close']:,.0f} | {row['pct_chg']:+.1f}% | {row.get('rsi', 50):.0f} | {row.get('macd_hist', 0):+.0f} | {row.get('money_flow', 0):+.1f} | {row.get('buy_ratio', 50):.0f}% |\n"
        
        # 添加30天趋势汇总
        if len(df) >= 30:
            last_30d = df.tail(30)
            price_30d_ago = last_30d.iloc[0]['close']
            change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
            
            # 30天内的关键统计
            high_30d = last_30d['high'].max()
            low_30d = last_30d['low'].min()
            avg_volume_30d = last_30d['amount'].mean() if 'amount' in last_30d.columns else 0
            
            # 资金流向统计
            if 'money_flow' in last_30d.columns:
                inflow_days = (last_30d['money_flow'] > 0).sum()
                outflow_days = (last_30d['money_flow'] < 0).sum()
                total_flow = last_30d['money_flow'].sum()
            else:
                inflow_days = outflow_days = 0
                total_flow = 0
            
            # RSI 统计
            if 'rsi' in last_30d.columns:
                oversold_days = (last_30d['rsi'] < 30).sum()
                overbought_days = (last_30d['rsi'] > 70).sum()
            else:
                oversold_days = overbought_days = 0
            
            prompt += f"""
---

## 📊 30天趋势汇总

| 统计项 | 数值 |
|--------|------|
| 30日涨跌幅 | {change_30d:+.2f}% |
| 30日最高价 | ${high_30d:,.2f} |
| 30日最低价 | ${low_30d:,.2f} |
| 当前价距高点 | {((high_30d - current_price) / current_price * 100):.2f}% |
| 当前价距低点 | {((current_price - low_30d) / current_price * 100):.2f}% |
| 日均成交额 | ${avg_volume_30d:,.0f} |
| 资金净流入天数 | {inflow_days}天 / 净流出天数 {outflow_days}天 |
| 累计资金流向 | {total_flow:+.2f} |
| RSI超卖天数(<30) | {oversold_days}天 |
| RSI超买天数(>70) | {overbought_days}天 |

"""
        
        # 添加新闻（如果有）
        if news_context:
            prompt += f"\n---\n\n## 📰 相关新闻与市场动态\n{news_context}\n"
        
        prompt += """
---

请根据以上量化数据（包括实时指标和30天历史趋势），严格按照系统提示词中的评分规则和策略逻辑，输出 JSON 格式的分析结果。

**重要要求**：
1. 分析结论必须引用具体的量化数据作为依据
2. 交易计划中的买入点/止损位/目标位必须给出具体价格
3. 综合考虑短期(10日)和中期(30日)趋势
4. **阶梯式买入价格约束（必须严格遵守）**：
   - 止损位 < 理想买入点 < 次优买入点 < 当前价格 < 目标位
   - 理想买入点 = 布林带下轨或支撑位附近（最激进的低位入场）
   - 次优买入点 = 理想买入点 × 1.01~1.03（略高于理想点1-3%）
   - 例如：当前价$90,000，支撑位$87,000，则理想买入点≈$87,000，次优买入点≈$87,870~$89,610
"""
        
        return prompt
    
    def analyze_crypto(
        self, 
        symbol: str,
        search_news: bool = True
    ) -> CryptoAnalysisResult:
        """
        分析单个加密货币
        
        Args:
            symbol: 加密货币符号（如 BTC, BTC-USD）
            search_news: 是否搜索相关新闻
            
        Returns:
            CryptoAnalysisResult 分析结果
        """
        # 标准化符号
        normalized_symbol = CryptoFetcher.normalize_symbol(symbol)
        name = CryptoFetcher.get_crypto_name(symbol)
        
        logger.info(f"========== 开始分析 {name}({normalized_symbol}) ==========")
        
        try:
            # Step 1: 获取数据
            logger.info(f"[{normalized_symbol}] 获取历史数据...")
            df = self.fetcher.get_daily_data(symbol)
            
            if df.empty:
                raise ValueError(f"无法获取 {normalized_symbol} 的数据")
            
            latest = df.iloc[-1]
            current_price = latest['close']
            
            logger.info(f"[{normalized_symbol}] 获取成功，当前价格: ${current_price:,.2f}")
            
            # Step 2: 搜索新闻（可选）
            news_context = None
            if search_news and self.search_service and self.search_service.is_available:
                logger.info(f"[{normalized_symbol}] 搜索相关新闻...")
                try:
                    # 使用加密货币名称搜索新闻
                    crypto_code = normalized_symbol.replace('-USD', '')
                    search_result = self.search_service.search_stock_news(
                        stock_code=crypto_code,
                        stock_name=f"{name} {crypto_code} 加密货币",
                        max_results=5
                    )
                    if search_result and search_result.success:
                        news_context = "\n".join([
                            f"- {r.title}: {r.snippet}" 
                            for r in search_result.results[:5]
                        ])
                        logger.info(f"[{normalized_symbol}] 获取到 {len(search_result.results)} 条新闻")
                except Exception as e:
                    logger.warning(f"[{normalized_symbol}] 搜索新闻失败: {e}")
            
            # Step 3: AI 分析
            self._init_ai_analyzer()
            
            if not self.analyzer or not self.analyzer.is_available():
                logger.warning("AI 分析器不可用，返回基础分析结果")
                return self._create_basic_result(normalized_symbol, name, df)
            
            # 格式化提示词
            prompt = self._format_crypto_prompt(normalized_symbol, name, df, news_context)
            
            logger.info(f"[{normalized_symbol}] 调用 AI 分析...")
            logger.debug(f"Prompt 长度: {len(prompt)} 字符")
            
            # 调用 AI（使用加密货币专用提示词）
            # 临时替换系统提示词
            original_prompt = self.analyzer.SYSTEM_PROMPT
            self.analyzer.SYSTEM_PROMPT = self.CRYPTO_SYSTEM_PROMPT
            
            try:
                generation_config = {
                    "temperature": 0.7,
                    "max_output_tokens": 64000,
                }
                
                response_text = self.analyzer._call_api_with_retry(prompt, generation_config)
                
                # 解析响应
                result = self._parse_ai_response(normalized_symbol, name, df, response_text)
                result.raw_response = response_text
                
                logger.info(f"[{normalized_symbol}] 分析完成: {result.operation_advice}, 评分 {result.sentiment_score}")
                
                return result
                
            finally:
                # 恢复原始提示词
                self.analyzer.SYSTEM_PROMPT = original_prompt
            
        except Exception as e:
            logger.error(f"[{symbol}] 分析失败: {e}")
            return CryptoAnalysisResult(
                symbol=normalized_symbol,
                name=name,
                sentiment_score=50,
                trend_prediction="震荡",
                operation_advice="观望",
                analysis_summary=f"分析失败: {str(e)}",
                success=False,
                error_message=str(e),
            )
    
    def _parse_ai_response(
        self, 
        symbol: str, 
        name: str, 
        df, 
        response_text: str
    ) -> CryptoAnalysisResult:
        """解析 AI 响应"""
        import json
        import re
        
        try:
            # 提取 JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("无法从响应中提取 JSON")
            
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            return CryptoAnalysisResult(
                symbol=symbol,
                name=name,
                sentiment_score=data.get('sentiment_score', 50),
                trend_prediction=data.get('trend_prediction', '震荡'),
                operation_advice=data.get('operation_advice', '观望'),
                confidence_level=data.get('confidence_level', '中'),
                current_price=latest['close'],
                price_change_24h=((latest['close'] - prev['close']) / prev['close']) * 100,
                trend_analysis=data.get('trend_analysis', ''),
                ma_analysis=data.get('ma_analysis', ''),
                volume_analysis=data.get('volume_analysis', ''),
                volatility_analysis=data.get('volatility_analysis', ''),
                market_sentiment=data.get('market_sentiment', ''),
                news_summary=data.get('news_summary', ''),
                analysis_summary=data.get('analysis_summary', ''),
                key_points=data.get('key_points', ''),
                risk_warning=data.get('risk_warning', ''),
                dashboard=data.get('dashboard'),
                success=True,
            )
            
        except Exception as e:
            logger.error(f"解析 AI 响应失败: {e}")
            return self._create_basic_result(symbol, name, df)
    
    def _create_basic_result(self, symbol: str, name: str, df) -> CryptoAnalysisResult:
        """创建基础分析结果（无 AI）"""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # 简单的趋势判断
        ma20 = latest.get('ma20', latest['close'])
        ma60 = latest.get('ma60', latest['close'])
        
        if ma20 > ma60 and latest['close'] > ma20:
            trend = "看多"
            advice = "持有"
            score = 65
        elif ma20 < ma60 and latest['close'] < ma20:
            trend = "看空"
            advice = "观望"
            score = 35
        else:
            trend = "震荡"
            advice = "观望"
            score = 50
        
        return CryptoAnalysisResult(
            symbol=symbol,
            name=name,
            sentiment_score=score,
            trend_prediction=trend,
            operation_advice=advice,
            current_price=latest['close'],
            price_change_24h=((latest['close'] - prev['close']) / prev['close']) * 100,
            analysis_summary=f"{name}当前价格 ${latest['close']:,.2f}，MA20: ${ma20:,.2f}，MA60: ${ma60:,.2f}",
            success=True,
        )
    
    def run(
        self, 
        symbols: Optional[List[str]] = None,
        send_notification: bool = True
    ) -> List[CryptoAnalysisResult]:
        """
        运行加密货币分析
        
        Args:
            symbols: 要分析的加密货币列表，默认使用配置
            send_notification: 是否发送通知
            
        Returns:
            分析结果列表
        """
        # 获取要分析的加密货币列表
        if symbols is None:
            symbols = getattr(self.config, 'crypto_list', ['BTC-USD'])
        
        if not symbols:
            symbols = ['BTC-USD']  # 默认分析 BTC
        
        logger.info(f"===== 开始加密货币分析 =====")
        logger.info(f"分析列表: {', '.join(symbols)}")
        
        results = []
        
        for symbol in symbols:
            try:
                result = self.analyze_crypto(symbol)
                results.append(result)
            except Exception as e:
                logger.error(f"分析 {symbol} 失败: {e}")
        
        # 输出摘要
        if results:
            logger.info("\n===== 加密货币分析结果摘要 =====")
            for r in results:
                emoji = r.get_emoji()
                logger.info(
                    f"{emoji} {r.name}({r.symbol}): {r.operation_advice} | "
                    f"评分 {r.sentiment_score} | ${r.current_price:,.2f}"
                )
        
        # 发送通知（如果需要）
        if send_notification and results:
            self._send_notifications(results)
        
        return results
    
    def _send_notifications(self, results: List[CryptoAnalysisResult]) -> None:
        """发送分析结果通知"""
        try:
            from notification import NotificationService
            notifier = NotificationService()
            
            if not notifier.is_available():
                logger.info("通知渠道未配置，跳过推送")
                return
            
            # 生成报告
            report = self._generate_report(results)
            
            # 保存到本地
            date_str = datetime.now().strftime('%Y%m%d')
            filepath = notifier.save_report_to_file(report, f"crypto_analysis_{date_str}.md")
            logger.info(f"加密货币分析报告已保存: {filepath}")
            
            # 推送通知
            if notifier.send(report):
                logger.info("加密货币分析报告推送成功")
            else:
                logger.warning("加密货币分析报告推送失败")
                
        except Exception as e:
            logger.error(f"发送通知失败: {e}")
    
    def _generate_report(self, results: List[CryptoAnalysisResult]) -> str:
        """生成加密货币分析报告（包含量化指标和买卖建议）"""
        date_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        report = f"# 🪙 加密货币量化分析报告\n\n"
        report += f"**生成时间**: {date_str}\n\n"
        report += "---\n\n"
        
        for r in results:
            emoji = r.get_emoji()
            report += f"## {emoji} {r.name} ({r.symbol})\n\n"
            
            # 核心指标表格
            report += "### 📊 核心指标\n\n"
            report += "| 指标 | 数值 |\n"
            report += "|------|------|\n"
            report += f"| 当前价格 | **${r.current_price:,.2f}** |\n"
            report += f"| 24h涨跌 | {r.price_change_24h:+.2f}% |\n"
            report += f"| 综合评分 | **{r.sentiment_score}/100** |\n"
            report += f"| 趋势判断 | {r.trend_prediction} |\n"
            report += f"| 操作建议 | **{r.operation_advice}** |\n"
            report += f"| 置信度 | {r.confidence_level} |\n\n"
            
            # 从 dashboard 提取交易计划
            if r.dashboard:
                dashboard = r.dashboard
                
                # 核心结论
                if 'core_conclusion' in dashboard:
                    core = dashboard['core_conclusion']
                    report += "### 🎯 核心结论\n\n"
                    if 'one_sentence' in core:
                        report += f"> {core['one_sentence']}\n\n"
                    if 'signal_type' in core:
                        report += f"**信号类型**: {core['signal_type']}\n\n"
                    if 'quant_signals' in core and core['quant_signals']:
                        report += "**量化信号**:\n"
                        for sig in core['quant_signals']:
                            report += f"- {sig}\n"
                        report += "\n"
                    if 'position_advice' in core:
                        pa = core['position_advice']
                        report += f"- 空仓者: {pa.get('no_position', '-')}\n"
                        report += f"- 持仓者: {pa.get('has_position', '-')}\n\n"
                
                # 量化分析
                if 'quant_analysis' in dashboard:
                    qa = dashboard['quant_analysis']
                    report += "### 📈 量化分析\n\n"
                    report += "| 指标 | 状态 |\n"
                    report += "|------|------|\n"
                    if 'rsi_status' in qa:
                        report += f"| RSI | {qa['rsi_status']} |\n"
                    if 'macd_status' in qa:
                        report += f"| MACD | {qa['macd_status']} |\n"
                    if 'boll_status' in qa:
                        report += f"| 布林带 | {qa['boll_status']} |\n"
                    if 'money_flow' in qa:
                        report += f"| 资金流向 | {qa['money_flow']} |\n"
                    if 'ma_trend' in qa:
                        report += f"| 均线趋势 | {qa['ma_trend']} |\n"
                    report += "\n"
                
                # 交易计划（买卖点位）
                if 'trading_plan' in dashboard:
                    tp = dashboard['trading_plan']
                    report += "### 💰 交易计划\n\n"
                    report += "| 项目 | 建议 |\n"
                    report += "|------|------|\n"
                    if 'ideal_buy' in tp:
                        report += f"| 🟢 理想买入点 | {tp['ideal_buy']} |\n"
                    if 'secondary_buy' in tp:
                        report += f"| 🔵 次优买入点 | {tp['secondary_buy']} |\n"
                    if 'stop_loss' in tp:
                        report += f"| 🔴 止损位 | {tp['stop_loss']} |\n"
                    if 'take_profit' in tp:
                        report += f"| 🎯 目标位 | {tp['take_profit']} |\n"
                    if 'position_size' in tp:
                        report += f"| 📊 建议仓位 | {tp['position_size']} |\n"
                    if 'risk_reward' in tp:
                        report += f"| ⚖️ 风险收益比 | {tp['risk_reward']} |\n"
                    report += "\n"
                
                # 信号检查清单
                if 'signal_checklist' in dashboard:
                    report += "### ✅ 信号检查清单\n\n"
                    for item in dashboard['signal_checklist']:
                        report += f"- {item}\n"
                    report += "\n"
            
            # 分析摘要
            if r.analysis_summary:
                report += f"### 📝 分析摘要\n\n{r.analysis_summary}\n\n"
            
            # 核心看点
            if r.key_points:
                report += f"### 🔑 核心看点\n\n{r.key_points}\n\n"
            
            # 趋势分析
            if r.trend_analysis:
                report += f"### 📊 趋势分析\n\n{r.trend_analysis}\n\n"
            
            # 风险提示
            if r.risk_warning:
                report += f"### ⚠️ 风险提示\n\n{r.risk_warning}\n\n"
            
            report += "---\n\n"
        
        report += "\n*本报告基于量化策略分析，仅供参考，不构成投资建议。加密货币市场风险较高，请谨慎投资。*\n"
        
        return report


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    analyzer = CryptoAnalyzer()
    
    # 测试分析 BTC
    result = analyzer.analyze_crypto('BTC-USD', search_news=False)
    
    print(f"\n===== {result.name} 分析结果 =====")
    print(f"价格: ${result.current_price:,.2f}")
    print(f"评分: {result.sentiment_score}")
    print(f"趋势: {result.trend_prediction}")
    print(f"建议: {result.operation_advice}")
    print(f"摘要: {result.analysis_summary}")
