# -*- coding: utf-8 -*-
"""
===================================
CryptoFetcher - 加密货币多数据源管理器
===================================

数据来源（按优先级）：
1. Binance API - 全球最大交易所，实时价格
2. OKX API - 备用交易所
3. CoinGecko API - 聚合数据源
4. Yahoo Finance - 兜底（有延迟）

特点：
1. 自动故障切换
2. 实时价格获取（延迟<1秒）
3. 支持历史数据（60天）
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS

logger = logging.getLogger(__name__)


# 常用加密货币名称映射
CRYPTO_NAME_MAP = {
    'BTC-USD': '比特币', 'BTC': '比特币', 'BTCUSDT': '比特币',
    'ETH-USD': '以太坊', 'ETH': '以太坊', 'ETHUSDT': '以太坊',
    'BNB-USD': '币安币', 'BNB': '币安币', 'BNBUSDT': '币安币',
    'SOL-USD': 'Solana', 'SOL': 'Solana', 'SOLUSDT': 'Solana',
    'XRP-USD': '瑞波币', 'XRP': '瑞波币', 'XRPUSDT': '瑞波币',
    'ADA-USD': '艾达币', 'DOGE-USD': '狗狗币', 'DOT-USD': '波卡',
    'MATIC-USD': 'Polygon', 'LTC-USD': '莱特币',
}


@dataclass
class CryptoTicker:
    """加密货币实时行情"""
    symbol: str
    price: float
    volume_24h: float = 0.0
    change_24h: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    source: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseCryptoSource(ABC):
    """加密货币数据源基类"""
    
    name: str = "BaseSource"
    priority: int = 99
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> CryptoTicker:
        """获取实时行情"""
        pass
    
    @abstractmethod
    def get_klines(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """获取K线数据"""
        pass
    
    @staticmethod
    def normalize_symbol(symbol: str, target_format: str = "standard") -> str:
        """
        标准化加密货币符号
        
        target_format:
        - "standard": BTC-USD
        - "binance": BTCUSDT
        - "okx": BTC-USDT
        - "coingecko": bitcoin
        """
        symbol = symbol.upper().strip()
        
        # 提取基础币种
        base = symbol.replace('-USD', '').replace('-USDT', '').replace('USDT', '').replace('USD', '')
        
        if target_format == "binance":
            return f"{base}USDT"
        elif target_format == "okx":
            return f"{base}-USDT"
        elif target_format == "coingecko":
            # CoinGecko 使用小写全名
            mapping = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
                      'SOL': 'solana', 'XRP': 'ripple', 'ADA': 'cardano',
                      'DOGE': 'dogecoin', 'DOT': 'polkadot', 'LTC': 'litecoin'}
            return mapping.get(base, base.lower())
        else:  # standard
            return f"{base}-USD"


class BinanceSource(BaseCryptoSource):
    """
    币安数据源 - 优先级最高
    
    API文档: https://binance-docs.github.io/apidocs/
    公开接口，无需 API Key
    """
    
    name = "Binance"
    priority = 1
    BASE_URL = "https://api.binance.com"
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def get_ticker(self, symbol: str) -> CryptoTicker:
        """获取实时行情"""
        binance_symbol = self.normalize_symbol(symbol, "binance")
        
        try:
            # 24小时行情
            url = f"{self.BASE_URL}/api/v3/ticker/24hr"
            resp = requests.get(url, params={"symbol": binance_symbol}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            return CryptoTicker(
                symbol=self.normalize_symbol(symbol, "standard"),
                price=float(data['lastPrice']),
                volume_24h=float(data['volume']),
                change_24h=float(data['priceChangePercent']),
                high_24h=float(data['highPrice']),
                low_24h=float(data['lowPrice']),
                source=self.name,
            )
        except Exception as e:
            logger.warning(f"[Binance] 获取 {symbol} 行情失败: {e}")
            raise DataFetchError(f"Binance: {e}")
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def get_klines(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """获取K线数据"""
        binance_symbol = self.normalize_symbol(symbol, "binance")
        
        try:
            url = f"{self.BASE_URL}/api/v3/klines"
            params = {
                "symbol": binance_symbol,
                "interval": "1d",
                "limit": days,
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            # 转换为 DataFrame
            # Binance K线: [开盘时间, 开, 高, 低, 收, 成交量, 收盘时间, 成交额, 成交笔数, 主动买入量, 主动买入额, 忽略]
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['date'] = pd.to_datetime(df['open_time'], unit='ms')
            df['code'] = self.normalize_symbol(symbol, "standard")
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 使用 API 返回的真实成交额 (quote_volume)
            df['amount'] = df['quote_volume']
            # 成交笔数
            df['trade_count'] = df['trades']
            # 主动买入占比 = 主动买入额 / 总成交额
            df['buy_ratio'] = (df['taker_buy_quote'] / df['quote_volume'] * 100).round(2)
            
            df['pct_chg'] = df['close'].pct_change() * 100
            df['pct_chg'] = df['pct_chg'].fillna(0).round(2)
            
            return df[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'trade_count', 'buy_ratio']]
            
        except Exception as e:
            logger.warning(f"[Binance] 获取 {symbol} K线失败: {e}")
            raise DataFetchError(f"Binance K线: {e}")


class OKXSource(BaseCryptoSource):
    """
    OKX数据源 - 优先级2
    
    API文档: https://www.okx.com/docs-v5/
    公开接口，无需 API Key
    """
    
    name = "OKX"
    priority = 2
    BASE_URL = "https://www.okx.com"
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def get_ticker(self, symbol: str) -> CryptoTicker:
        """获取实时行情"""
        okx_symbol = self.normalize_symbol(symbol, "okx")
        
        try:
            url = f"{self.BASE_URL}/api/v5/market/ticker"
            resp = requests.get(url, params={"instId": okx_symbol}, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            
            if result.get('code') != '0' or not result.get('data'):
                raise DataFetchError(f"OKX API 错误: {result.get('msg')}")
            
            data = result['data'][0]
            
            return CryptoTicker(
                symbol=self.normalize_symbol(symbol, "standard"),
                price=float(data['last']),
                volume_24h=float(data['vol24h']),
                change_24h=float(data.get('sodUtc8', 0)) if data.get('sodUtc8') else 0,
                high_24h=float(data['high24h']),
                low_24h=float(data['low24h']),
                source=self.name,
            )
        except Exception as e:
            logger.warning(f"[OKX] 获取 {symbol} 行情失败: {e}")
            raise DataFetchError(f"OKX: {e}")
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def get_klines(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """获取K线数据"""
        okx_symbol = self.normalize_symbol(symbol, "okx")
        
        try:
            url = f"{self.BASE_URL}/api/v5/market/candles"
            params = {
                "instId": okx_symbol,
                "bar": "1D",
                "limit": str(days),
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            result = resp.json()
            
            if result.get('code') != '0':
                raise DataFetchError(f"OKX API 错误: {result.get('msg')}")
            
            # OKX K线返回: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
            # vol = 成交量(币), volCcy = 成交额(计价币), volCcyQuote = 成交额(USDT)
            data = result['data']
            df = pd.DataFrame(data, columns=[
                'ts', 'open', 'high', 'low', 'close', 'volume', 'vol_ccy', 'vol_quote', 'confirm'
            ])
            
            df['date'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
            df['code'] = self.normalize_symbol(symbol, "standard")
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'vol_ccy', 'vol_quote']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 使用 API 返回的真实成交额 (vol_quote = USDT成交额)
            df['amount'] = df['vol_quote']
            df['pct_chg'] = df['close'].pct_change() * 100
            df['pct_chg'] = df['pct_chg'].fillna(0).round(2)
            
            # OKX 不提供成交笔数和买入占比，设为空
            df['trade_count'] = 0
            df['buy_ratio'] = 50.0  # 默认50%
            
            # OKX 返回倒序，需要反转
            df = df.sort_values('date').reset_index(drop=True)
            
            return df[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'trade_count', 'buy_ratio']]
            
        except Exception as e:
            logger.warning(f"[OKX] 获取 {symbol} K线失败: {e}")
            raise DataFetchError(f"OKX K线: {e}")


class CoinGeckoSource(BaseCryptoSource):
    """
    CoinGecko数据源 - 优先级3
    
    API文档: https://www.coingecko.com/api/documentation
    免费30次/分钟，可选 API Key
    """
    
    name = "CoinGecko"
    priority = 3
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def get_ticker(self, symbol: str) -> CryptoTicker:
        """获取实时行情"""
        coin_id = self.normalize_symbol(symbol, "coingecko")
        
        try:
            url = f"{self.BASE_URL}/simple/price"
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if coin_id not in data:
                raise DataFetchError(f"CoinGecko 未找到 {coin_id}")
            
            coin_data = data[coin_id]
            
            return CryptoTicker(
                symbol=self.normalize_symbol(symbol, "standard"),
                price=float(coin_data['usd']),
                volume_24h=float(coin_data.get('usd_24h_vol', 0)),
                change_24h=float(coin_data.get('usd_24h_change', 0)),
                source=self.name,
            )
        except Exception as e:
            logger.warning(f"[CoinGecko] 获取 {symbol} 行情失败: {e}")
            raise DataFetchError(f"CoinGecko: {e}")
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def get_klines(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """获取K线数据（CoinGecko 返回简化数据）"""
        coin_id = self.normalize_symbol(symbol, "coingecko")
        
        try:
            url = f"{self.BASE_URL}/coins/{coin_id}/ohlc"
            params = {"vs_currency": "usd", "days": str(days)}
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            # [timestamp, open, high, low, close]
            df = pd.DataFrame(data, columns=['ts', 'open', 'high', 'low', 'close'])
            df['date'] = pd.to_datetime(df['ts'], unit='ms')
            df['code'] = self.normalize_symbol(symbol, "standard")
            
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['volume'] = 0  # CoinGecko OHLC 不含成交量
            df['amount'] = 0
            df['pct_chg'] = df['close'].pct_change() * 100
            df['pct_chg'] = df['pct_chg'].fillna(0).round(2)
            df['trade_count'] = 0
            df['buy_ratio'] = 50.0
            
            return df[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'trade_count', 'buy_ratio']]
            
        except Exception as e:
            logger.warning(f"[CoinGecko] 获取 {symbol} K线失败: {e}")
            raise DataFetchError(f"CoinGecko K线: {e}")


class YfinanceSource(BaseCryptoSource):
    """
    Yahoo Finance数据源 - 兜底
    
    有15分钟延迟，作为最后备选
    """
    
    name = "YFinance"
    priority = 4
    
    def get_ticker(self, symbol: str) -> CryptoTicker:
        """获取行情（有延迟）"""
        try:
            import yfinance as yf
            std_symbol = self.normalize_symbol(symbol, "standard")
            
            ticker = yf.Ticker(std_symbol)
            info = ticker.info
            
            return CryptoTicker(
                symbol=std_symbol,
                price=float(info.get('regularMarketPrice', 0) or info.get('previousClose', 0)),
                volume_24h=float(info.get('volume24Hr', 0) or info.get('volume', 0)),
                change_24h=float(info.get('regularMarketChangePercent', 0)),
                high_24h=float(info.get('dayHigh', 0)),
                low_24h=float(info.get('dayLow', 0)),
                source=self.name,
            )
        except Exception as e:
            logger.warning(f"[YFinance] 获取 {symbol} 行情失败: {e}")
            raise DataFetchError(f"YFinance: {e}")
    
    def get_klines(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """获取K线数据"""
        try:
            import yfinance as yf
            std_symbol = self.normalize_symbol(symbol, "standard")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days * 2)
            
            df = yf.download(
                std_symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True,
            )
            
            if df.empty:
                raise DataFetchError(f"YFinance 无数据: {std_symbol}")
            
            # 处理 MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            df.columns = [str(c).lower() for c in df.columns]
            df['code'] = std_symbol
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['amount'] = df['volume'] * df['close']
            df['pct_chg'] = df['close'].pct_change() * 100
            df['pct_chg'] = df['pct_chg'].fillna(0).round(2)
            df['trade_count'] = 0  # YFinance 不提供
            df['buy_ratio'] = 50.0  # 默认50%
            
            return df[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'trade_count', 'buy_ratio']].tail(days)
            
        except Exception as e:
            logger.warning(f"[YFinance] 获取 {symbol} K线失败: {e}")
            raise DataFetchError(f"YFinance K线: {e}")


class CryptoDataManager:
    """
    加密货币数据源管理器
    
    特点：
    - 多数据源自动切换
    - 实时价格获取
    - 历史K线数据
    """
    
    def __init__(self):
        self.sources: List[BaseCryptoSource] = [
            BinanceSource(),
            OKXSource(),
            CoinGeckoSource(),
            YfinanceSource(),
        ]
        # 按优先级排序
        self.sources.sort(key=lambda s: s.priority)
        logger.info(f"加密货币数据源初始化: {[s.name for s in self.sources]}")
    
    def get_realtime_price(self, symbol: str) -> CryptoTicker:
        """
        获取实时价格（自动故障切换）
        """
        errors = []
        
        for source in self.sources:
            try:
                ticker = source.get_ticker(symbol)
                logger.info(f"[{source.name}] {symbol} 实时价格: ${ticker.price:,.2f}")
                return ticker
            except Exception as e:
                errors.append(f"[{source.name}] {e}")
                continue
        
        error_msg = f"所有数据源获取 {symbol} 实时价格失败:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise DataFetchError(error_msg)
    
    def get_daily_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """
        获取日K线数据（自动故障切换）
        """
        errors = []
        
        for source in self.sources:
            try:
                df = source.get_klines(symbol, days)
                if not df.empty:
                    # 计算技术指标
                    df = self._calculate_indicators(df)
                    logger.info(f"[{source.name}] {symbol} 获取 {len(df)} 条K线数据")
                    return df
            except Exception as e:
                errors.append(f"[{source.name}] {e}")
                continue
        
        error_msg = f"所有数据源获取 {symbol} K线数据失败:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise DataFetchError(error_msg)
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算量化策略指标
        
        包含：
        - 均线系统：MA5/10/20/60
        - 动量指标：RSI、MACD
        - 波动指标：布林带、ATR
        - 资金流向：买入占比、资金流入
        - 趋势指标：趋势强度、均线多空排列
        - 支撑阻力：近期高低点
        """
        df = df.copy()
        
        # ==================== 均线系统 ====================
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean().round(2)
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean().round(2)
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean().round(2)
        df['ma60'] = df['close'].rolling(window=60, min_periods=1).mean().round(2)
        
        # 均线多空排列 (1=多头排列, -1=空头排列, 0=震荡)
        df['ma_trend'] = 0
        bullish_ma = (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20'])
        bearish_ma = (df['ma5'] < df['ma10']) & (df['ma10'] < df['ma20'])
        df.loc[bullish_ma, 'ma_trend'] = 1
        df.loc[bearish_ma, 'ma_trend'] = -1
        
        # ==================== RSI (14日) ====================
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 0.0001)
        df['rsi'] = (100 - (100 / (1 + rs))).round(2)
        
        # RSI 信号 (超买>70, 超卖<30)
        df['rsi_signal'] = 'neutral'
        df.loc[df['rsi'] > 70, 'rsi_signal'] = 'overbought'
        df.loc[df['rsi'] < 30, 'rsi_signal'] = 'oversold'
        
        # ==================== MACD ====================
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = (ema12 - ema26).round(2)
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean().round(2)
        df['macd_hist'] = (df['macd'] - df['macd_signal']).round(2)
        
        # MACD 金叉/死叉
        df['macd_cross'] = 0
        df.loc[(df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1)), 'macd_cross'] = 1  # 金叉
        df.loc[(df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1)), 'macd_cross'] = -1  # 死叉
        
        # ==================== 布林带 ====================
        df['boll_mid'] = df['close'].rolling(window=20, min_periods=1).mean().round(2)
        std = df['close'].rolling(window=20, min_periods=1).std()
        df['boll_upper'] = (df['boll_mid'] + 2 * std).round(2)
        df['boll_lower'] = (df['boll_mid'] - 2 * std).round(2)
        
        # 布林带位置 (0-100, 50为中轨)
        boll_range = df['boll_upper'] - df['boll_lower']
        df['boll_position'] = ((df['close'] - df['boll_lower']) / boll_range.replace(0, 1) * 100).round(1)
        
        # ==================== ATR (平均真实波幅) ====================
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14, min_periods=1).mean().round(2)
        df['atr_percent'] = (df['atr'] / df['close'] * 100).round(2)  # ATR占价格百分比
        
        # ==================== 量能指标 ====================
        # 量比
        avg_vol = df['volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ratio'] = (df['volume'] / avg_vol.shift(1)).fillna(1.0).round(2)
        
        # 成交额变化率
        if 'amount' in df.columns:
            df['amount_change'] = df['amount'].pct_change().fillna(0).round(4)
        
        # 波动率 (年化)
        df['volatility'] = (
            df['pct_chg'].rolling(window=20, min_periods=5).std()
            * (365 ** 0.5) / 100
        ).round(4)
        
        # ==================== 资金流向 (基于 buy_ratio) ====================
        if 'buy_ratio' in df.columns:
            # 资金流入强度 = 买入占比 - 50 (正值=净流入, 负值=净流出)
            df['money_flow'] = (df['buy_ratio'] - 50).round(2)
            
            # 连续资金流入/流出天数
            df['flow_streak'] = 0
            for i in range(1, len(df)):
                if df.iloc[i]['money_flow'] > 0 and df.iloc[i-1]['money_flow'] > 0:
                    df.iloc[i, df.columns.get_loc('flow_streak')] = max(1, df.iloc[i-1]['flow_streak'] + 1)
                elif df.iloc[i]['money_flow'] < 0 and df.iloc[i-1]['money_flow'] < 0:
                    df.iloc[i, df.columns.get_loc('flow_streak')] = min(-1, df.iloc[i-1]['flow_streak'] - 1)
        
        # ==================== 趋势强度 ====================
        # 价格相对MA20的偏离度
        df['ma20_deviation'] = ((df['close'] - df['ma20']) / df['ma20'] * 100).round(2)
        
        # 趋势强度评分 (综合多个指标)
        df['trend_strength'] = 0.0
        # RSI 贡献
        df['trend_strength'] += (df['rsi'] - 50) / 50  # -1 to 1
        # MA趋势贡献
        df['trend_strength'] += df['ma_trend']  # -1, 0, 1
        # MACD贡献
        df['trend_strength'] += (df['macd_hist'] / df['close'] * 1000).clip(-1, 1)
        # 资金流向贡献
        if 'money_flow' in df.columns:
            df['trend_strength'] += (df['money_flow'] / 10).clip(-1, 1)
        df['trend_strength'] = df['trend_strength'].round(2)
        
        # ==================== 支撑阻力位 ====================
        # 近20日高低点
        df['resistance_20d'] = df['high'].rolling(window=20, min_periods=1).max().round(2)
        df['support_20d'] = df['low'].rolling(window=20, min_periods=1).min().round(2)
        
        # 距离支撑/阻力的百分比
        df['to_resistance'] = ((df['resistance_20d'] - df['close']) / df['close'] * 100).round(2)
        df['to_support'] = ((df['close'] - df['support_20d']) / df['close'] * 100).round(2)
        
        # ==================== 综合信号 ====================
        # 买入信号强度 (0-100)
        df['buy_signal'] = 50.0  # 基础分
        # RSI 超卖加分
        df.loc[df['rsi'] < 30, 'buy_signal'] += 20
        df.loc[df['rsi'] < 40, 'buy_signal'] += 10
        # 均线多头加分
        df.loc[df['ma_trend'] == 1, 'buy_signal'] += 15
        # MACD 金叉加分
        df.loc[df['macd_cross'] == 1, 'buy_signal'] += 15
        # 布林带下轨附近加分
        df.loc[df['boll_position'] < 20, 'buy_signal'] += 15
        # 资金流入加分
        if 'money_flow' in df.columns:
            df.loc[df['money_flow'] > 5, 'buy_signal'] += 10
        df['buy_signal'] = df['buy_signal'].clip(0, 100).round(0)
        
        return df


# 向后兼容的 CryptoFetcher 类
class CryptoFetcher(BaseFetcher):
    """
    加密货币数据获取器（兼容 BaseFetcher 接口）
    
    内部使用 CryptoDataManager 管理多数据源
    """
    
    name = "CryptoFetcher"
    priority = 0
    
    def __init__(self):
        self._manager = CryptoDataManager()
    
    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """标准化符号为 BTC-USD 格式"""
        return BaseCryptoSource.normalize_symbol(symbol, "standard")
    
    @staticmethod
    def get_crypto_name(symbol: str) -> str:
        """获取加密货币中文名称"""
        symbol = symbol.upper()
        for key, name in CRYPTO_NAME_MAP.items():
            if key in symbol or symbol.replace('-USD', '').replace('USDT', '') in key:
                return name
        return symbol.replace('-USD', '')
    
    def get_realtime_quote(self, symbol: str) -> CryptoTicker:
        """获取实时行情"""
        return self._manager.get_realtime_price(symbol)
    
    def get_daily_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 60
    ) -> pd.DataFrame:
        """获取日K线数据"""
        return self._manager.get_daily_data(symbol, days)
    
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """BaseFetcher 接口实现"""
        return self._manager.get_daily_data(stock_code, 60)
    
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """BaseFetcher 接口实现"""
        return df  # 已经是标准格式


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    
    manager = CryptoDataManager()
    
    # 测试实时价格
    print("\n===== 测试实时价格 =====")
    try:
        ticker = manager.get_realtime_price("BTC")
        print(f"BTC 实时价格: ${ticker.price:,.2f}")
        print(f"24h涨跌: {ticker.change_24h:+.2f}%")
        print(f"数据源: {ticker.source}")
    except Exception as e:
        print(f"获取失败: {e}")
    
    # 测试K线数据
    print("\n===== 测试K线数据 =====")
    try:
        df = manager.get_daily_data("BTC", days=60)
        print(f"获取 {len(df)} 条数据")
        print(df.tail(5)[['date', 'close', 'ma20', 'ma60', 'volatility']])
    except Exception as e:
        print(f"获取失败: {e}")
