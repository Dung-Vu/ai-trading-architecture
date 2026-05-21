from src.memory import TradeMemory, TradeMemoryInterface


def test_trade_memory_implements_interface():
    assert issubclass(TradeMemory, TradeMemoryInterface)