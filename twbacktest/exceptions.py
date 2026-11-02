"""套件對外使用的例外型別。"""


class ProviderError(RuntimeError):
    """單一資料提供者無法提供可用資料。"""


class AllProvidersFailedError(RuntimeError):
    """所有資料提供者皆失敗，訊息包含各 Provider 的原因。"""
