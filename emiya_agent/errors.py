"""异常层级。"""


class EmiyaError(Exception):
    """所有 EmiyaAgent 异常的基类。"""


class ConfigError(EmiyaError):
    """配置缺失或非法（如 API key 未设置、路由格式错误）。"""


class ProviderError(EmiyaError):
    """模型 API 调用失败（网络、鉴权、限流等），面向用户展示 message。"""


class ToolError(EmiyaError):
    """工具执行失败。"""


class UserInterrupt(EmiyaError):
    """用户主动中断当前回合。"""
