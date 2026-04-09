from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

from adapters import model_provider


class FakeModel:
    def __init__(self) -> None:
        self.captured_messages = None

    def invoke(self, messages):
        self.captured_messages = messages
        return SimpleNamespace(content="这是一张界面截图，包含最近项目文件列表。")


def test_summarize_visual_assets_uses_summary_prompt_and_image_only_message(
    monkeypatch,
) -> None:
    fake_model = FakeModel()
    monkeypatch.setattr(model_provider, "create_chat_model", lambda: fake_model)
    monkeypatch.setattr(model_provider, "_load_summary_prompt", lambda: "你是视觉摘要助手。")

    asset = SimpleNamespace(image_bytes=b"fake-image-bytes", mime_type="image/png")

    result = model_provider.summarize_visual_assets(
        (asset,),
        depth="L2",
        file_name="report.png",
        file_type="png",
    )

    assert result.startswith("由视觉模型查看生成：")
    assert "界面截图" in result
    assert fake_model.captured_messages is not None
    assert len(fake_model.captured_messages) == 2

    system_message, human_message = fake_model.captured_messages
    assert isinstance(system_message, SystemMessage)
    assert "你是视觉摘要助手。" in system_message.content
    assert "当前文件名：report.png" in system_message.content
    assert "当前文件类型：png" in system_message.content
    assert "当前预览深度：L2" in system_message.content

    assert isinstance(human_message, HumanMessage)
    assert isinstance(human_message.content, list)
    assert len(human_message.content) == 1
    assert human_message.content[0]["type"] == "image_url"
    assert human_message.content[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_summarize_visual_assets_wraps_model_errors(monkeypatch) -> None:
    class RaisingModel:
        def invoke(self, messages):
            del messages
            raise RuntimeError("model backend unavailable")

    monkeypatch.setattr(model_provider, "create_chat_model", lambda: RaisingModel())
    monkeypatch.setattr(model_provider, "_load_summary_prompt", lambda: "你是视觉摘要助手。")

    asset = SimpleNamespace(image_bytes=b"fake-image-bytes", mime_type="image/png")

    try:
        model_provider.summarize_visual_assets(
            (asset,),
            depth="L1",
            file_name="chart.png",
            file_type="png",
        )
    except ValueError as exc:
        assert "视觉摘要调用失败" in str(exc)
        assert "model backend unavailable" in str(exc)
    else:
        raise AssertionError("expected ValueError")
