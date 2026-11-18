"""纯 Actor ONNX 的 CPU 推理入口。

调用方用外置动作构造目标并维护观测历史，模型输入只有 obs。
固定批量为 1，多环境评估逐行执行，返回原始动作供仿真器缩放。
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import Tensor


def deployment_obs(history: Tensor, goal: Tensor) -> Tensor:
    """把 [B, H, D] 的帧主序历史和 [B, G] 的目标拼成 [B, H×D+G]。"""
    joints = (history.shape[-1] - 6) // 3
    terms = history.split((3, 3, joints, joints, joints), dim=-1)
    return torch.cat((*[term.flatten(1) for term in terms], goal), dim=-1)


class OnnxActor:
    def __init__(self, path: Path) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        metadata = self._session.get_modelmeta().custom_metadata_map
        try:
            self.history_length = int(metadata["history_length"])
            self.joint_names: tuple[str, ...] = tuple(json.loads(metadata["joint_names"]))
            self.iteration = int(metadata["iteration"])
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError("ONNX 缺少有效的 Actor 部署元数据") from error

        joints = len(self.joint_names)
        width = self.history_length * (6 + 3 * joints) + 10 * (2 * joints + 6)
        inputs = [(item.name, item.shape, item.type) for item in self._session.get_inputs()]
        outputs = [(item.name, item.shape, item.type) for item in self._session.get_outputs()]
        if inputs != [("obs", [1, width], "tensor(float)")] or outputs != [("actions", [1, joints], "tensor(float)")]:
            raise ValueError("ONNX 接口须为纯 Actor 的 obs 输入和 actions 输出")

    def __call__(self, history: Tensor, goal: Tensor) -> Tensor:
        """按环境推理，各行返回 [J] 的原始动作，拼回输入所在设备。"""
        observations = deployment_obs(history, goal).detach().cpu()
        actions = []
        for obs in observations.split(1):
            (result,) = self._session.run(["actions"], {"obs": obs.numpy()})
            if not isinstance(result, np.ndarray):
                raise ValueError("ONNX 的 actions 必须为张量")
            actions.append(torch.from_numpy(result))
        return torch.cat(actions, dim=0).to(history.device)
