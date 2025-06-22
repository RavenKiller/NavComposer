import os
import time
from pathlib import Path
from collections import deque
import json
from collections import Counter
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import transformers
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel

from tools.registry import registry
import torch.distributed as dist

from torchmetrics import Accuracy, F1Score
from torch.utils.tensorboard import SummaryWriter

from action.datasets import ActionDataset

from accelerate import Accelerator

try:
    accelerator = Accelerator()
except RuntimeError:
    accelerator = Accelerator(cpu=True)


#####################################################
# Model
#####################################################
class AttentionalPoolerWithMask(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        n_head: int = 8,
        n_queries: int = 2048,
        norm_layer: nn.LayerNorm = nn.LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(
            d_model, n_head, kdim=context_dim, vdim=context_dim
        )
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        if attn_mask is not None:
            if len(attn_mask.shape) == 2:
                attn_mask = attn_mask.unsqueeze(1).expand(-1, q.shape[0], -1)
            out = self.attn(
                q.unsqueeze(1).expand(-1, N, -1),
                x,
                x,
                need_weights=False,
                attn_mask=attn_mask,
            )[0]
        else:
            out = self.attn(q.unsqueeze(1).expand(-1, N, -1), x, x, need_weights=False)[
                0
            ]
        return out.permute(1, 0, 2)  # LND -> NLD


class RNActionClassifier(nn.Module):
    def __init__(self, model_name="microsoft/resnet-50", hidden_size=2048):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        if "clip" in model_name:
            self.vit = CLIPVisionModel.from_pretrained(model_name)
        else:
            self.vit = AutoModel.from_pretrained(model_name)
        self.feat_dropout = nn.Dropout(0.5)
        self.attn_pooler1 = AttentionalPoolerWithMask(
            d_model=hidden_size,
            context_dim=hidden_size,
            n_head=hidden_size // 64,
            n_queries=1,
        )
        self.attn_pooler2 = AttentionalPoolerWithMask(
            d_model=hidden_size,
            context_dim=hidden_size,
            n_head=hidden_size // 64,
            n_queries=1,
        )
        self.class_head = nn.Linear(hidden_size, 3)
        self.criterion = nn.CrossEntropyLoss()

        for param in self.vit.parameters():
            param.requires_grad_(False)

        for param in self.vit.encoder.stages[3].parameters():
            param.requires_grad_(True)
        for param in self.vit.pooler.parameters():
            param.requires_grad_(True)

    def get_image_features(self, image):
        return (
            self.vit(pixel_values=image)
            .last_hidden_state.flatten(start_dim=2)
            .permute((0, 2, 1))
        )

    def forward(self, batch):
        first_image = batch["first_image"]
        second_image = batch["second_image"]
        action = batch.get("action", None)

        first_features = self.feat_dropout(self.get_image_features(first_image))
        second_features = self.feat_dropout(self.get_image_features(second_image))
        attn_res1 = self.attn_pooler1(first_features).squeeze(1)
        attn_res2 = self.attn_pooler2(second_features).squeeze(1)
        logits = self.class_head(attn_res2 - attn_res1)

        if action is not None and len(logits) == len(action):
            loss = self.criterion(logits, action)
        else:
            loss = 0

        return {
            "logits": logits,
            "loss": loss,
        }


@registry.register_action
class RNAction:
    def __init__(
        self,
        config,
        # model_name="microsoft/resnet-50",
        # hidden_size=2048,
        # ckpt_path="data/checkpoints/actionclassifier/best.pth",
        # device="cuda:0",
    ):
        self.rn_config = config.action_config.rn_config
        self.model = RNActionClassifier(
            model_name=self.rn_config.model_name, hidden_size=self.rn_config.hidden_size
        )
        self.processor = AutoImageProcessor.from_pretrained(self.rn_config.model_name)
        self.post_process = self.rn_config.post_process
        if self.rn_config.ckpt_path:
            ckpt = torch.load(self.rn_config.ckpt_path, map_location="cpu")
            state_dict = {
                k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()
            }
            self.model.load_state_dict(state_dict)
        self.device = torch.device(dist.get_rank())
        self.model.to(self.device)
        self.model.eval()
        self.action_map = {
            0: "stop",
            1: "move forward",
            2: "turn left",
            3: "turn right",
        }

    def __call__(self, first_image, second_image, post_process=2):
        assert isinstance(first_image, list)
        assert isinstance(second_image, list)
        first_image = torch.cat(
            [
                self.processor(images=v, return_tensors="pt").pixel_values
                for v in first_image
            ],
            dim=0,
        )
        second_image = torch.cat(
            [
                self.processor(images=v, return_tensors="pt").pixel_values
                for v in second_image
            ],
            dim=0,
        )
        batch = {
            "first_image": first_image,
            "second_image": second_image,
        }
        with torch.no_grad():
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            res = self.model(batch)
            logits = res["logits"]
            preds = logits.argmax(dim=1).cpu() + 1  # Plus 1 to recover stop action
            if self.post_process >= 2:
                # ABA -> AAA
                for i in range(1, len(preds) - 1):
                    if (
                        preds[i - 1] == preds[i + 1]
                        and preds[i - 1] == 1
                        and preds[i - 1] != preds[i]
                    ):
                        preds[i] = preds[i - 1]
            if self.post_process >= 1:
                # AAB -> AAA
                i = 0
                while i < len(preds):
                    j = i + 1
                    if preds[i] == 2 or preds[i] == 3:
                        local_actions = Counter([preds[i].item()])
                        while j < len(preds) and (preds[j] == 2 or preds[j] == 3):
                            local_actions[preds[j].item()] += 1
                            j += 1
                        local_actions = local_actions.most_common(2)
                        if (
                            len(local_actions) > 1
                            and local_actions[0][1] >= local_actions[1][1]
                        ):
                            preds[i:j] = local_actions[0][0]
                    i = j
            preds = [self.action_map[v] for v in preds.tolist()]
        return preds


#####################################################
# Epoch functions
#####################################################
def train_one_epoch(model, optimizer, loader, scheduler, writer, iter_num=0):
    model.train()
    if accelerator.is_main_process:
        batch_bar = tqdm(loader)
        smooth_loss = deque(maxlen=16)
    else:
        batch_bar = loader

    for batch in batch_bar:
        optimizer.zero_grad()
        # batch = {k:v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        res = model(batch)
        loss = res["loss"]
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        if accelerator.is_main_process:
            smooth_loss.append(loss.item())
            batch_bar.set_postfix({"loss": np.mean(smooth_loss)})
            writer.add_scalar("loss", loss.item(), iter_num)
        iter_num += 1
    return iter_num


def evaluate_one_epoch(model, loader):
    """Preds and actions are in [0,1,2]"""
    model.eval()
    metric_acc = Accuracy(task="multiclass", num_classes=3)
    metric_f1 = F1Score(task="multiclass", num_classes=3)
    p = []
    a = []
    if accelerator.is_main_process:
        batch_bar = tqdm(loader)
    else:
        batch_bar = loader
    with torch.no_grad():
        for batch in batch_bar:
            # batch = {k:v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            res = model(batch)
            logits = res["logits"]
            preds = logits.argmax(dim=1)
            actions = batch["action"]
            gathered_preds = accelerator.gather_for_metrics(preds)
            gathered_actions = accelerator.gather_for_metrics(actions)
            p.append(gathered_preds.cpu())
            a.append(gathered_actions.cpu())
    p = torch.cat(p, dim=0)
    a = torch.cat(a, dim=0)
    return {
        "accuracy": metric_acc(p, a).item(),
        "f1score": metric_f1(p, a).item(),
    }


def infer_one_epoch(model, loader, write_action=None, post_process=0):
    """Preds and actions are in [0,1,2,3]
    Args:
        post_process: Post process level. None or 0: no process, 1: three actions consistency, 2: majority voting
    """
    device = accelerator.device
    model.eval()
    metric_acc = Accuracy(task="multiclass", num_classes=4)
    metric_f1 = F1Score(task="multiclass", num_classes=4)
    device = accelerator.device
    if accelerator.is_main_process:
        batch_bar = tqdm(loader)
    else:
        batch_bar = loader
    p = []
    a = []
    with torch.no_grad():
        for batch in batch_bar:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            res = model(batch)
            logits = res["logits"]
            preds = logits.argmax(dim=1) + 1  # Plus 1 to recover stop action
            preds = nn.functional.pad(preds, (0, 1)).cpu()  # Pad a stop action
            actions = batch["action"].cpu()
            assert len(preds) == len(actions)
            if post_process == 1:
                for i in range(1, len(preds) - 1):
                    if (
                        preds[i - 1] == preds[i + 1]
                        and (preds[i - 1] == 2 or preds[i - 1] == 3)
                        and preds[i - 1] != preds[i]
                    ):
                        preds[i] = preds[i - 1]
            elif post_process == 2:
                i = 0
                while i < len(preds):
                    j = i + 1
                    if preds[i] == 2 or preds[i] == 3:
                        local_actions = Counter([preds[i].item()])
                        while preds[j] < len(preds) and (
                            preds[j] == 2 or preds[j] == 3
                        ):
                            local_actions[preds[j].item()] += 1
                            j += 1
                        local_actions = local_actions.most_common(2)
                        if (
                            len(local_actions) > 1
                            and local_actions[0][1] > local_actions[1][1]
                        ):
                            preds[i:j] = local_actions[0][0]
                    i = j
            if write_action:
                episode = batch["episode"]
                action_path = Path(episode) / write_action / "0.json"
                os.makedirs(action_path.parent, exist_ok=True)
                with open(action_path, "w") as f:
                    json.dump(preds.tolist(), f)
            p.append(preds)
            a.append(actions)
    p = torch.cat(p, dim=0)
    a = torch.cat(a, dim=0)
    return {
        "accuracy": metric_acc(p, a).item(),
        "f1score": metric_f1(p, a).item(),
    }


#####################################################
# Main functions
#####################################################
def train_and_evaluate():
    ## Parameters
    model_name = "microsoft/resnet-50"
    device = accelerator.device
    epochs = 10
    batch_size = 64
    lr = 1e-5
    ckpt_folder = Path("data/checkpoints")
    assert os.path.exists(ckpt_folder)
    ckpt_path = (
        ckpt_folder
        / "actionclassifier"
        / "best_{}.pth".format(model_name.replace("/", "_"))
    )
    os.makedirs(ckpt_path.parent, exist_ok=True)
    ckpt_old = None
    if os.path.exists(ckpt_path):
        ckpt_old = torch.load(ckpt_path, map_location="cpu")

    writer = SummaryWriter(
        "data/tensorboards/acctionclassifier/{}{}".format(
            model_name.replace("/", "_"),
            time.strftime("%y%m%d-%H%M%S", time.localtime()),
        )
    )

    model = RNActionClassifier(model_name=model_name, hidden_size=2048)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_ds = ActionDataset(
        model_name=model_name, split=["train"], color_aug=True, action_aug=False
    )
    # print(len(train_ds))

    train_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=True, batch_size=batch_size
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=epochs * len(train_loader)
    )

    seen_ds = ActionDataset(model_name=model_name, split="val_seen")
    seen_loader = torch.utils.data.DataLoader(
        seen_ds, shuffle=False, batch_size=batch_size
    )

    unseen_ds = ActionDataset(model_name=model_name, split="val_unseen")
    unseen_loader = torch.utils.data.DataLoader(
        unseen_ds, shuffle=False, batch_size=batch_size
    )

    (
        model,
        optimizer,
        train_loader,
        seen_loader,
        unseen_loader,
        scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_loader, seen_loader, unseen_loader, scheduler
    )
    iter_num = 0
    best_f1 = 0
    if ckpt_old:
        best_f1 = ckpt_old["unseen_result"]["f1score"]
    for epoch in range(epochs):
        iter_num = train_one_epoch(
            model, optimizer, train_loader, scheduler, writer, iter_num
        )
        seen_result = evaluate_one_epoch(model, seen_loader)
        unseen_result = evaluate_one_epoch(model, unseen_loader)
        if accelerator.is_main_process:
            print("Epoch {}: seen {}".format(epoch, seen_result))
            print("Epoch {}: unseen {}".format(epoch, unseen_result))
            writer.add_scalar("seen_accuracy", seen_result["accuracy"], iter_num)
            writer.add_scalar("unseen_accuracy", unseen_result["accuracy"], iter_num)

            if unseen_result["f1score"] > best_f1:
                best_f1 = unseen_result["f1score"]
                torch.save(
                    {
                        "state_dict": accelerator.unwrap_model(model).state_dict(),
                        "seen_result": seen_result,
                        "unseen_result": unseen_result,
                    },
                    ckpt_path,
                )


def inference():
    ## Parameters
    model_name = "microsoft/resnet-50"
    device = accelerator.device
    epochs = 10
    batch_size = 64
    lr = 1e-5
    ckpt_folder = Path("data/checkpoints")
    assert os.path.exists(ckpt_folder)
    ckpt_path = (
        ckpt_folder
        / "actionclassifier"
        / "best_{}.pth".format(model_name.replace("/", "_"))
    )

    model = RNActionClassifier(model_name=model_name, hidden_size=2048)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["state_dict"])

    # seen_ds = ActionDataset(
    #     folder="data/vlntj-ce-clean",
    #     model_name=model_name,
    #     split="val_seen",
    #     need_infer=True,
    # )
    # unseen_ds = ActionDataset(
    #     model_name=model_name, split="val_unseen", need_infer=True
    # )
    # model = accelerator.prepare(model)
    # seen_result = infer_one_epoch(model, seen_ds.get_episode_batch())
    # unseen_result = infer_one_epoch(model, unseen_ds.get_episode_batch())
    # if accelerator.is_main_process:
    #     print("seen {}".format(seen_result))
    #     print("unseen {}".format(unseen_result))
    #     res_file = str(ckpt_path).replace(".pth", "_seen.json")
    #     with open(res_file, "w") as f:
    #         json.dump(seen_result, f)
    #     res_file = str(ckpt_path).replace(".pth", "_unseen.json")
    #     with open(res_file, "w") as f:
    #         json.dump(unseen_result, f)

    merge_ds = ActionDataset(
        model_name=model_name, split=["val_seen", "val_unseen"], need_infer=True
    )
    model = accelerator.prepare(model)
    merge_result = infer_one_epoch(model, merge_ds.get_episode_batch())
    if accelerator.is_main_process:
        print("seen and unseen {}".format(merge_result))
        res_file = str(ckpt_path).replace(".pth", "_merge.json")
        with open(res_file, "w") as f:
            json.dump(merge_result, f)


#####################################################
# Test
#####################################################

if __name__ == "__main__":
    train_and_evaluate()
    # inference()
