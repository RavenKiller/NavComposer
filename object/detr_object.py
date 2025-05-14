from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests
import torch.distributed as dist
from tools.registry import registry


class DETRObjectDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "facebook/detr-resnet-50"
        self.detector = DetrForObjectDetection.from_pretrained(
            self.model_name, revision="no_timm"
        )

    def forward(self, batch):
        images = batch["images"]
        return self.detector(pixel_values=images)


@registry.register_object
class DETRObject:
    def __init__(self, config):
        self.detr_config = config.object_config.detr_config
        self.model_name = self.detr_config.model_name
        self.device = torch.device(dist.get_rank())
        self.processor = DetrImageProcessor.from_pretrained(
            self.model_name, revision="no_timm"
        )
        self.model = DETRObjectDetector().to(self.device)
        self.id2label = self.model.detector.config.id2label
        self.max_num = 3

    def describe_position(self, w, h, bx, by):
        """Nine-grid splits"""
        dx = w // 3
        dy = h // 3
        col = min(max(int(bx // dx), 0), 2)
        row = min(max(int(by // dy), 0), 2)

        grid_positions = [
            ["far left side", "far front", "far right side"],
            ["left side", "front", "right side"],
            ["left side and very close", "very close", "right side and very close"],
        ]
        return grid_positions[row][col]

    def __call__(self, images):
        pixel_values = torch.cat(
            [
                self.processor(images=v, return_tensors="pt").pixel_values
                for v in images
            ],
            dim=0,
        )
        batch = {"images": pixel_values}
        with torch.no_grad():
            batch = {
                k: v.to(self.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            outputs = self.model(batch)
            target_sizes = torch.tensor([image.size[::-1] for image in images]).to(
                self.device
            )
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5
            )
            objects = []
            for i, result in enumerate(results):
                object_str = ""
                if not len(result["scores"]):
                    object_str = "No object."
                else:
                    idx = torch.argsort(result["scores"], descending=True)
                    scores = result["scores"][idx]
                    labels = result["labels"][idx]
                    boxes = result["boxes"][idx]

                    scores = scores[: self.max_num]
                    labels = labels[: self.max_num]
                    boxes = boxes[: self.max_num]

                    for j in range(len(scores)):
                        object_name = self.id2label[labels[j].item()]
                        box_center_x = (boxes[j][0].item() + boxes[j][2].item()) // 2
                        box_center_y = (boxes[j][1].item() + boxes[j][3].item()) // 2

                        object_position = self.describe_position(
                            images[i].size[0],
                            images[i].size[1],
                            box_center_x,
                            box_center_y,
                        )
                        object_str += object_name + ": " + object_position + ". "
                objects.append(object_str)
        return objects
