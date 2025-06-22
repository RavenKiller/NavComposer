import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import Counter
from tqdm import tqdm
import cv2
import json
from pathlib import Path
from natsort import natsorted
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchmetrics import Accuracy, F1Score
from torch.utils.tensorboard import SummaryWriter

from action.datasets import ActionDatasetForVO


class PinholeCamera:
    def __init__(
        self, width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0
    ):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = abs(k1) > 0.0000001
        self.d = [k1, k2, p1, p2, k3]


class SimpleVisualOdometry:
    def __init__(
        self,
        cam,
        action_path=None,
        feature="SIFT",
        matcher_type="FLANN",
        angle_threshold=7,
    ):
        self.cam = cam
        self.cur_R = None
        self.cur_t = None
        self.keypoints1 = []
        self.keypoints2 = []
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
        self.knn_k = 2
        self.feature = feature  # ORB or SIFT
        self.matcher_type = matcher_type  # FLANN or BF
        if feature == "SIFT":
            # Initiate SIFT detector
            self.detector = cv2.SIFT_create()
            if matcher_type == "BF":
                self.matcher = cv2.BFMatcher()
            else:
                # FLANN parameters
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)  # or pass empty dictionary
                self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif feature == "ORB":
            # Initiate ORB detector
            self.detector = cv2.ORB_create()
            if matcher_type == "BF":
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            else:
                # FLANN parameters
                FLANN_INDEX_LSH = 6
                index_params = dict(
                    algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=10,  # 20
                    multi_probe_level=1,
                )  # 2
                search_params = dict(checks=50)  # or pass empty dictionary
                self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        self.turn_angle = 15
        self.scale = 10.0
        self.angle_threshold = angle_threshold
        if action_path and os.path.exists(action_path):
            with open(action_path, "r") as f:
                self.actions = json.load(f)
            self.positions, _ = self.calc_pos(self.actions)
        else:
            self.actions = []
            self.positions = []

    @property
    def default_action(self):
        return 1

    def get_transform(self):
        return self.cur_R, self.cur_t

    def calc_pos(self, action):
        current_position = (0.0, 0.0)
        current_heading = 0
        res_position = [current_position]
        res_heading = [current_heading]
        for a in action:
            if a == 1:
                dx = 0.25 * np.cos(np.deg2rad(current_heading))
                dy = 0.25 * np.sin(np.deg2rad(current_heading))
                current_position = (current_position[0] + dx, current_position[1] + dy)
                # print(np.sqrt(dx*dx+dy*dy))
            elif a == 2:
                dw = self.turn_angle
                current_heading = current_heading + dw
                if current_heading >= 360:
                    current_heading = current_heading - 360
            elif a == 3:
                dw = -self.turn_angle
                current_heading = current_heading + dw
                if current_heading < 0:
                    current_heading = current_heading + 360
            res_position.append(current_position)
            res_heading.append(current_heading)
        return res_position, res_heading

    def getAbsoluteScale(self, frame_id):
        x_prev = self.positions[frame_id - 1][0] * self.scale
        y_prev = 0.0
        z_prev = self.positions[frame_id - 1][1] * self.scale

        x = self.positions[frame_id][0] * self.scale
        y = 0.0
        z = self.positions[frame_id][1] * self.scale
        return np.sqrt(
            (x - x_prev) * (x - x_prev)
            + (y - y_prev) * (y - y_prev)
            + (z - z_prev) * (z - z_prev)
        )

    def classify_motion(self, R, t):
        # beta>0 means objects move to right in the image, so the camera turn left
        # beta<0 means objects move to left in the image, so the camera turn right
        beta = np.rad2deg(-np.arcsin(R[2, 0]))
        if beta < -self.angle_threshold:
            return 3  # turn right
        elif beta > self.angle_threshold:
            return 2  # turn left
        else:
            return 1  # move forward

    def process_two_frames(self, img1, img2, frame_id=0):
        self.keypoints1 = []
        self.keypoints2 = []

        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        if len(kp1) < self.knn_k:
            # FAIL: no enought keypoints to run knn match
            return self.default_action
        kp2, des2 = self.detector.detectAndCompute(img2, None)
        if len(kp2) < self.knn_k:
            # FAIL: no enought keypoints to run knn match
            return self.default_action

        matches = self.matcher.knnMatch(des1, des2, k=self.knn_k)

        # ratio test as per Lowe's paper
        for i, pair in enumerate(matches):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.7 * n.distance:
                x1, y1 = kp1[m.queryIdx].pt
                x2, y2 = kp2[m.trainIdx].pt
                self.keypoints1.append([x1, y1])
                self.keypoints2.append([x2, y2])
        self.keypoints1 = np.ascontiguousarray(self.keypoints1)
        self.keypoints2 = np.ascontiguousarray(self.keypoints2)
        if len(self.keypoints1) < 5 or len(self.keypoints2) < 5:
            # FAIL: no enought keypoints to find essential matrix
            return self.default_action
        E, mask = cv2.findEssentialMat(
            self.keypoints1,
            self.keypoints2,
            # focal=self.focal,
            # pp=self.pp,
            cameraMatrix=self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        if E is None:
            # FAIL: no essential matrix
            return self.default_action

        if E.shape[0] != 3:
            # FAIL: no enought keypoints to find one essential matrix
            E = E[:3, :]
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(
            E,
            self.keypoints1,
            self.keypoints2,
            cameraMatrix=self.K,  # focal=self.focal, pp=self.pp
        )
        pred_action = self.classify_motion(self.cur_R, self.cur_t)
        return pred_action

    def __call__(self, batch):
        first_image = batch["first_image"]
        second_image = batch["second_image"]
        assert len(first_image) == len(second_image)
        action = batch.get("action", None)
        preds = []
        for i in range(len(first_image)):
            preds.append(
                self.process_two_frames(
                    cv2.cvtColor(
                        np.array(first_image[i]), cv2.COLOR_RGB2GRAY
                    ),  # PIL to cv2 image
                    cv2.cvtColor(np.array(second_image[i]), cv2.COLOR_RGB2GRAY),
                )
            )
        return preds

    def match_draw_two_frames(self, img1, img2):
        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)

        matches = self.matcher.knnMatch(des1, des2, k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i, pair in enumerate(matches):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matchesMask,
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
        plt.figure()
        plt.imshow(img3)
        plt.savefig("tmp.jpg")
        # plt.show()


from tools.registry import registry


@registry.register_action
class VOAction:
    def __init__(self, config):
        self.vo_config = config.action_config.vo_config
        self.angle_threshold = self.vo_config.angle_threshold
        self.post_process = self.vo_config.post_process

        self.cam = PinholeCamera(*(self.vo_config.camera_params))
        self.vo = SimpleVisualOdometry(
            self.cam,
            feature=self.vo_config.feature,
            matcher_type=self.vo_config.matcher_type,
            angle_threshold=self.vo_config.angle_threshold,
        )
        self.action_map = {
            0: "stop",
            1: "move forward",
            2: "turn left",
            3: "turn right",
        }

    def __call__(self, first_image, second_image):
        assert isinstance(first_image, list)
        assert isinstance(second_image, list)
        assert len(first_image) == len(second_image)
        batch = {"first_image": first_image, "second_image": second_image}
        preds = self.vo(batch)
        preds = np.array(preds)
        if self.post_process >= 2:
            # ABA -> AAA
            for i in range(1, len(preds) - 1):
                if (
                    preds[i - 1] == preds[i + 1]
                    and preds[i - 1] != preds[i]
                    # and preds[i - 1] == 1
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
        preds = [self.action_map[v] for v in list(preds)]
        return preds


def infer_one_epoch(model, loader, write_action=None, post_process=0):
    """Preds and actions are in [0,1,2,3]
    Args:
        post_process: Post process level. None or 0: no process, 1: three actions consistency, 2: majority voting
    """
    metric_acc = Accuracy(task="multiclass", num_classes=4)
    metric_f1 = F1Score(task="multiclass", num_classes=4)
    batch_bar = tqdm(loader)
    p = []
    a = []
    for batch in batch_bar:
        preds = model(batch)
        preds = torch.tensor(preds, dtype=int, device="cpu")
        preds = torch.nn.functional.pad(preds, (0, 1))  # Pad a stop action
        actions = torch.tensor(batch["action"], device="cpu")
        assert len(preds) == len(actions)
        if post_process >= 2:
            # ABA -> AAA
            for i in range(1, len(preds) - 1):
                if (
                    preds[i - 1] == preds[i + 1]
                    and preds[i - 1] == 1
                    and preds[i - 1] != preds[i]
                ):
                    preds[i] = preds[i - 1]
        if post_process >= 1:
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


def inference():
    ckpt_folder = Path("data/checkpoints")
    assert os.path.exists(ckpt_folder)
    fake_ckpt_path = ckpt_folder / "voaction" / "best.pth"
    ## Parameters
    # cam = PinholeCamera(640.0,480.0,320.0,240.0,320.0,240.0)
    cam = PinholeCamera(224.0, 224.0, 112.0, 112.0, 111.0, 111.0)
    model = SimpleVisualOdometry(cam, feature="SIFT", matcher_type="FLANN")

    merge_ds = ActionDatasetForVO(
        split=["val_seen", "val_unseen"],
        need_infer=True,
        folder="data/vlnce_traj_action",
    )
    merge_result = infer_one_epoch(
        model, merge_ds.get_episode_batch(), write_action="action_vo"
    )
    print("seen and unseen {}".format(merge_result))

    # unseen_ds = ActionDatasetForVO(split="val_unseen", need_infer=True, folder="data/vlnce_traj_action_hr")
    # unseen_result = infer_one_epoch(model, unseen_ds.get_episode_batch(), write_action="action_vo")
    # print("unseen {}".format(unseen_result))

    # train_ds = ActionDatasetForVO(split="train", need_infer=True, folder="data/vlnce_traj_action_hr")
    # train_result = infer_one_epoch(model, train_ds.get_episode_batch(), write_action="action_vo")
    # print("train {}".format(train_result))

    # res_file = str(fake_ckpt_path).replace(".pth", "_seen.json")
    # os.makedirs(fake_ckpt_path.parent, exist_ok=True)
    # with open(res_file, "w") as f:
    #     json.dump(seen_result, f)
    # res_file = str(fake_ckpt_path).replace(".pth", "_unseen.json")
    # with open(res_file, "w") as f:
    #     json.dump(unseen_result, f)


def single_test():
    data_folder = Path("data/vlntj-ce-clean/val_seen/0")
    cam = PinholeCamera(224.0, 224.0, 112.0, 112.0, 111.0, 111.0)
    vo = SimpleVisualOdometry(cam, feature="ORB")

    img1 = cv2.imread(
        "data/vlntj-ce-clean/val_seen/0/rgb/5.jpg",
        cv2.IMREAD_COLOR,
    )
    img2 = cv2.imread(
        "data/vlntj-ce-clean/val_seen/0/rgb/6.jpg",
        cv2.IMREAD_COLOR,
    )
    vo.match_draw_two_frames(img1, img2)
    print(vo.process_two_frames(img1, img2))
    # for img_id, img_path in enumerate(natsorted(list((data_folder/"rgb").glob("*.jpg")))):
    #     if img_id==0:
    #         old_img_path = img_path
    #         continue
    #     img1 = cv2.imread(str(old_img_path), cv2.IMREAD_COLOR)
    #     img2 = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    #     print(f"======{img_id}")
    #     res = vo.process_two_frames(img1, img2, img_id)
    #     # print(res)
    #     # print(vo.positions[img_id])
    #     # print(vo.cur_R, vo.cur_t)
    #     old_img_path = img_path


if __name__ == "__main__":
    inference()
    # single_test()
