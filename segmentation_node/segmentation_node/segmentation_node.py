import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch

import numpy as np
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, fcn_resnet50

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')

        # CUDA対応の確認
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")

        # パラメータの宣言と取得
        self.declare_parameter('model_type', 'lraspp')
        self.model_type = self.get_parameter('model_type').get_parameter_value().string_value

        # モデルの選択
        if self.model_type == 'lraspp':
            self.model = lraspp_mobilenet_v3_large(pretrained=True).to(self.device)
        elif self.model_type == 'fastscnn':
            self.model = fcn_resnet50(pretrained=True).to(self.device)
        else:
            self.get_logger().error("Invalid model type. Choose 'lraspp' or 'fastscnn'.")
            return

        self.model.eval()
        self.bridge = CvBridge()

        # サブスクライバとパブリッシャーの設定
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            10
        )

        self.publisher = self.create_publisher(Image, '/segmentation_output', 10)
        self.get_logger().info(f"SegmentationNode started with model: {self.model_type}")

    def listener_callback(self, msg):
        # 画像の受け取りと変換
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        input_tensor = self.preprocess(frame)

        # 推論処理
        with torch.no_grad():
            output = self.model(input_tensor)

        # セグメンテーションマップの後処理
        segmentation_map = self.postprocess(output)

        # ROS2で送信（カラーで送信）
        segmented_image_msg = self.bridge.cv2_to_imgmsg(segmentation_map, "bgr8")
        self.publisher.publish(segmented_image_msg)


    def preprocess(self, frame):
        # 前処理
        frame = cv2.resize(frame, (320, 320))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        # GPUに転送
        tensor = tensor.to(self.device)
        return tensor

    def postprocess(self, output):
        # 出力マップを単一チャンネルに変換
        segmentation_map = output['out'].argmax(dim=1).squeeze().cpu().numpy().astype('uint8')

        # カラーマップの作成 (例: 21クラス対応)
        color_map = np.array([
            [0, 0, 0],          # クラス0: 黒 (背景)
            [128, 0, 0],        # クラス1: 赤
            [0, 128, 0],        # クラス2: 緑
            [128, 128, 0],      # クラス3: 黄
            [0, 0, 128],        # クラス4: 青
            [128, 0, 128],      # クラス5: 紫
            [0, 128, 128],      # クラス6: 水色
            [128, 128, 128],    # クラス7: 灰色
            [64, 0, 0],         # クラス8: 暗い赤
            [192, 0, 0],        # クラス9: 明るい赤
            [64, 128, 0],       # クラス10: 暗い緑
            [192, 128, 0],      # クラス11: 明るい緑
            [64, 0, 128],       # クラス12: 暗い青
            [192, 0, 128],      # クラス13: 明るい青
            [64, 128, 128],     # クラス14: 暗い水色
            [192, 128, 128],    # クラス15: 明るい水色
            [0, 64, 0],         # クラス16: ダークグリーン
            [128, 64, 0],       # クラス17: ダークオレンジ
            [0, 192, 0],        # クラス18: ライトグリーン
            [128, 192, 0],      # クラス19: ライトオレンジ
            [0, 64, 128],       # クラス20: ダークブルー
        ])

        # セグメンテーションマップをカラーマップに変換
        color_mask = color_map[segmentation_map]
        return color_mask


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
