import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, fcn_resnet50

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')

        # パラメータの宣言と取得
        self.declare_parameter('model_type', 'lraspp')
        self.model_type = self.get_parameter('model_type').get_parameter_value().string_value

        # モデルの選択
        if self.model_type == 'lraspp':
            self.model = lraspp_mobilenet_v3_large(pretrained=True)
        elif self.model_type == 'fastscnn':
            self.model = fcn_resnet50(pretrained=True)
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

        # ROS2で送信
        segmented_image_msg = self.bridge.cv2_to_imgmsg(segmentation_map, "mono8")
        self.publisher.publish(segmented_image_msg)

    def preprocess(self, frame):
        # 前処理
        frame = cv2.resize(frame, (320, 320))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return tensor

    def postprocess(self, output):
        # 出力マップを単一チャンネルに変換
        segmentation_map = output['out'].argmax(dim=1).squeeze().cpu().numpy().astype('uint8')
        return segmentation_map

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
