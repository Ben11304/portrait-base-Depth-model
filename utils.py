import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import numpy as np
import pydensecrf.densecrf as dcrf
def get_all_image_files(root_folder):
    # Các định dạng ảnh thường gặp
    image_extensions = {'.jpg', '.jpeg', '.HEIC','.JPG'}
    image_files = []
    
    # Sử dụng os.walk để đệ quy qua tất cả các thư mục và file
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            # Kiểm tra xem file có phải là file ảnh không
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_files.append(os.path.join(dirpath, filename))
    return image_files

def apply_crf(image, depth):
    h, w = image.shape[:2]
    d = dcrf.DenseCRF2D(w, h, 2)

    # Chuẩn bị U và pairwise
    U = np.stack([depth.flatten(), 1 - depth.flatten()], axis=0)
    U = -np.log(U + 1e-6)
    U = U.reshape((2, h * w))
    d.setUnaryEnergy(U)

    # Thêm tính năng màu sắc
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=image, compat=10)

    # Thực hiện suy luận
    Q = d.inference(5)
    depth_crf = np.argmax(Q, axis=0).reshape((h, w))

    return depth_crf.astype('float32')

def load_model():

    # Khởi tạo mô hình
    model = smp.UnetPlusPlus(
        encoder_name='resnet50',        # Backbone encoder      # Pretrained weights
        in_channels=3,                   # Số kênh đầu vào (3 cho ảnh RGB)
        classes=1                        # Số kênh đầu ra (1 cho depth map)
    )

    # Load checkpoint
    checkpoint = torch.load('/Users/mac/Dev/Computer-vision/Manual-Brokeh/model_prepair/U++.pt', map_location=torch.device('cpu'))
    state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k.replace("model.", "")
        else:
            new_key = k
        new_state_dict[new_key] = v

    # Load state_dict đã được chỉnh sửa vào mô hình
    model.load_state_dict(new_state_dict)

    print("Checkpoint loaded successfully!")
def depth_map(img, model, device="cpu"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = model(img_rgb)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    # Chuẩn hóa bản đồ độ sâu
    depth_min = output.min()
    depth_max = output.max()
    depth_map = (output - depth_min) / (depth_max - depth_min)
    depth_map = 1.0 - depth_map 
    return depth_map


def depth_transform(depth_map, focus_point):
    depth_map_transform = abs(depth_map - focus_point)
    depth_map_transform /= depth_map_transform.max()  # Chuẩn hóa về [0, 1]
    return depth_map_transform

def blur(depth_map, image, num_levels=5, focus_point=0.5, max_kernel_size=71):
    depth_map = depth_transform(depth_map, focus_point)
    # Tạo danh sách các ngưỡng
    levels = np.linspace(0, 1, num_levels + 1)
    # Tạo danh sách lưu các mặt nạ và độ mờ tương ứng
    masks = []
    blur_values = []

    for i in range(num_levels):
        # Xác định mặt nạ cho mức hiện tại
        lower = levels[i]
        upper = levels[i + 1]
        mask = cv2.inRange(depth_map, lower, upper)
        masks.append(mask)
        
        # Xác định giá trị độ mờ cho mức hiện tại (kernel size)
        kernel_size = int(1 + (max_kernel_size - 1) * (i / (num_levels - 1)))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Đảm bảo kernel_size là số lẻ
        blur_values.append(kernel_size)

    blurred_images = []

    for kernel_size in blur_values:
        # Áp dụng Gaussian blur với kernel_size
        if kernel_size >= 3:
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            blurred = image.copy()
        blurred_images.append(blurred)

    # Khởi tạo ảnh kết quả với giá trị 0
    result = np.zeros_like(image)

    for mask, blurred in zip(masks, blurred_images):
        # Tạo mặt nạ 3 kênh
        mask_3ch = cv2.merge([mask, mask, mask])
        
        # Áp dụng mặt nạ lên ảnh làm mờ và thêm vào ảnh kết quả
        masked_blur = cv2.bitwise_and(blurred, mask_3ch)
        result = cv2.add(result, masked_blur)

    # Tạo mặt nạ tổng hợp từ các mặt nạ đã tạo
    combined_mask = np.zeros_like(depth_map, dtype=np.uint8)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Tạo mặt nạ nghịch đảo cho vùng không được làm mờ
    inverse_mask = cv2.bitwise_not(combined_mask)
    inverse_mask_3ch = cv2.merge([inverse_mask, inverse_mask, inverse_mask])

    # Áp dụng mặt nạ nghịch đảo lên ảnh gốc và thêm vào kết quả
    masked_original = cv2.bitwise_and(image, inverse_mask_3ch)
    result = cv2.add(result, masked_original)

    return result
