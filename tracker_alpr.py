from ultralytics import YOLO
import cv2
import os
import time
import numpy as np 
from paddleocr import PaddleOCR
from collections import defaultdict
from tracker import Tracker
from collections import Counter

import datetime


ocr = PaddleOCR(lang='en',rec_algorithm='CRNN')
# Load a model
model = YOLO("alpr_yolov8n_8000img_100epochs.pt") 


def perform_ocr(image):
    ocr_res = ocr.ocr(image, cls=False, det=False)
    return ocr_res

def rotate_and_split_license_plate(image):
    # Chuyển đổi ảnh sang độ xám để dễ dàng xử lý
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tìm các đường viền trong ảnh
    edges = cv2.Canny(gray, 50, 150, apertureSize=3) # dùng kernel 3x3 để phát hiện cạnh

    # Tìm các đoạn thẳng trong ảnh
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    # np.pi / 180: Giá trị tối thiểu cho góc theta (trong đơn vị radians)
    # threshold=100: Đây là ngưỡng (threshold) cho việc xem xét một đường thẳng.
    # Nếu có nhiều đường thẳng có cùng một góc và tương phản (sự tương quan) cao,
    # chỉ những đường thẳng có tương phản cao hơn ngưỡng này mới được xem xét là một đường thẳng hợp lệ.
    # Khi giá trị threshold thấp, nhiều đường thẳng sẽ được phát hiện;
    # khi nó cao, chỉ các đường thẳng rất rõ ràng mới được xem xét
    # chạy thử đoạn dưới đây để xem chi tiết
    # cv2.imshow("edges", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Kiểm tra xem lines có giá trị hợp lệ hay không
    if lines is not None:
        # Tính toán góc xoay trung bình của các đoạn thẳng
        total_angle = 0
        count = 0

        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # Tính góc xoay của đoạn thẳng
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            # Loại bỏ các đoạn thẳng gần vuông (có góc nhỏ hơn một ngưỡng)
            if abs(angle) > 10 and abs(angle) < 80:
                total_angle += angle
                count += 1

        # Tính góc xoay trung bình
        if count > 0:
            average_angle = total_angle / count
        else:
            average_angle = 0

        # Xoay lại ảnh để biển số xe nằm ngang
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, average_angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        # Tính toán điểm chia ảnh thành hai phần trên và dưới
        split_point = height // 2

        # Tạo hai phần ảnh con từ ảnh rotated_image
        upper_part = rotated_image[:split_point, :]
        lower_part = rotated_image[split_point:, :]

        print("Đã tìm thấy đoạn thẳng.")
        print("angle:", angle)
        print("rotate angle:", average_angle)
        return upper_part, lower_part
    else:
        print("Không tìm thấy đoạn thẳng.")
        return None, None


def test_vid_yolov8(vid_dir, out_path):
    # Declaring variables for video processing.
    cap = cv2.VideoCapture(vid_dir)
    # Get the total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # file_name = os.path.join(out_path, 'out_' + vid_dir.split('/')[-1] + )
    # out = cv2.VideoWriter(file_name, codec, fps, (width, height))
    out = cv2.VideoWriter(out_path, codec, fps, (width, height))

    # Frame count variable.
    ct = 0
    
    # Reading video frame by frame.
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            print(str(ct) + "/" + str(total_frames))

            # Noting time for calculating FPS.
            prev_time = time.time()

            # Use the model
            results = model(img)  # predict on an image
            img_tmp = img
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    # Get the bounding boxes and image for each result
                    bboxes = result.boxes[0].cpu().numpy()
                    # img = cv2.imread(input)  # Đảm bảo bạn đã định nghĩa biến 'input'
            
                    # Loop through bboxes and apply OCR, then draw on the image
                    
                    try:
                        for bbox in bboxes:
                            xyxy = bbox.xyxy
                            x1, y1, x2, y2 = xyxy[0]
                
                            # Kiểm tra xem biển số xe có gần vuông không (ví dụ: tỷ lệ 1:1)
                            width = x2 - x1
                            height = y2 - y1
                            aspect_ratio = width / height
                            print("aspect_ratio:", aspect_ratio)
                
                            if 0 <= aspect_ratio <= 1.5:
                                # Cắt và xoay lại biển số xe --------------
                                image_upper, image_lower = rotate_and_split_license_plate(
                                    img[int(y1):int(y2), int(x1):int(x2)])

                                if image_upper is None and image_lower is None:
                                    # Biển số xe gần vuông hoặc hình vuông
                    
                                    # Tính toán điểm chia ảnh thành hai phần trên và dưới
                                    split_point = y1 + (y2 - y1) // 2

                                    
                                    # Tạo hai phần ảnh con từ ảnh cr_img
                                    upper_part = img[int(y1):int(split_point), int(x1):int(x2)]
                                    lower_part = img[int(split_point):int(y2), int(x1):int(x2)]
                                
                                    image_upper=cv2.resize(upper_part,(int(width),int(height/2)))
                                    image_lower=cv2.resize(lower_part,(int(width),int(height/2)))

                                image_collage_horizontal =np.hstack([image_upper, image_lower])
                                image_filename = str(ct) + ".jpg"  # Tên file ảnh đầu ra
                                cv2.imwrite("results/crop_image/" + image_filename, image_collage_horizontal)
                
                                # Tiếp tục xử lý ảnh chữ nhật cr_img ở đây
                                ocr_res = perform_ocr(image_collage_horizontal)
                
                                # Lưu hai phần ảnh
                                # cv2.imwrite("results/upper_part.jpg", upper_part)
                                # cv2.imwrite("results/" + str(ct) + ".jpg", lower_part)
                            else:
                                # Biển số xe không gần vuông
                                # Xử lý ảnh bình thường ở đây (cr_img = img[int(y1):int(y2), int(x1):int(x2)])
                                cr_img = img[int(y1):int(y2), int(x1):int(x2)]
                                image_filename = str(ct) + ".jpg"  # Tên file ảnh đầu ra
                                cv2.imwrite("results/crop_image/" + image_filename, cr_img)
                                ocr_res = perform_ocr(cr_img)
                            recognized_text = ocr_res[0][0][0] if ocr_res else "No Text"
                            print("recognized_text: ",recognized_text)
                            
                            ocr_conf = ocr_res[0][0][1] if ocr_res else "No Conf"
                            ocr_conf = round(ocr_conf,3)
                            print("ocr_conf: ",ocr_conf)
                            # Draw on the image
                            
                            cv2.rectangle(img_tmp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red bounding box
                            cv2.putText(img_tmp, recognized_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            cv2.putText(img_tmp, str(ocr_conf), (int(x1) + 150, int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    except Exception as e:
                        print("----------------------------------------")
                        print("Error:", e)
                        print("----------------------------------------")
                        continue
                else:
                    # If there are no bounding boxes or the result is empty, skip this result
                    continue

            # Calculating time taken and FPS for the whole process.
            tot_time = time.time() - prev_time
            fps = round(1 / tot_time,2)

            # Writing information onto the frame and saving it to be processed in a video.
            cv2.putText(img_tmp, 'frame: %d fps: %s' % (ct, fps),
                        (0, int(100 * 1)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
            out.write(img_tmp)

            ct = ct + 1
        else:
            break

def validate_plate(str_plate):

    regex_pattern = r'^\d{2}(-[A-HK-PR-Z]{2}|-[A-HK-PR-Z]{1}\d{1}|[A-HK-PR-Z]{1})[- ]?((?!000\.00)([0-9]\d{0,2}|0)\.\d{2}|[0-9]{4})$'  # [- ]?

    if re.match(regex_pattern, str_plate):
        return str_plate
    else:
        return None

def save_info_file(name_folder, name_file, data):
    # Kiểm tra nếu thư mục không tồn tại thì tạo mới
    if not os.path.exists(name_folder):
        os.makedirs(name_folder)

    # Tạo đường dẫn đầy đủ đến tệp tin
    file_path = os.path.join(name_folder, f"{name_file}.txt")

    with open(file_path, 'a') as file:
        
        # Thêm dữ liệu vào file
        for entry in data:
            file.write(f"{entry['Track_id']}\t{entry['Recognized_text']}\t{entry['Confidence']}\t{entry['Time']}\n")
            # file.write(f"{entry['Track_id']}\t{entry['Recognized_text']}\t{entry['Confidence']}\n")

    print("Dữ liệu đã được thêm vào file.")


def get_best_ocr(data, track_ids):
    counter_dict = Counter((item['track_id'], item['ocr_txt']) for item in data)

    most_common_recognized_text = {}
    rec_conf = ""
    ocr_res = ""
    for item in data:
        track_id = item['track_id']
        recognized_text = item['ocr_txt']
        confidence = item['ocr_conf']
        count = counter_dict[(track_id, recognized_text)]

        current_count, current_confidence, current_text = most_common_recognized_text.get(track_id, (0, 0, ''))

        if count > current_count or (count == current_count and confidence > current_confidence):
            most_common_recognized_text[track_id] = (count, confidence, recognized_text)

    if track_ids in most_common_recognized_text:
        rec_conf, ocr_res = most_common_recognized_text[track_ids][1], most_common_recognized_text[track_ids][2]

    return rec_conf, ocr_res

def tracker_test_vid_with_deep_sort(vid_dir,out_path):
    # Declaring variables for video processing.
    cap = cv2.VideoCapture(vid_dir)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(out_path, codec, fps, (width, height))

    tracker = Tracker()
    preds = []
    # Initializing some helper variables.
    ct = 0
    preds = []
    total_obj = 0
    CONFIDENCE_THRESHOLD = 0.5
    track_id_flag = 0
    # Reading video frame by frame.
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:

            h, w = img.shape[:2]
            overlay_img = img.copy()
            prev_time = time.time()
            detections = model(img)[0]
            # detections = model.track(img, persist=True)
            
            results = []

            datas = detections.boxes.data.tolist()

            
            for data in datas:
                # Lấy ra conf của object detection
                confidence = data[4]
                
                if float(confidence) > CONFIDENCE_THRESHOLD:
                    # continue

                    # get the bounding box and the class id
                    xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                    print("Toạ do 1: ",xmin, ymin, xmax, ymax)
                    class_id = int(data[5])

                    # Thêm bounding box(x, y, w, h) và conf vào list
                    results.append([xmin, ymin, xmax, ymax, confidence])
            # Update các info vào tracker 
            tracker.update(img, results)
            # loop over the tracks
            for detection in detections:
                if detection.boxes is not None and len(detection.boxes) > 0:
                    try:
                        for track in tracker.tracks:

                        #Lấy track_id và bounding box
                            track_id = track.track_id
                            x1, y1, x2, y2 = track.bbox
                            
                            width = x2 - x1
                            height = y2 - y1
                            aspect_ratio = width / height

                            if 0 <= aspect_ratio <= 1.5:
                                # Cắt và xoay lại biển số xe --------------
                                image_upper, image_lower = rotate_and_split_license_plate(img[int(y1):int(y2), int(x1):int(x2)])

                                if image_upper is None and image_lower is None:
                                    # Biển số xe gần vuông hoặc hình vuông

                                    # Tính toán điểm chia ảnh thành hai phần trên và dưới
                                    split_point = y1 + (y2 - y1) // 2

                                    # Tạo hai phần ảnh con từ ảnh cr_img
                                    upper_part = img[int(y1):int(split_point), int(x1):int(x2)]
                                    lower_part = img[int(split_point):int(y2), int(x1):int(x2)]

                                    image_upper=cv2.resize(upper_part,(int(width),int(height/2)))
                                    image_lower=cv2.resize(lower_part,(int(width),int(height/2)))

                                image_collage_horizontal =np.hstack([image_upper, image_lower])
                                image_filename = str(ct) + ".jpg"  # Tên file ảnh đầu ra
                                cv2.imwrite("results/crop_image/" + image_filename, image_collage_horizontal)

                                # Tiếp tục xử lý ảnh chữ nhật cr_img ở đây
                                ocr_res = perform_ocr(image_collage_horizontal)
                                print('text:', ocr_res)

                                # Lưu hai phần ảnh
                                # cv2.imwrite("results/upper_part.jpg", upper_part)
                                # cv2.imwrite("results/" + str(ct) + ".jpg", lower_part)
                            else:
                                # Biển số xe không gần vuông
                                # Xử lý ảnh bình thường ở đây (cr_img = img[int(y1):int(y2), int(x1):int(x2)])
                                cr_img = img[int(y1):int(y2), int(x1):int(x2)]
                                image_filename = str(ct) + ".jpg"  # Tên file ảnh đầu ra
                                cv2.imwrite("results/crop_image/" + image_filename, cr_img)
                                ocr_res = perform_ocr(cr_img)
                                print('text:', ocr_res)

                            print('-------------------')
                            # Draw bounding box and put text
                            recognized_text = ocr_res[0][0][0] if ocr_res else "No Text"
                            print("recognized_text: ",recognized_text)

                            ocr_conf = ocr_res[0][0][1] if ocr_res else "No Conf"
                            ocr_conf = round(ocr_conf,3)
                            print("ocr_conf: ",ocr_conf)

                            output_frame = {"track_id": track_id, "ocr_txt": recognized_text, "ocr_conf": ocr_conf}

                            # Thêm track_id vào list, nếu nó không tồn tại trong list.
                            if track_id not in list(set(ele['track_id'] for ele in preds)):
                                total_obj = total_obj + 1
                                preds.append(output_frame)
                            
                            else:
                                # Tìm kiếm track hiện tại trong list và update conf cao nhất của nó
                                preds, rec_conf, ocr_resc = get_best_ocr(preds, ocr_conf, recognized_text, track_id)

                            current_time = datetime.datetime.now().strftime("%Y%m%d")
                            
                            if track_id != track_id_flag:
                                save_infor_file(f"./save_file_txt/{current_time}.txt", [(track_id, ocr_resc, rec_conf)])
                            else:
                                continue
                            track_id_flag = track_id

                            txt = str(track_id) + ": " + str(ocr_resc) + '-' + str(rec_conf)
                            # txt = str(track_id) + ": "
                            # Vẽ bounding box và track id 
                            cv2.rectangle(overlay_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(overlay_img, txt, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                # cv2.putText(img, str(rec_conf), (int(x1) + 150, int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)   
                                            
                    except Exception as e:
                        continue
                else:
                    # If there are no bounding boxes or the result is empty, skip this result
                    continue
            
            tot_time = time.time() - prev_time
            fps = round(1 / tot_time,2)

            if w < 2000:
                size = 1
            else:
                size = 2
            
            cv2.putText(overlay_img, 'frame: %d fps: %s' % (ct, int(fps)),
                        (0, int(100 * 1)), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 255), thickness=2)
            out.write(overlay_img)
            # Increasing frame count.
            ct = ct + 1
        else:
            break

def tracker_test_vid_with_yolo_track(vid_dir,out_path):
    # Declaring variables for video processing.
    cap = cv2.VideoCapture(vid_dir)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #   file_name = os.path.join(out_path, 'out_' + vid_dir.split('/')[-1])
    
    out = cv2.VideoWriter(out_path, codec, fps, (width, height))

    # Lưu trữ track
    track_history = defaultdict(lambda: [])

    preds = []
    # Initializing some helper variables.
    ct = 0
    total_obj = 0
    alpha = 0.5

    # Reading video frame by frame.
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret:
            h, w = img.shape[:2]
            print(ct)
            
            w_scale = w/1.55
            h_scale = h/17
    
            # Method to blend two images, here used to make the information box transparent.
            overlay_img = img.copy()
            cv2.rectangle(img, (int(w_scale), 0), (w, int(h_scale*3.4)), (0,0,0), -1)
            cv2.addWeighted(img, alpha, overlay_img, 1 - alpha, 0, overlay_img)
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
            
            prev_time = time.time()

            results = model.track(img, persist=True)
            boxes = results[0].boxes.xyxy.cpu()
            boxes_ct = results[0].boxes.xywh.cpu()
            try:
                track_ids = results[0].boxes.id.int().cpu().tolist()

            except Exception as e:
                continue

            # print("Bbox: ", boxes)
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0: 
                    try:
                        for box, box_ct ,track_id in zip(boxes, boxes_ct, track_ids):
                            x1, y1, x2, y2 = box
                            x, y ,w, h = box_ct

                            track = track_history[track_id]
                            track.append((float(x), float(y)))  # x, y center point
                            if len(track) > 30:  # retain 90 tracks for 90 frames
                                track.pop(0)

                            width = w
                            height = h
                            aspect_ratio = width / height

                            if 0 <= aspect_ratio <= 1.5:
                                # Cắt và xoay lại biển số xe --------------
                                image_upper, image_lower = rotate_and_split_license_plate(img[int(y1):int(y2), int(x1):int(x2)])

                                if image_upper is None and image_lower is None:
                                    # Biển số xe gần vuông hoặc hình vuông

                                    # Tính toán điểm chia ảnh thành hai phần trên và dưới
                                    split_point = y1 + (y2 - y1) // 2

                                    # Tạo hai phần ảnh con từ ảnh cr_img
                                    upper_part = img[int(y1):int(split_point), int(x1):int(x2)]
                                    lower_part = img[int(split_point):int(y2), int(x1):int(x2)]

                                    image_upper=cv2.resize(upper_part,(int(width),int(height/2)))
                                    image_lower=cv2.resize(lower_part,(int(width),int(height/2)))

                                image_collage_horizontal =np.hstack([image_upper, image_lower])
                                image_filename = str(ct) + ".jpg"  # Tên file ảnh đầu ra
                                cv2.imwrite("results/crop_image/" + image_filename, image_collage_horizontal)

                                # Tiếp tục xử lý ảnh chữ nhật cr_img ở đây
                                ocr_res = perform_ocr(image_collage_horizontal)
                                print('text:', ocr_res)

                                # Lưu hai phần ảnh
                                # cv2.imwrite("results/upper_part.jpg", upper_part)
                                # cv2.imwrite("results/" + str(ct) + ".jpg", lower_part)
                            else:
                                # Biển số xe không gần vuông
                                # Xử lý ảnh bình thường ở đây (cr_img = img[int(y1):int(y2), int(x1):int(x2)])
                                cr_img = img[int(y1):int(y2), int(x1):int(x2)]
                                image_filename = str(ct) + ".jpg"  # Tên file ảnh đầu ra
                                cv2.imwrite("results/crop_image/" + image_filename, cr_img)
                                ocr_res = perform_ocr(cr_img)
                                print('text:', ocr_res)

                            print('-------------------')
                            # Draw bounding box and put text
                            recognized_text = ocr_res[0][0][0] if ocr_res else "No Text"
                            print("recognized_text: ",recognized_text)

                            ocr_conf = ocr_res[0][0][1] if ocr_res else "No Conf"
                            ocr_conf = round(ocr_conf,3)
                            print("ocr_conf: ",ocr_conf)

                            output_frame = {"track_id": track_id, "ocr_txt": recognized_text, "ocr_conf": ocr_conf}

                            if track_id not in list(set(ele['track_id'] for ele in preds)):
                                    total_obj = total_obj + 1
                                    preds.append(output_frame)
                                # Looking for the current track in the list and updating the highest confidence of it.
                            else:
                                preds, rec_conf, ocr_resc = get_best_ocr(preds, ocr_conf, recognized_text, track_id) 

                            txt = str(track_id) + ": " + str(ocr_resc) + '-' + str(rec_conf)
                            cv2.rectangle(overlay_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red bounding box
                            cv2.putText(overlay_img, txt, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    except Exception as e:
                        continue
                else:
                    continue
            
            tot_time = time.time() - prev_time
            fps = round(1 / tot_time,2)

            if w < 2000:
                size = 1
            else:
                size = 2
            
            cv2.putText(overlay_img, 'frame: %d fps: %s' % (ct, int(fps)),
                        (0, int(100 * 1)), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 255), thickness=2)
            out.write(overlay_img)
            # Increasing frame count.
            ct = ct + 1
        else:
            break

input_dir = './test_video/test_1.mp4'

out_path = './results/test_1.mp4'
tracker_test_vid_with_yolo_track(input_dir, out_path)