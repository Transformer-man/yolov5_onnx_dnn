from yolov5_dnn import mult_test

if __name__ == "__main__":

    onnx_path = r'.\weights\yolov5n.onnx'
    input_path = r'./input_image'
    save_path = r'./output_image'

    #video=True代表开启摄像头
    mult_test(onnx_path, input_path, save_path, video=False)
