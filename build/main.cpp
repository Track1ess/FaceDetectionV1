#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>      // for std::system
#include <string>
#include <thread>
#include <chrono>

int main(int argc, char** argv) {
    int cam_index = 0;
    if (argc >= 2) {
        cam_index = std::stoi(argv[1]);
    }

    cv::VideoCapture cap(cam_index);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera index " << cam_index << std::endl;
        return 1;
    }

    // Optional: set resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::cout << "Camera opened on index " << cam_index << std::endl;
    std::cout << "Press 'c' to capture and run Python MobileFaceNet." << std::endl;
    std::cout << "Press 'q' to quit." << std::endl;

    cv::Mat frame;
    const std::string window_name = "Camera (C++ capture)";
    const std::string saved_path  = "capturedFace.jpg";   // relative to current directory

    while (true) {
        if (!cap.read(frame)) {
            std::cerr << "Failed to grab frame" << std::endl;
            break;
        }

        cv::imshow(window_name, frame);
        char key = static_cast<char>(cv::waitKey(1) & 0xFF);

        if (key == 'q') {
            break;
        } else if (key == 'c') {
            // Here you can add your own face detection / cropping.
            // For now, we just save the full frame.
            if (!cv::imwrite(saved_path, frame)) {
                std::cerr << "Error: Failed to save image to " << saved_path << std::endl;
                continue;
            }

            std::cout << "Saved image to: " << saved_path << std::endl;

            // std::this_thread::sleep_for(std::chrono::seconds(10));

            // Build command to call Python script.
            // Assumes you're running this binary inside your conda env (facenet),
            // and mobilefacenet_infer.py is in the SAME directory as the executable.
            std::string command = "python3 mobilefacenet_infer.py ";

            std::cout << "Running Python command: " << command << std::endl;
            int ret = std::system(command.c_str());
            if (ret != 0) {
                std::cerr << "Python script returned non-zero exit code: " << ret << std::endl;
            }
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
