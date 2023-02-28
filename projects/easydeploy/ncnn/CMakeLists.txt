cmake_minimum_required(VERSION 3.4.1)

option(USE_T7		"USE T7"            OFF)
option(USE_MNN      "USE MNN"           OFF)
option(USE_NCNN     "USE NCNN"          ON)
option(USE_LINUX	"USE LINUX"         ON)
option(USE_ROCKCHIP "USE ROCKCHIP"      OFF)

message("==========================================================")
message("-- option:")
message("-- USE_T7                               is ${USE_T7}")
message("-- USE_MNN                              is ${USE_MNN}")
message("-- USE_NCNN                             is ${USE_NCNN}")
message("-- USE_LINUX                            is ${USE_LINUX}")
message("-- USE_ROCKCHIP                         is ${USE_ROCKCHIP}")
message("==========================================================")

project(HOD)

# set(CMAKE_BUILD_TYPE DEBUG)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set(libraries )

# windows
if (CMAKE_HOST_WIN32)
	#opencv
	set(OPENCV_DIR "E:/MyDrivers/software/opencv/build")
	include_directories(${OPENCV_DIR}/include/)
	link_directories(${OPENCV_DIR}/x64/vc15/lib/)
	set(OpenCV_LIBS opencv_world3410.lib)
	# set(OpenCV_LIBS opencv_world3410d.lib)
	#ncnn
	set(NCNN_DIR "E:/MyDrivers/software/cnn_deploy/ncnn/install")
	include_directories(${NCNN_DIR}/include/)
	link_directories(${NCNN_DIR}/lib/)
	set(NCNN_LIBS ncnn.lib)
	# set(NCNN_LIBS ncnnd.lib)
# linux
elseif (CMAKE_HOST_UNIX)
    set(CMAKE_CXX_FLAGS "-std=c++11 -lpthread -fopenmp")
    # OPENCV
    set(OPENCV_DIR "/home/ckn/library/opencv-3.4.16/build_linux/install")
    # MNN
    set(MNN_DIR "/home/ckn/library/MNN")
    include_directories(${MNN_DIR}/include)
    link_directories(${MNN_DIR}/build/)
    # NCNN
    set(NCNN_DIR "/home/ckn/library/ncnn/build/install")

    include_directories(${OPENCV_DIR}/include)
    link_directories(${OPENCV_DIR}/lib)
    set(OpenCV_LIBS libopencv_calib3d.so
        #libopencv_contrib.so
        libopencv_core.so
        libopencv_features2d.so
        libopencv_flann.so
        #libopencv_gpu.so
        libopencv_highgui.so
        libopencv_imgproc.so
        #libopencv_legacy.so
        libopencv_ml.so
        libopencv_objdetect.so
        #libopencv_ocl.so
        libopencv_photo.so
        libopencv_stitching.so
        # libopencv_superres.so
        libopencv_video.so
        libopencv_videoio.so
        libopencv_imgcodecs.so
        # libopencv_videostab.so
        # libopencv_nonfree.so
        )

    include_directories(${NCNN_DIR}/include)
    link_directories(${NCNN_DIR}/lib/)
    set(NCNN_LIBS libncnn.a)
    set(MNN_LIBS libMNN.so)
endif()

add_definitions(-DUSE_MNN)
add_definitions(-DUSE_NCNN)


add_executable(HOD-RTMDET ./mmyolo_rtmdet.cpp)
target_link_libraries(HOD-RTMDET ${libraries} ${MNN_LIBS} ${NCNN_LIBS} ${OpenCV_LIBS})