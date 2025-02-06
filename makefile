# Saideep Arikontham
# Feb 2025
# CS 5330 OpenCV tutorial


# Compiler
CC = /usr/bin/g++
CXX = $(CC)

# Compiler flags
CFLAGS = -std=c++17 -Wall -g \
         -I/opt/homebrew/Cellar/opencv/4.10.0_18/include/opencv4 \
         -I/opt/homebrew/Cellar/opencv/4.10.0_18/include \
         -I/opt/homebrew/Cellar/onnxruntime-osx-arm64-1.20.1/include/core \
         -I/opt/homebrew/Cellar/onnxruntime-osx-arm64-1.20.1/include

CXXFLAGS = $(CFLAGS)

# Library paths
LDFLAGS = -L/opt/homebrew/Cellar/opencv/4.10.0_18/lib \
          -L/opt/homebrew/Cellar/onnxruntime-osx-arm64-1.20.1/lib \
          -Wl,-rpath,/opt/homebrew/Cellar/onnxruntime-osx-arm64-1.20.1/lib

# OpenCV libraries
LDLIBS = -lopencv_gapi -lopencv_stitching -lopencv_alphamat -lopencv_aruco \
         -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect \
         -lopencv_dnn_superres -lopencv_dpm -lopencv_face -lopencv_freetype \
         -lopencv_fuzzy -lopencv_hfs -lopencv_img_hash -lopencv_intensity_transform \
         -lopencv_line_descriptor -lopencv_mcc -lopencv_quality -lopencv_rapid \
         -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_sfm -lopencv_signal \
         -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping \
         -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking \
         -lopencv_highgui -lopencv_datasets -lopencv_text -lopencv_plot \
         -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_wechat_qrcode \
         -lopencv_xfeatures2d -lopencv_shape -lopencv_ml -lopencv_ximgproc \
         -lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d \
         -lopencv_imgcodecs -lopencv_features2d -lopencv_dnn -lopencv_flann \
         -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core \
         -lonnxruntime

# Output directory
BINDIR = ./bin

# Source directory
SRCDIR = ./src


# Compile and link vidDisplay with filters
generate_feature_vectors: $(SRCDIR)/generate_feature_vectors.o $(SRCDIR)/csv_util.o $(SRCDIR)/utils.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

$(SRCDIR)/generate_feature_vectors.o: $(SRCDIR)/generate_feature_vectors.cpp $(SRCDIR)/csv_util.h $(SRCDIR)/utils.h
	$(CC) $(CXXFLAGS) -c $< -o $@


# Compile and link vidDisplay with filters
pattern_recognition: $(SRCDIR)/pattern_recognition.o $(SRCDIR)/csv_util.o $(SRCDIR)/utils.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

$(SRCDIR)/pattern_recognition.o: $(SRCDIR)/pattern_recognition.cpp $(SRCDIR)/csv_util.h $(SRCDIR)/utils.h
	$(CC) $(CXXFLAGS) -c $< -o $@

# Compile and link vidDisplay with filters
test: $(SRCDIR)/test.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

$(SRCDIR)/test.o: $(SRCDIR)/test.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@


# Compile utils
$(SRCDIR)/csv_util.o: $(SRCDIR)/csv_util.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

$(SRCDIR)/utils.o: $(SRCDIR)/utils.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@


clean:
	rm -f $(SRCDIR)/*.o *~ $(BINDIR)/vidDisplay \
	rm -f $(SRCDIR)/*.o *~ $(BINDIR)/imgDisplay \
    rm -f $(SRCDIR)/*.o *~ ./feature_vectors/feature*.csv
