// -------------------------------------------------------------------------------------------
// Saideep Arikontham
// Feb 2025
// Pattern Recognition and Computer Vision - Project 2
// -------------------------------------------------------------------------------------------




#include <cstdlib>
#include <dirent.h>
#include <cstdio> // gives printf
#include <cstring> // gives strcpy
#include <opencv2/opencv.hpp> // openCV
#include "csv_util.h" // CSV utility functions
#include "DA2Network.hpp"



// -------------------------------------------------------------------------------------------
// Functions from previous project
// -------------------------------------------------------------------------------------------

int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
    // separable filter
    int horizontal[3] = {-1, 0, 1};
    int vertical[3] = {1, 2, 1};
    cv::Mat temp;

    // copying src size to dst and temp with 16 bit signed 3 channel
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    temp = cv::Mat::zeros(src.size(), CV_16SC3);

    // Horizontal
    for (int i=0; i<src.rows; i++){
        cv::Vec3b *src_ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *temp_ptr = temp.ptr<cv::Vec3s>(i);
        for (int j=1; j<src.cols-1; j++){
            int blue = 0, green = 0, red = 0;

            blue = src_ptr[j-1][0]*horizontal[0] + src_ptr[j][0]*horizontal[1] + src_ptr[j+1][0]*horizontal[2];
            green = src_ptr[j-1][1]*horizontal[0] + src_ptr[j][1]*horizontal[1] + src_ptr[j+1][1]*horizontal[2];
            red = src_ptr[j-1][2]*horizontal[0] + src_ptr[j][2]*horizontal[1] + src_ptr[j+1][2]*horizontal[2];

            temp_ptr[j][0] = blue;
            temp_ptr[j][1] = green;
            temp_ptr[j][2] = red;
        }
    }

    // Vertical
    for (int i=1; i<src.rows-1; i++){
        for (int j=0; j<src.cols; j++){
            int blue = 0, green = 0, red = 0;

            blue = temp.at<cv::Vec3s>(i - 1, j)[0] * vertical[0] + temp.at<cv::Vec3s>(i, j)[0] * vertical[1] + temp.at<cv::Vec3s>(i + 1, j)[0] * vertical[2];
            green = temp.at<cv::Vec3s>(i - 1, j)[1] * vertical[0] + temp.at<cv::Vec3s>(i, j)[1] * vertical[1] + temp.at<cv::Vec3s>(i + 1, j)[1] * vertical[2];
            red = temp.at<cv::Vec3s>(i - 1, j)[2] * vertical[0] + temp.at<cv::Vec3s>(i, j)[2] * vertical[1] + temp.at<cv::Vec3s>(i + 1, j)[2] * vertical[2];

            dst.at<cv::Vec3s>(i, j)[0] = blue;
            dst.at<cv::Vec3s>(i, j)[1] = green;
            dst.at<cv::Vec3s>(i, j)[2] = red;
        }
    }

    return 0;
}


int sobelY3x3( cv::Mat &src, cv::Mat &dst ) {
    // Separable filter
    int horizontal[3] = {1, 2, 1};
    int vertical[3] = {1, 0, -1};
    cv::Mat temp;

    // Initialize temp and dst with CV_16SC3 (16-bit signed, 3 channels)
    dst = cv::Mat::zeros(src.size(), CV_16SC3);
    temp = cv::Mat::zeros(src.size(), CV_16SC3);

    // Horizontal pass
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *src_ptr = src.ptr<cv::Vec3b>(i);
        cv::Vec3s *temp_ptr = temp.ptr<cv::Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            int blue = 0, green = 0, red = 0;

            blue = src_ptr[j - 1][0] * horizontal[0] + src_ptr[j][0] * horizontal[1] + src_ptr[j + 1][0] * horizontal[2];
            green = src_ptr[j - 1][1] * horizontal[0] + src_ptr[j][1] * horizontal[1] + src_ptr[j + 1][1] * horizontal[2];
            red = src_ptr[j - 1][2] * horizontal[0] + src_ptr[j][2] * horizontal[1] + src_ptr[j + 1][2] * horizontal[2];

            temp_ptr[j][0] = blue;
            temp_ptr[j][1] = green;
            temp_ptr[j][2] = red;
        }
    }

    // Vertical pass
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 0; j < src.cols; j++) {
            int blue = 0, green = 0, red = 0;

            blue = temp.at<cv::Vec3s>(i - 1, j)[0] * vertical[0] + temp.at<cv::Vec3s>(i, j)[0] * vertical[1] + temp.at<cv::Vec3s>(i + 1, j)[0] * vertical[2];
            green = temp.at<cv::Vec3s>(i - 1, j)[1] * vertical[0] + temp.at<cv::Vec3s>(i, j)[1] * vertical[1] + temp.at<cv::Vec3s>(i + 1, j)[1] * vertical[2];
            red = temp.at<cv::Vec3s>(i - 1, j)[2] * vertical[0] + temp.at<cv::Vec3s>(i, j)[2] * vertical[1] + temp.at<cv::Vec3s>(i + 1, j)[2] * vertical[2];


            dst.at<cv::Vec3s>(i, j)[0] = blue;
            dst.at<cv::Vec3s>(i, j)[1] = green;
            dst.at<cv::Vec3s>(i, j)[2] = red;
        }
    }

    return 0;
}


int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {

    dst = cv::Mat::zeros(sx.size(), CV_8UC3);

    // Compute gradient magnitude for each channel
    for (int i = 0; i < sx.rows; i++) {
        cv::Vec3s *sx_ptr = sx.ptr<cv::Vec3s>(i);
        cv::Vec3s *sy_ptr = sy.ptr<cv::Vec3s>(i);
        cv::Vec3b *dst_ptr = dst.ptr<cv::Vec3b>(i);

        for (int j = 0; j < sx.cols; j++) {
            float blue_mag = std::sqrt(sx_ptr[j][0] * sx_ptr[j][0] + sy_ptr[j][0] * sy_ptr[j][0]);
            float green_mag = std::sqrt(sx_ptr[j][1] * sx_ptr[j][1] + sy_ptr[j][1] * sy_ptr[j][1]);
            float red_mag = std::sqrt(sx_ptr[j][2] * sx_ptr[j][2] + sy_ptr[j][2] * sy_ptr[j][2]);

            dst_ptr[j][0] = std::min(255.0f, blue_mag);
            dst_ptr[j][1] = std::min(255.0f, green_mag);
            dst_ptr[j][2] = std::min(255.0f, red_mag);
        }
    }

    return 0; 
}


// -------------------------------------------------------------------------------------------
// Functions to read and display images
// -------------------------------------------------------------------------------------------

int read_image(char img_file[256], cv::Mat &src){
    src = cv::imread(img_file);
    cv::resize(src, src, cv::Size(640, 512));

    if (src.data == NULL){
        printf("error: unable to read image %s\n", img_file);
        return -1;
    }
    return 0;
}


int display_image(std::string display_name, cv::Mat &img){
    cv::Mat temp;
    img.copyTo(temp);

    cv::resize(temp, temp, cv::Size(350, 325));
    cv::imshow(display_name, temp);
    return 0;
}


int loop_until_q_pressed(){
    int key_pressed;
    // Entering loop to wait for a key press - "q" to quit
    while(1){
        key_pressed = cv::waitKey(0); // returns ASCII for pressed key
        if(key_pressed == 113 || key_pressed == 81){ // ASCII for 'q' (113) and 'Q' (81)
            printf("key pressed: %c, terminating\n", static_cast<char>(key_pressed));
            exit(0); // exit the loop and terminate the program
        } 
        else{
            printf("key pressed: %c, continuing\n", static_cast<char>(key_pressed));
        }
    }
}


int get_top_N_matches(const std::vector<char*>& filenames, const std::vector<float>& ssd_data, int N, bool ascending=true) {
    // Create a vector of pairs (SSD value, filename)
    std::vector<std::pair<float, char*>> ssd_pairs;
    for (size_t i = 0; i < filenames.size(); i++) {
        ssd_pairs.emplace_back(ssd_data[i], filenames[i]);
    }

    // Sort the pairs based on SSD values (ascending order)
    std::sort(ssd_pairs.begin(), ssd_pairs.end());

    // Step 3: Retrieve the top N matches
    if(ascending){
        for (int i = 1; i < N+1 && i < ssd_pairs.size(); i++) {
            // std::cout << "Filename: " << ssd_pairs[i].second << ", SSD: " << ssd_pairs[i].first << "\n";
            std::string display_name = "Top " + std::to_string(i) + " Match: " + ssd_pairs[i].second;

            cv::Mat src;
            read_image(ssd_pairs[i].second, src);
            display_image(display_name, src);
        }
    }
    else{
        for (int i = ssd_pairs.size()-2; i > ssd_pairs.size()-N-2 && i > 0; i--) {
            // std::cout << "Filename: " << ssd_pairs[i].second << ", SSD: " << ssd_pairs[i].first << "\n";
            std::string display_name = "Top " + std::to_string(ssd_pairs.size()-i-1) + " Match: " + ssd_pairs[i].second;

            cv::Mat src;
            read_image(ssd_pairs[i].second, src);
            display_image(display_name, src);
        }
    }

    return 0;
}


// -------------------------------------------------------------------------------------------
// Functions to get image histogram
// -------------------------------------------------------------------------------------------


int get_rg_histogram(cv::Mat &src, cv::Mat &hist, const int histsize) {

    // Initialize the histogram (use floats so we can make probabilities)
    hist = cv::Mat::zeros(cv::Size(histsize, histsize), CV_32FC1);

    // Loop over all pixels
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i); // Pointer to row i
        for (int j = 0; j < src.cols; j++) {
            // Get the RGB values
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];

            // Compute the r, g chromaticity
            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0; // Check for all zeros
            float r = R / divisor;
            float g = G / divisor;

            // Compute indexes, r, g are in [0, 1]
            int rindex = (int)(r * (histsize - 1) + 0.5);
            int gindex = (int)(g * (histsize - 1) + 0.5);

            // Increment the histogram
            hist.at<float>(rindex, gindex)++;
        }
    }

    // Histogram is complete
    // Normalize the histogram by the number of pixels
    hist /= (src.rows * src.cols); // Divides all elements of a cv::Mat by a scalar

    // The chromaticity histogram is ready for saving as a feature vector now
    return 0;
}


int get_rgb_histogram(cv::Mat &src, cv::MatND &hist, const int histsize) {
    // Define histogram size (3D)
    int histSize[] = {histsize, histsize, histsize};

    // Initialize a true 3D histogram using cv::MatND
    hist = cv::MatND(3, histSize, CV_32FC1, cv::Scalar(0));

    // Loop over all pixels manually
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i); // Pointer to row i
        for (int j = 0; j < src.cols; j++) {
            // Extract R, G, B values
            int B = ptr[j][0];
            int G = ptr[j][1];
            int R = ptr[j][2];

            // Compute the rgb
            float divisor = R + G + B;
            divisor = divisor > 0.0 ? divisor : 1.0; // Check for all zeros

            float r = R / divisor;
            float g = G / divisor;
            float b = B / divisor;

            // Compute bin indices
            int rindex = (int)(r * (histsize - 1) + 0.5);
            int gindex = (int)(g * (histsize - 1) + 0.5);
            int bindex = (int)(b * (histsize - 1) + 0.5);

            // Increment histogram directly at (r, g, b)
            hist.at<float>(rindex, gindex, bindex)++;
        }
    }

    // Normalize the histogram so the sum is 1
    hist /= (src.rows * src.cols);

    return 0;
}


int get_texture_histogram(cv::Mat &src, cv::MatND &hist, const int histsize)
{
    // Compute the Sobel gradients along the x and y axes.
    cv::Mat grad_x, grad_y;
    sobelX3x3( src, grad_x );
    sobelY3x3( src, grad_y );

    // Compute the gradient magnitude.
    cv::Mat dst;
    magnitude(grad_x, grad_y, dst);

    get_rgb_histogram(dst, hist, 8);

    return 0;
}

int get_depth(cv::Mat &src, cv::Mat &dst){
    // make a DANetwork object, if you use a different network, you have
  // to include the input and output layer names
  DA2Network da_net( "./src/model_fp16.onnx" );

  // scale the network input so it's not larger than 512 on the small side
  float scale_factor = 512.0 / (src.rows > src.cols ? src.cols : src.rows);
  scale_factor = scale_factor > 1.0 ? 1.0 : scale_factor;

  // set up the network input
  da_net.set_input( src, scale_factor );

  // run the network
  da_net.run_network( dst, src.size() );

  return 0;
}


int get_depth_rgb_histogram(cv::Mat &src, cv::MatND &hist, const int histsize) {
    // Compute the depth map
    cv::Mat depth;
    get_depth(src, depth);

    // Normalize depth to 0-255
    cv::normalize(depth, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::threshold(depth, depth, 150, 255, cv::THRESH_BINARY);

    // Define histogram size (3D)
    int histSize[] = {histsize, histsize, histsize};

    // Initialize a true 3D histogram using cv::MatND
    hist = cv::MatND(3, histSize, CV_32FC1, cv::Scalar(0));

    // Loop over all pixels manually
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);    // Pointer to row i (RGB)
        uchar *depth_ptr = depth.ptr<uchar>(i);   // Pointer to row i (Depth)

        for (int j = 0; j < src.cols; j++) {
            if (depth_ptr[j] > 150){
                // Extract R, G, B values
                int B = ptr[j][0];
                int G = ptr[j][1];
                int R = ptr[j][2];

                // Compute normalized RGB values
                float divisor = R + G + B;
                divisor = divisor > 0.0 ? divisor : 1.0; // Avoid division by zero

                float r = R / divisor;
                float g = G / divisor;
                float b = B / divisor;

                // Compute bin indices
                int rindex = static_cast<int>(r * (histsize - 1) + 0.5);
                int gindex = static_cast<int>(g * (histsize - 1) + 0.5);
                int bindex = static_cast<int>(b * (histsize - 1) + 0.5);

                // Increment histogram directly at (r, g, b)
                hist.at<float>(rindex, gindex, bindex)++;
            }
        }
    }

    // Normalize the histogram so the sum is 1
    hist /= (src.rows * src.cols);

    return 0;
}


int get_embedding( cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net) {
  const int ORNet_size = 224;
  cv::Mat blob;

  // have the function do the ImageNet mean and SD normalization
  // the function also scales the image to 224 x 224
  cv::dnn::blobFromImage( src, // input image
			  blob, // output array
			  (1.0/255.0) * (1/0.226), // scale factor
			  cv::Size( ORNet_size, ORNet_size ), // resize the image to this
			  cv::Scalar( 124, 116, 104),  // subtract mean prior to scaling
			  true,   // swapRB
			  false,  // center crop after scaling short side to size
			  CV_32F ); // output depth/type

  net.setInput( blob );
  embedding = net.forward( "onnx_node!resnetv22_flatten0_reshape0" ); // the name of the embedding layer to grab

  return(0);
}


// -------------------------------------------------------------------------------------------
// Functions to get feature vectors for an image
// -------------------------------------------------------------------------------------------


int get_feature_vector_7x7(char img_file[256], std::vector<float> &img_data){
    cv::Mat src;
    read_image(img_file, src);

    int center_x = src.rows / 2;
    int center_y = src.cols / 2;

    // Extract 7x7 RGB patch centered in the image
    for (int i = center_x - 3; i <= center_x + 3; i++) {
        for (int j = center_y - 3; j <= center_y + 3; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j); // Get BGR pixel values
            img_data.push_back(pixel[0]);
            img_data.push_back(pixel[1]);
            img_data.push_back(pixel[2]);
        }
    }
    return 0;
}


int get_feature_vector_rg_hist(char img_file[256], std::vector<float> &img_data){
    cv::Mat src;
    read_image(img_file, src);

    cv::Mat hist;
    get_rg_histogram(src, hist, 16);

    // Extract upper triangular part of the histogram
    for (int i = 0; i < hist.rows; i++) {
        for (int j = 0; j < hist.cols; j++) {
            img_data.push_back(hist.at<float>(i, j));
        }
    }
    return 0;
}


int get_feature_vector_two_rgb_hist(char img_file[256], std::vector<float> &img_data) {
    cv::Mat src;
    read_image(img_file, src);

    cv::MatND hist1, hist2;
    int histsize = 8;

    // Compute the half width and half height 320 256
    cv::Mat part2 = src(cv::Rect(160, 128, 320, 256));


    get_rgb_histogram(src, hist1, histsize); // Compute 3D histogram
    get_rgb_histogram(part2, hist2, histsize); 

    // Flatten 3D histograms into the feature vector (equally weighted)
    for (int r = 0; r < histsize; r++) {
        for (int g = 0; g < histsize; g++) {
            for (int b = 0; b < histsize; b++) {
                img_data.push_back(hist1.at<float>(r, g, b)); // Push top-half histogram values
            }
        }
    }

    for (int r = 0; r < histsize; r++) {
        for (int g = 0; g < histsize; g++) {
            for (int b = 0; b < histsize; b++) {
                img_data.push_back(hist2.at<float>(r, g, b)); // Push top-half histogram values
            }
        }
    }

    return 0;
}


int get_feature_vector_rgb_texture_hist(char img_file[256], std::vector<float> &img_data) {
    cv::Mat src;
    read_image(img_file, src);

    cv::MatND hist1, hist2;
    int histsize = 8;

    get_rgb_histogram(src, hist1, histsize); // Compute 3D histogram
    get_texture_histogram(src, hist2, histsize); 

    // Flatten 3D histograms into the feature vector (equally weighted)
    for (int r = 0; r < histsize; r++) {
        for (int g = 0; g < histsize; g++) {
            for (int b = 0; b < histsize; b++) {
                img_data.push_back(hist1.at<float>(r, g, b)); // Push top-half histogram values
            }
        }
    }

    for (int r = 0; r < histsize; r++) {
        for (int g = 0; g < histsize; g++) {
            for (int b = 0; b < histsize; b++) {
                img_data.push_back(hist2.at<float>(r, g, b)); // Push top-half histogram values
            }
        }
    }

    return 0;
}


int get_feature_vector_depth_rgb_hist(char img_file[256], std::vector<float> &img_data) {
    cv::Mat src;
    read_image(img_file, src);

    cv::MatND hist1, hist2;
    int histsize = 8;

    get_depth_rgb_histogram(src, hist1, histsize); // Compute 3D histogram
    get_texture_histogram(src, hist2, histsize); 

    // Flatten 3D histograms into the feature vector (equally weighted)
    for (int r = 0; r < histsize; r++) {
        for (int g = 0; g < histsize; g++) {
            for (int b = 0; b < histsize; b++) {
                img_data.push_back(hist1.at<float>(r, g, b)); // Push top-half histogram values
            }
        }
    }

    return 0;
}


int get_feature_vector_dnn(char img_file[256], std::vector<float> &img_data) {
    cv::Mat src;
    read_image(img_file, src);

    char mod_filename[256];
    strncpy(mod_filename, "./src/resnet18-v2-7.onnx", 255);
    cv::dnn::Net net = cv::dnn::readNet( mod_filename );
    
    cv::Mat embedding;
    get_embedding( src, embedding, net);

    // Flatten 3D histograms into the feature vector (equally weighted)
    for (int i = 0; i < embedding.rows; i++) {
        for (int j = 0; j < embedding.cols; j++) {
                img_data.push_back(embedding.at<float>(i,j)); // Push top-half histogram values
        }
    }

    return 0;
}


// -------------------------------------------------------------------------------------------
// Functions to write feature vectors file for a directory
// -------------------------------------------------------------------------------------------


// Define function pointer type for the feature vector function
typedef int (*FeatureVectorWriter)(char*, std::vector<float>&);

int write_feature_vectors(char dirname[256], char feature_file[256], FeatureVectorWriter write_feature_vector) {
    char buffer[256];
    DIR *dirp;
    struct dirent *dp;
    int counter = 0;

    printf("Processing directory %s\n", dirname);

    // open the directory
    dirp = opendir(dirname);
    if (dirp == NULL) {
        printf("Cannot open directory %s\n", dirname);
        exit(-1);
    }

    if (remove(feature_file) == 0) {
        printf("Deleted existing %s\n", feature_file);
    } else {
        //std::perror("");
        printf("%s does not exist\n", feature_file);
    }


    // loop over all the files in the image file listing
    while ((dp = readdir(dirp)) != NULL) {
        // check if the file is an image
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif")) {
            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            // printf("Full path name: %s\n", buffer);
            
            // Call the function pointer to process the image
            std::vector<float> img_data;
            write_feature_vector(buffer, img_data);
            append_image_data_csv( feature_file, buffer, img_data);

            counter++;
        }
    }
    printf("Processed %d files\n", counter);
    
    closedir(dirp);
    return 0;
}


// -------------------------------------------------------------------------------------------
// Image histogram evaluation metrics
// -------------------------------------------------------------------------------------------

float sum_of_squared_difference(std::vector<float> &img_data1, std::vector<float> &img_data2){
    float ssd = 0.0f;

    if(img_data1.size() != img_data2.size()){
        printf("error: feature vectors are not the same size\n");
        return -1;
    }

    for (int i = 0; i < img_data1.size(); i++){
        ssd = ssd + (img_data1[i] - img_data2[i]) * (img_data1[i] - img_data2[i]);
    }
    return ssd;
}


float histogram_intersection(std::vector<float> &img_data1, std::vector<float> &img_data2){
    float intersection = 0.0f;

    if(img_data1.size() != img_data2.size()){
        printf("error: feature vectors are not the same size\n");
        return -1;
    }

    for (int i = 0; i < img_data1.size(); i++){
        intersection = intersection + std::min(img_data1[i], img_data2[i]);
    }
    return intersection;
}


double chiSquareDistance(std::vector<float> &img_data1, std::vector<float> &img_data2){
    double eps = 1e-10;
    if(img_data1.size() != img_data2.size()){
        printf("error: feature vectors are not the same size\n");
        return -1;
    }

    double distance = 0.0;
    for (int i = 0; i < img_data1.size(); i++) {
        float a = img_data1[i];
        float b = img_data2[i];
        if (a + b > eps)
            distance += ((a - b) * (a - b)) / (a + b + eps);
    }
    return distance;
}


double cosine_similarity(std::vector<float>& A, std::vector<float>& B) {
    double dot = 0.0;
    double normA = 0.0, normB = 0.0;
    
    for (size_t i = 0; i < A.size(); ++i) {
        dot   += A[i] * B[i];
        normA += A[i] * A[i];
        normB += B[i] * B[i];
    }
    
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}


double computeCosineDistance(std::vector<float>& img_data1, std::vector<float>& img_data2) {
    /* Cosine distance is defined as 1.0 - cos_theta. */
    if (img_data1.size() != img_data2.size()) {
        std::cerr << "Error: Vectors are not the same size!" << std::endl;
        return -1.0;
    }

     cv::normalize(img_data1, img_data1, 1.0, 0.0, cv::NORM_L2);
    cv::normalize(img_data2, img_data2, 1.0, 0.0, cv::NORM_L2);   
    
    // For unit-length vectors, cosine similarity is simply their dot product.
    double cos_sim = cosine_similarity(img_data1, img_data2);
    
    // Cosine distance is defined as 1 - cosine similarity.
    return 1.0 - cos_sim;
}

// -------------------------------------------------------------------------------------------
// End
// -------------------------------------------------------------------------------------------
